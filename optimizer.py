"""Optimizer — применяет fusion к графу, слою или модели."""
import numpy as np
from ninewas9 import RoPECache, scaled_dot_product_attention

from xanosmerks.nsofi40.attention_parts import scaled_dot_product_attention_parts

from .ops import (
    RMSNormOp, LayerNormOp, LinearOp, FusedSwiGLUOp,
    FusedRMSNormLinearOp, FusedLayerNormLinearOp, FusedResidualNormOp,
)
from .fusion import apply_fusions


class OptimizedAttention:
    """Версия MultiHeadAttention с fused-norm + QKV."""

    def __init__(self, original_attn, norm_weight, norm_bias, norm_type, eps):
        self.original = original_attn
        self.n_heads = original_attn.n_heads
        self.n_kv_heads = original_attn.n_kv_heads
        self.head_dim = original_attn.head_dim
        self.hidden = original_attn.hidden
        self.kv_cache = original_attn.kv_cache
        self.rope_cache = original_attn.rope_cache
        self.norm_type = norm_type
        self.norm_eps = eps

        self.is_mla = original_attn.__class__.__name__ == "MultiHeadLatentAttention"

        has_bias = False
        if not self.is_mla:
            has_bias = any(b is not None for b in (
                original_attn.b_q, original_attn.b_k, original_attn.b_v,
            ))

        can_fuse = (
            not self.is_mla
            and not has_bias
            and getattr(original_attn, "fused_qkv", None) is not None
        )

        if can_fuse:
            fused_w = original_attn.fused_qkv.fused_weight
            if norm_type == "rmsnorm":
                self.fused_op = FusedRMSNormLinearOp(
                    norm_weight, fused_w, None, eps,
                )
            else:
                self.fused_op = FusedLayerNormLinearOp(
                    norm_weight, norm_bias, fused_w, None, eps,
                )
            self.use_fused = True
            self.q_out = original_attn.w_q.shape[0]
            self.k_out = original_attn.w_k.shape[0]
            self.v_out = original_attn.w_v.shape[0]
        else:
            self.use_fused = False
            if norm_type == "rmsnorm":
                self.norm_op = RMSNormOp(norm_weight, eps)
            else:
                self.norm_op = LayerNormOp(norm_weight, norm_bias, eps)

    def __call__(self, x, layer_idx, position_offset):
        if self.is_mla:
            h = self.norm_op.forward(x)
            return self.original(h, layer_idx, position_offset)

        batch, seq_len, _ = x.shape

        if self.use_fused:
            qkv = self.fused_op.forward(x)
            q = qkv[..., :self.q_out]
            k = qkv[..., self.q_out:self.q_out + self.k_out]
            v = qkv[..., self.q_out + self.k_out:]
        else:
            h = self.norm_op.forward(x)
            q = h @ self.original.w_q.T
            if self.original.b_q is not None:
                q = q + self.original.b_q
            k = h @ self.original.w_k.T
            if self.original.b_k is not None:
                k = k + self.original.b_k
            v = h @ self.original.w_v.T
            if self.original.b_v is not None:
                v = v + self.original.b_v

        q = q.reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.rope_cache is not None:
            cos, sin = self.rope_cache.slice_for_positions(
                position_offset, position_offset + seq_len,
            )
            q, k = RoPECache.apply(q, k, cos, sin)

        use_parts = self.kv_cache is not None and hasattr(self.kv_cache, "get_parts")

        if use_parts:
            self.kv_cache.append(layer_idx, k, v)
            sink_k, sink_v, window_k, window_v = self.kv_cache.get_parts(layer_idx)
            sink_count = sink_k.shape[2]
            window_count = window_k.shape[2]
            mask = None
            if seq_len > 1:
                mask = self._build_causal_mask_nsofi(
                    seq_len, sink_count, window_count,
                )
            attn_out = scaled_dot_product_attention_parts(
                q, sink_k, sink_v, window_k, window_v, mask=mask,
            )
        elif self.kv_cache is not None:
            self.kv_cache.append(layer_idx, k, v)
            kv_end = position_offset + seq_len
            k_full = self.kv_cache.k[layer_idx, :, :, :kv_end, :]
            v_full = self.kv_cache.v[layer_idx, :, :, :kv_end, :]
            kv_len = k_full.shape[2]
            mask = None
            if seq_len > 1:
                mask = self._build_causal_mask(seq_len, kv_len, position_offset)
            attn_out = scaled_dot_product_attention(q, k_full, v_full, mask=mask)
        else:
            mask = None
            if seq_len > 1:
                mask = self._build_causal_mask(seq_len, seq_len, 0)
            attn_out = scaled_dot_product_attention(q, k, v, mask=mask)

        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.hidden)
        out = attn_out @ self.original.w_o.T
        if self.original.b_o is not None:
            out = out + self.original.b_o
        return out

    @staticmethod
    def _build_causal_mask(seq_len, kv_len, position_offset):
        mask = np.full((seq_len, kv_len), -np.inf, dtype=np.float32)
        for i in range(seq_len):
            end = min(position_offset + i + 1, kv_len)
            mask[i, :end] = 0.0
        return mask

    @staticmethod
    def _build_causal_mask_nsofi(seq_len, sink_count, window_count):
        past_len = sink_count + window_count
        kv_len = past_len + seq_len
        mask = np.full((seq_len, kv_len), -np.inf, dtype=np.float32)
        mask[:, :past_len] = 0.0
        for i in range(seq_len):
            mask[i, past_len:past_len + i + 1] = 0.0
        return mask


class OptimizedLayer:
    """Замена TransformerLayer с fused-операциями."""

    def __init__(self, original_layer):
        self.original = original_layer
        self.layer_idx = original_layer.layer_idx
        self.norm_type = original_layer.norm_type
        self.norm_eps = original_layer.norm_eps

        self.attn = OptimizedAttention(
            original_layer.attn,
            original_layer.norm1_w,
            original_layer.norm1_b,
            self.norm_type,
            self.norm_eps,
        )

        self.use_original_mlp = False
        self.use_fused_mlp = False
        self.use_fused_residual = False
        self.fused_norm_gate_up = None
        self.inter = 0
        self.down_op = None
        self.norm2_op = None
        self.gate_op = None
        self.up_op = None
        self.swiglu_op = None

        if original_layer.moe is not None:
            self.use_original_mlp = True
            return

        if original_layer.activation not in ("silu", "swish"):
            self.use_original_mlp = True
            return

        if original_layer.fused_mlp is not None:
            fused_w = original_layer.fused_mlp.fused_gate_up.fused_weight
            inter = fused_w.shape[0] // 2
            w_down = original_layer.fused_mlp.w_down

            if self.norm_type == "rmsnorm":
                self.fused_norm_gate_up = FusedRMSNormLinearOp(
                    original_layer.norm2_w, fused_w, None, self.norm_eps,
                )
            else:
                self.fused_norm_gate_up = FusedLayerNormLinearOp(
                    original_layer.norm2_w, original_layer.norm2_b,
                    fused_w, None, self.norm_eps,
                )
            self.inter = inter
            self.down_op = LinearOp(w_down)
            self.use_fused_mlp = True
        else:
            w_gate = original_layer.w_gate
            b_gate = original_layer.b_gate
            w_up = original_layer.w_up
            b_up = original_layer.b_up
            w_down = original_layer.w_down
            b_down = original_layer.b_down

            if self.norm_type == "rmsnorm":
                self.norm2_op = FusedResidualNormOp(
                    original_layer.norm2_w, self.norm_eps,
                )
                self.use_fused_residual = True
            else:
                self.norm2_op = LayerNormOp(
                    original_layer.norm2_w, original_layer.norm2_b, self.norm_eps,
                )
                self.use_fused_residual = False

            self.gate_op = LinearOp(w_gate, b_gate)
            self.up_op = LinearOp(w_up, b_up)
            self.down_op = LinearOp(w_down, b_down)
            self.swiglu_op = FusedSwiGLUOp()

    def __call__(self, x, position_offset):
        h_attn = self.attn(x, self.layer_idx, position_offset)
        x_mid = x + h_attn

        if self.use_original_mlp:
            h_mlp = self.original._mlp(x_mid)
        elif self.use_fused_mlp:
            gate_up = self.fused_norm_gate_up.forward(x_mid)
            gate = gate_up[..., :self.inter]
            up = gate_up[..., self.inter:]
            hidden = FusedSwiGLUOp().forward(gate, up)
            h_mlp = self.down_op.forward(hidden)
        else:
            if self.use_fused_residual:
                h_norm = self.norm2_op.forward(x, h_attn)
            else:
                h_norm = self.norm2_op.forward(x_mid)
            gate = self.gate_op.forward(h_norm)
            up = self.up_op.forward(h_norm)
            hidden = self.swiglu_op.forward(gate, up)
            h_mlp = self.down_op.forward(hidden)

        x_out = x_mid + h_mlp
        return x_out

    def __repr__(self):
        return f"<OptimizedLayer idx={self.layer_idx}>"


class OptimizedModel:
    """Замена TransformerModel с оптимизированными слоями."""

    def __init__(self, original_model):
        self.config = original_model.config
        self.embed_tokens = original_model.embed_tokens
        self.layers = [OptimizedLayer(l) for l in original_model.layers]

        self.norm_type = original_model.norm_type
        self.norm_eps = original_model.norm_eps
        self.final_norm_w = original_model.final_norm_w
        self.final_norm_b = original_model.final_norm_b

        if self.norm_type == "rmsnorm":
            self.final_norm_op = RMSNormOp(self.final_norm_w, self.norm_eps)
        else:
            self.final_norm_op = LayerNormOp(
                self.final_norm_w, self.final_norm_b, self.norm_eps,
            )

        self.lm_head = original_model.lm_head
        self.lm_head_T = original_model.lm_head_T

    def __call__(self, input_ids, position_offset=0, last_only=False,
                 kv_attention_mask=None):
        if kv_attention_mask is not None:
            raise NotImplementedError(
                "OptimizedModel не поддерживает padding (kv_attention_mask). "
                "Используйте базовую TransformerModel для batch-генерации "
                "с разной длиной промптов."
            )
        x = self.embed_tokens[input_ids]
        for layer in self.layers:
            x = layer(x, position_offset)
        x = self.final_norm_op.forward(x)
        if last_only:
            return x[:, -1, :] @ self.lm_head_T
        return x @ self.lm_head_T

    def __repr__(self):
        return f"<OptimizedModel layers={len(self.layers)}>"


def optimize(graph):
    """Применяет fusion к графу. Возвращает новый Graph."""
    return apply_fusions(graph)[0]


def optimize_transformer_layer(layer):
    """Возвращает OptimizedLayer для TransformerLayer из genof11mo."""
    return OptimizedLayer(layer)


def optimize_transformer_model(model):
    """Возвращает OptimizedModel для TransformerModel из genof11mo."""
    return OptimizedModel(model)


__all__ = [
    "OptimizedAttention", "OptimizedLayer", "OptimizedModel",
    "optimize", "optimize_transformer_layer", "optimize_transformer_model",
]