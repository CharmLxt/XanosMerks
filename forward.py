"""Forward-pass через слои трансформера."""
import numpy as np
from .primitives import rms_norm, layer_norm, swiglu, apply_activation
from .attention import MultiHeadAttention
from .moe import MoELayer
from .mla import MultiHeadLatentAttention


def _nest_weights(flat):
    result = {}
    for key, value in flat.items():
        parts = key.split(".")
        node = result
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return result


def _cache_position(cache):
    return getattr(cache, "absolute_position", cache.cur_len)


class TransformerLayer:
    def __init__(self, config, weights, layer_idx, kv_cache=None, rope_cache=None):
        self.layer_idx = layer_idx
        self.config = config
        self.attention_type = config.get("attention_type", "mha")
        if self.attention_type == "mla":
            self.attn = MultiHeadLatentAttention(
                config, weights["self_attn"], kv_cache, rope_cache
            )
        else:
            self.attn = MultiHeadAttention(
                config, weights["self_attn"], kv_cache, rope_cache
            )

        self.activation = config.get("hidden_act", "silu")
        self.norm_eps = config.get("norm_eps", 1e-5)
        self.norm_type = config.get("norm_type", "rmsnorm")
        self.is_moe = config.get("is_moe", False)

        if self.is_moe:
            self.moe = MoELayer(config, weights["block_sparse_moe"])
            self.fused_mlp = None
            self.w_gate = self.b_gate = None
            self.w_up = self.b_up = None
            self.w_down = self.b_down = None
        else:
            self.moe = None
            mlp = weights["mlp"]
            self.w_gate = mlp["gate_proj"]["weight"].astype(np.float32)
            self.b_gate = mlp["gate_proj"].get("bias")
            if self.b_gate is not None: self.b_gate = self.b_gate.astype(np.float32)
            self.w_up = mlp["up_proj"]["weight"].astype(np.float32)
            self.b_up = mlp["up_proj"].get("bias")
            if self.b_up is not None: self.b_up = self.b_up.astype(np.float32)
            self.w_down = mlp["down_proj"]["weight"].astype(np.float32)
            self.b_down = mlp["down_proj"].get("bias")
            if self.b_down is not None: self.b_down = self.b_down.astype(np.float32)

            has_mlp_bias = any(b is not None for b in (self.b_gate, self.b_up, self.b_down))
            if not has_mlp_bias:
                from ninewas9 import FusedMLP
                self.fused_mlp = FusedMLP(
                    self.w_gate, self.w_up, self.w_down,
                    activation=self.activation,
                )
                self.w_gate = self.w_up = self.w_down = None
            else:
                self.fused_mlp = None

        n1 = weights["input_layernorm"]
        self.norm1_w = n1["weight"].astype(np.float32)
        self.norm1_b = n1.get("bias")
        if self.norm1_b is not None: self.norm1_b = self.norm1_b.astype(np.float32)

        n2 = weights["post_attention_layernorm"]
        self.norm2_w = n2["weight"].astype(np.float32)
        self.norm2_b = n2.get("bias")
        if self.norm2_b is not None: self.norm2_b = self.norm2_b.astype(np.float32)

    def _norm(self, x, w, b):
        if self.norm_type == "rmsnorm":
            return rms_norm(x, w, self.norm_eps)
        return layer_norm(x, w, b, self.norm_eps)

    def _mlp(self, x):
        if self.moe is not None:
            return self.moe(x)
        if self.fused_mlp is not None:
            return self.fused_mlp(x)
        gate = x @ self.w_gate.T
        if self.b_gate is not None: gate = gate + self.b_gate
        up = x @ self.w_up.T
        if self.b_up is not None: up = up + self.b_up
        if self.activation in ("silu", "swish"):
            h = swiglu(gate, up)
        else:
            h = apply_activation(gate, self.activation) * up
        out = h @ self.w_down.T
        if self.b_down is not None: out = out + self.b_down
        return out

    def __call__(self, x, position_offset, kv_attention_mask=None):
        residual = x
        h = self._norm(x, self.norm1_w, self.norm1_b)
        h = self.attn(h, self.layer_idx, position_offset,
                      kv_attention_mask=kv_attention_mask)
        x = residual + h

        residual = x
        h = self._norm(x, self.norm2_w, self.norm2_b)
        h = self._mlp(h)
        x = residual + h
        return x


class TransformerModel:
    def __init__(self, config, weights, kv_cache=None, rope_cache=None):
        self.config = config
        self.kv_cache = kv_cache
        self.norm_eps = config.get("norm_eps", 1e-5)
        self.norm_type = config.get("norm_type", "rmsnorm")

        w = _nest_weights(weights)
        self.embed_tokens = w["model"]["embed_tokens"]["weight"].astype(np.float32)

        self.layers = []
        for i in range(config["num_layers"]):
            layer_w = w["model"]["layers"][str(i)]
            self.layers.append(TransformerLayer(
                config, layer_w, i, kv_cache, rope_cache,
            ))

        final_norm = w["model"]["norm"]
        self.final_norm_w = final_norm["weight"].astype(np.float32)
        self.final_norm_b = final_norm.get("bias")
        if self.final_norm_b is not None: self.final_norm_b = self.final_norm_b.astype(np.float32)

        if config.get("tie_word_embeddings", False):
            self.lm_head = self.embed_tokens
        else:
            self.lm_head = w["lm_head"]["weight"].astype(np.float32)
        self.lm_head_T = self.lm_head.T

    def _norm(self, x, w, b):
        if self.norm_type == "rmsnorm":
            return rms_norm(x, w, self.norm_eps)
        return layer_norm(x, w, b, self.norm_eps)

    def _forward(self, input_ids, position_offset, last_only, kv_attention_mask):
        x = self.embed_tokens[input_ids]
        for layer in self.layers:
            x = layer(x, position_offset, kv_attention_mask=kv_attention_mask)
        x = self._norm(x, self.final_norm_w, self.final_norm_b)
        if last_only:
            return x[:, -1, :] @ self.lm_head_T
        return x @ self.lm_head_T

    def _needs_chunked_prefill(self, seq_len):
        """Chunking нужен в двух случаях:

        1. seq_len > window_size — append физически не может записать
           столько токенов за раз (WindowOverflowError в append).
        2. cur + seq_len > capacity (sink + window) — eviction случится
           внутри append, маска для полного prefill будет неправильной.

        Если ни одно условие не выполнено — обычный one-shot forward.
        """
        if self.kv_cache is None or seq_len <= 1:
            return False
        if not hasattr(self.kv_cache, "window_size"):
            return False
        window_size = self.kv_cache.window_size
        sink_size = getattr(self.kv_cache, "sink_size", 0)
        capacity = sink_size + window_size
        cur = _cache_position(self.kv_cache)

        if seq_len > window_size:
            return True
        return cur + seq_len > capacity

    def __call__(self, input_ids, position_offset=0, last_only=False,
                 kv_attention_mask=None):
        if self.kv_cache is not None:
            expected = _cache_position(self.kv_cache)
            if position_offset != expected:
                raise ValueError(
                    f"position_offset={position_offset} != {expected}. "
                    f"Возможно, забыли advance()."
                )

        if input_ids.ndim != 2:
            raise ValueError(f"input_ids должен быть 2D, получено {input_ids.ndim}D")
        if input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError(f"input_ids пустой: {input_ids.shape}")

        seq_len = input_ids.shape[1]

        if self._needs_chunked_prefill(seq_len):
            outputs = []
            for i in range(seq_len):
                xi = input_ids[:, i:i + 1]
                li = self._forward(
                    xi, position_offset + i, last_only=False,
                    kv_attention_mask=None,
                )
                outputs.append(li)
            full = np.concatenate(outputs, axis=1)
            if last_only:
                return full[:, -1, :]
            return full

        return self._forward(input_ids, position_offset, last_only, kv_attention_mask)


__all__ = ["TransformerLayer", "TransformerModel"]