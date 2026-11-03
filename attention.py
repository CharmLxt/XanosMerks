"""Multi-Head Attention с поддержкой GQA/MQA, nsofi40, H2O и padding."""
import numpy as np
from ninewas9 import RoPECache, FusedLinear, scaled_dot_product_attention, fast_softmax
from xanosmerks.nsofi40.attention_parts import scaled_dot_product_attention_parts


class MultiHeadAttention:
    def __init__(self, config, weights, kv_cache=None, rope_cache=None):
        self.n_heads = config["num_heads"]
        self.n_kv_heads = config["num_kv_heads"]
        self.head_dim = config["head_dim"]
        self.hidden = config["hidden_size"]

        def _get(mod, param):
            sub = weights.get(mod, {})
            v = sub.get(param)
            return v.astype(np.float32) if v is not None else None

        self.w_q = _get("q_proj", "weight")
        self.b_q = _get("q_proj", "bias")
        self.w_k = _get("k_proj", "weight")
        self.b_k = _get("k_proj", "bias")
        self.w_v = _get("v_proj", "weight")
        self.b_v = _get("v_proj", "bias")
        self.w_o = _get("o_proj", "weight")
        self.b_o = _get("o_proj", "bias")

        self.kv_cache = kv_cache
        self.rope_cache = rope_cache

        if all(b is None for b in (self.b_q, self.b_k, self.b_v)):
            self.fused_qkv = FusedLinear([
                ("q", self.w_q), ("k", self.w_k), ("v", self.w_v),
            ])
        else:
            self.fused_qkv = None

    def __call__(self, x, layer_idx, position_offset, kv_attention_mask=None):
        batch, seq_len, _ = x.shape
        if seq_len < 1:
            raise ValueError(f"MultiHeadAttention: seq_len={seq_len}.")

        if self.fused_qkv is not None:
            qkv = self.fused_qkv(x)
            q, k, v = qkv["q"], qkv["k"], qkv["v"]
        else:
            q = x @ self.w_q.T
            if self.b_q is not None: q = q + self.b_q
            k = x @ self.w_k.T
            if self.b_k is not None: k = k + self.b_k
            v = x @ self.w_v.T
            if self.b_v is not None: v = v + self.b_v

        q = q.reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.rope_cache is not None:
            cos, sin = self.rope_cache.slice_for_positions(
                position_offset, position_offset + seq_len
            )
            q, k = RoPECache.apply(q, k, cos, sin)

        use_parts = self.kv_cache is not None and hasattr(self.kv_cache, "get_parts")
        need_weights = self.kv_cache is not None and getattr(
            self.kv_cache, "needs_attention_scores", False
        )

        if use_parts:
            if kv_attention_mask is not None:
                raise ValueError(
                    "nsofi40 не поддерживает padding (kv_attention_mask)."
                )
            self.kv_cache.append(layer_idx, k, v)
            sink_k, sink_v, window_k, window_v = self.kv_cache.get_parts(layer_idx)
            sink_count = sink_k.shape[2]
            window_count = window_k.shape[2]
            kv_len = sink_count + window_count
            if kv_len < 1:
                raise ValueError(f"kv_len={kv_len} после append.")
            mask = None
            if seq_len > 1:
                mask = self._build_causal_mask_nsofi(seq_len, kv_len)
            if need_weights:
                attn_out, attn_weights = scaled_dot_product_attention_parts(
                    q, sink_k, sink_v, window_k, window_v,
                    mask=mask, return_weights=True,
                )
                if self.n_heads == self.n_kv_heads:
                    per_head = attn_weights.sum(axis=-2)
                else:
                    r = self.n_heads // self.n_kv_heads
                    B_, _, S_, L_ = attn_weights.shape
                    per_head = attn_weights.reshape(
                        B_, self.n_kv_heads, r, S_, L_
                    ).sum(axis=(2, 3))
                self.kv_cache.record_attention(layer_idx, per_head)
            else:
                attn_out = scaled_dot_product_attention_parts(
                    q, sink_k, sink_v, window_k, window_v, mask=mask,
                )

        elif self.kv_cache is not None:
            self.kv_cache.append(layer_idx, k, v)
            kv_end = position_offset + seq_len
            k_full = self.kv_cache.k[layer_idx, :, :, :kv_end, :]
            v_full = self.kv_cache.v[layer_idx, :, :, :kv_end, :]
            kv_len = k_full.shape[2]
            if kv_len < 1:
                raise ValueError(f"kv_len={kv_len} после append.")

            if kv_attention_mask is not None:
                if kv_attention_mask.shape[1] != kv_len:
                    raise ValueError(
                        f"kv_attention_mask.shape[1]={kv_attention_mask.shape[1]} "
                        f"!= kv_len={kv_len}"
                    )
                mask_3d = self._build_mask_3d(
                    seq_len, kv_len, position_offset, kv_attention_mask,
                )
                attn_out = self._attention_with_3d_mask(q, k_full, v_full, mask_3d)
            else:
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
        out = attn_out @ self.w_o.T
        if self.b_o is not None:
            out = out + self.b_o
        return out

    def _attention_with_3d_mask(self, q, k, v, mask_3d):
        scale = 1.0 / np.sqrt(self.head_dim)
        if self.n_heads == self.n_kv_heads:
            scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
            scores = scores + mask_3d[:, None, :, :]
            attn_w = fast_softmax(scores, axis=-1)
            return np.matmul(attn_w, v)
        r = self.n_heads // self.n_kv_heads
        B, _, S, D = q.shape
        q_exp = q.reshape(B, self.n_kv_heads, r, S, D)
        k_exp = k[:, :, None, :, :]
        v_exp = v[:, :, None, :, :]
        scores = np.matmul(q_exp, np.swapaxes(k_exp, -1, -2)) * scale
        scores = scores + mask_3d[:, None, None, :, :]
        attn_w = fast_softmax(scores, axis=-1)
        out = np.matmul(attn_w, v_exp)
        return out.reshape(B, self.n_heads, S, D)

    @staticmethod
    def _build_mask_3d(seq_len, kv_len, position_offset, kv_attention_mask):
        causal = np.zeros((seq_len, kv_len), dtype=np.float32)
        for i in range(seq_len):
            end = min(position_offset + i + 1, kv_len)
            causal[i, :end] = 1.0
        kv_valid = kv_attention_mask.astype(np.float32)
        combined = causal[None, :, :] * kv_valid[:, None, :]
        return np.where(combined > 0, 0.0, -np.inf).astype(np.float32)

    @staticmethod
    def _build_causal_mask(seq_len, kv_len, position_offset):
        mask = np.full((seq_len, kv_len), -np.inf, dtype=np.float32)
        for i in range(seq_len):
            end = min(position_offset + i + 1, kv_len)
            mask[i, :end] = 0.0
        return mask

    @staticmethod
    def _build_causal_mask_nsofi(seq_len, total_kv_len):
        """Causal mask для nsofi40.

        total_kv_len = sink_count + window_count (уже включает новые токены,
        потому что append выполнен ДО вызова этой функции).
        past_len = total_kv_len - seq_len — сколько прошлых токенов видят все
        новые. Если seq_len > total_kv_len (бывает при eviction в prefill),
        past_len = 0 — все прошлые вытеснены, новые друг друга видят causal.
        """
        past_len = max(0, total_kv_len - seq_len)
        mask = np.full((seq_len, total_kv_len), -np.inf, dtype=np.float32)
        if past_len > 0:
            mask[:, :past_len] = 0.0
        for i in range(seq_len):
            end = min(past_len + i + 1, total_kv_len)
            mask[i, past_len:end] = 0.0
        return mask


__all__ = ["MultiHeadAttention"]