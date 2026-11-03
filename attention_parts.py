"""Attention для разделённых sink и window K/V без concat.

Эквивалентно scaled_dot_product_attention из ninewas9, но принимает
sink и window как отдельные тензоры. Считает два matmul, потом
concat'ит scores (маленькие) и делает одну softmax. K/V не копируются.
"""
import numpy as np
from ninewas9 import fast_softmax


def scaled_dot_product_attention_parts(
    q, sink_k, sink_v, window_k, window_v,
    mask=None, scale=None, dropout_p=0.0, return_weights=False,
):
    """Attention по частям.

    Параметры:
        q: (B, H_q, S, D)
        sink_k, sink_v: (B, H_kv, sink_len, D)
        window_k, window_v: (B, H_kv, win_len, D)
        mask: (S, sink_len + win_len) или None
        scale: default = 1/sqrt(D)
        return_weights: если True — вернуть (out, attn_weights)

    Возвращает:
        out: (B, H_q, S, D) или (out, attn_weights) с attn_weights формы
             (B, H_q, S, sink_len + win_len).
    """
    if scale is None:
        scale = 1.0 / np.sqrt(q.shape[-1])

    H_q = q.shape[1]
    H_kv = sink_k.shape[1]
    sink_len = sink_k.shape[2]
    win_len = window_k.shape[2]

    if H_kv == H_q:
        scores_sink = np.matmul(q, np.swapaxes(sink_k, -1, -2)) * scale
        scores_win = np.matmul(q, np.swapaxes(window_k, -1, -2)) * scale
        scores = np.concatenate([scores_sink, scores_win], axis=-1)
        if mask is not None:
            scores = scores + mask
        attn_weights = fast_softmax(scores, axis=-1)
        if dropout_p > 0:
            keep = np.random.rand(*attn_weights.shape) > dropout_p
            attn_weights = attn_weights * keep / (1.0 - dropout_p)
        out = (
            np.matmul(attn_weights[..., :sink_len], sink_v)
            + np.matmul(attn_weights[..., sink_len:], window_v)
        )
        if return_weights:
            return out, attn_weights
        return out

    if H_q % H_kv != 0:
        raise ValueError(
            f"scaled_dot_product_attention_parts: H_q={H_q} не делится на H_kv={H_kv}"
        )
    r = H_q // H_kv
    B, _, S, D = q.shape
    q_exp = q.reshape(B, H_kv, r, S, D)
    sink_k_exp = sink_k[:, :, None, :, :]
    window_k_exp = window_k[:, :, None, :, :]
    sink_v_exp = sink_v[:, :, None, :, :]
    window_v_exp = window_v[:, :, None, :, :]

    scores_sink = np.matmul(q_exp, np.swapaxes(sink_k_exp, -1, -2)) * scale
    scores_win = np.matmul(q_exp, np.swapaxes(window_k_exp, -1, -2)) * scale
    scores = np.concatenate([scores_sink, scores_win], axis=-1)
    if mask is not None:
        scores = scores + mask[None, None, None, :, :]
    attn_weights = fast_softmax(scores, axis=-1)
    if dropout_p > 0:
        keep = np.random.rand(*attn_weights.shape) > dropout_p
        attn_weights = attn_weights * keep / (1.0 - dropout_p)
    out = (
        np.matmul(attn_weights[..., :sink_len], sink_v_exp)
        + np.matmul(attn_weights[..., sink_len:], window_v_exp)
    ).reshape(B, H_q, S, D)
    if return_weights:
        return out, attn_weights.reshape(B, H_q, S, sink_len + win_len)
    return out


__all__ = ["scaled_dot_product_attention_parts"]