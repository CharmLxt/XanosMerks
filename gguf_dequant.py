"""Алгоритмы деквантования GGUF-тензоров.

Реализованы:
- Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 — простые блочные (32 элемента)
- Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K — K-quants (256 элементов)

Не реализованы (raise NotImplementedError):
- IQ1_S, IQ2_*, IQ3_*, IQ4_* — очень сложные, требуют таблиц квантования
- TQ1_0, TQ2_0 — тернарные кванты, редкие
"""
import numpy as np


# ============================================================
# ВСПОМОГАТЕЛЬНОЕ: чтение fp16 из байтов
# ============================================================
def _read_fp16(byte_slice):
    """byte_slice: (N, 2) uint8. Возвращает (N,) float32."""
    return np.frombuffer(byte_slice.tobytes(), dtype=np.float16).astype(np.float32)


# ============================================================
# Q4_0: 18 байт на 32 элемента
# ============================================================
def dequant_q4_0(raw_bytes, shape):
    QK = 32
    block_size = 18
    n_elements = int(np.prod(shape))
    n_blocks = n_elements // QK

    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    blocks = raw[: n_blocks * block_size].reshape(n_blocks, block_size)

    scales = _read_fp16(blocks[:, 0:2])
    qs = blocks[:, 2:18]

    low = (qs & 0x0F).astype(np.int8)
    high = ((qs >> 4) & 0x0F).astype(np.int8)

    values = np.empty((n_blocks, QK), dtype=np.int8)
    values[:, 0::2] = low
    values[:, 1::2] = high

    out = (values.astype(np.float32) - 8.0) * scales[:, None]
    return out.reshape(shape).astype(np.float16)


# ============================================================
# Q4_1: 20 байт на 32 элемента
# ============================================================
def dequant_q4_1(raw_bytes, shape):
    QK = 32
    block_size = 20
    n_elements = int(np.prod(shape))
    n_blocks = n_elements // QK

    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    blocks = raw[: n_blocks * block_size].reshape(n_blocks, block_size)

    scales = _read_fp16(blocks[:, 0:2])
    mins = _read_fp16(blocks[:, 2:4])
    qs = blocks[:, 4:20]

    low = (qs & 0x0F).astype(np.float32)
    high = ((qs >> 4) & 0x0F).astype(np.float32)

    values = np.empty((n_blocks, QK), dtype=np.float32)
    values[:, 0::2] = low
    values[:, 1::2] = high

    out = values * scales[:, None] + mins[:, None]
    return out.reshape(shape).astype(np.float16)


# ============================================================
# Q5_0: 22 байта на 32 элемента
# ============================================================
def dequant_q5_0(raw_bytes, shape):
    QK = 32
    block_size = 22
    n_elements = int(np.prod(shape))
    n_blocks = n_elements // QK

    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    blocks = raw[: n_blocks * block_size].reshape(n_blocks, block_size)

    scales = _read_fp16(blocks[:, 0:2])
    qh = blocks[:, 2:6]
    qs = blocks[:, 6:22]

    low = (qs & 0x0F).astype(np.int16)
    high = ((qs >> 4) & 0x0F).astype(np.int16)

    qh_uint32 = qh.view(np.uint32).reshape(-1)
    qh_bits = ((qh_uint32[:, None] >> np.arange(32)) & 1).astype(np.int16)

    values = np.empty((n_blocks, QK), dtype=np.int16)
    values[:, 0::2] = low
    values[:, 1::2] = high

    values |= (qh_bits << 4)
    values = values.astype(np.int16) - 16

    out = values.astype(np.float32) * scales[:, None]
    return out.reshape(shape).astype(np.float16)


# ============================================================
# Q5_1: 24 байта на 32 элемента
# ============================================================
def dequant_q5_1(raw_bytes, shape):
    QK = 32
    block_size = 24
    n_elements = int(np.prod(shape))
    n_blocks = n_elements // QK

    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    blocks = raw[: n_blocks * block_size].reshape(n_blocks, block_size)

    scales = _read_fp16(blocks[:, 0:2])
    mins = _read_fp16(blocks[:, 2:4])
    qh = blocks[:, 4:8]
    qs = blocks[:, 8:24]

    low = (qs & 0x0F).astype(np.int16)
    high = ((qs >> 4) & 0x0F).astype(np.int16)

    qh_uint32 = qh.view(np.uint32).reshape(-1)
    qh_bits = ((qh_uint32[:, None] >> np.arange(32)) & 1).astype(np.int16)

    values = np.empty((n_blocks, QK), dtype=np.int16)
    values[:, 0::2] = low
    values[:, 1::2] = high
    values |= (qh_bits << 4)

    out = values.astype(np.float32) * scales[:, None] + mins[:, None]
    return out.reshape(shape).astype(np.float16)


# ============================================================
# Q8_0: 34 байта на 32 элемента
# ============================================================
def dequant_q8_0(raw_bytes, shape):
    QK = 32
    block_size = 34
    n_elements = int(np.prod(shape))
    n_blocks = n_elements // QK

    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    blocks = raw[: n_blocks * block_size].reshape(n_blocks, block_size)

    scales = _read_fp16(blocks[:, 0:2])
    qs = blocks[:, 2:34].view(np.int8)

    out = qs.astype(np.float32) * scales[:, None]
    return out.reshape(shape).astype(np.float16)


# ============================================================
# Q8_1: 40 байт на 32 элемента
# ============================================================
def dequant_q8_1(raw_bytes, shape):
    QK = 32
    block_size = 40
    n_elements = int(np.prod(shape))
    n_blocks = n_elements // QK

    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    blocks = raw[: n_blocks * block_size].reshape(n_blocks, block_size)

    scales = np.frombuffer(blocks[:, 0:4].tobytes(), dtype=np.float32)
    mins = np.frombuffer(blocks[:, 4:8].tobytes(), dtype=np.float32)
    qs = blocks[:, 8:40].view(np.int8)

    out = qs.astype(np.float32) * scales[:, None] + mins[:, None]
    return out.reshape(shape).astype(np.float16)


# ============================================================
# Q2_K: 84 байта на 256 элементов
#   scales[16], qs[64], d(fp16), dmin(fp16)
# ============================================================
def dequant_q2_k(raw_bytes, shape):
    QK = 256
    block_size = 84
    n_elements = int(np.prod(shape))
    n_blocks = n_elements // QK

    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    blocks = raw[: n_blocks * block_size].reshape(n_blocks, block_size)

    scales = blocks[:, 0:16]
    qs = blocks[:, 16:80]
    d = _read_fp16(blocks[:, 80:82])
    dmin = _read_fp16(blocks[:, 82:84])

    out = np.empty((n_blocks, QK), dtype=np.float32)

    for b in range(n_blocks):
        q_arr = qs[b].astype(np.int32)
        sc = scales[b].astype(np.int32)
        d_b = float(d[b])
        dmin_b = float(dmin[b])
        is_idx = 0
        for n_start in range(0, QK, 128):
            q_off = (n_start // 128) * 32
            shift = 0
            for j in range(4):
                sc_val = int(sc[is_idx]); is_idx += 1
                dl = d_b * (sc_val & 0xF)
                ml = dmin_b * (sc_val >> 4)
                for l in range(16):
                    qval = (q_arr[q_off + l] >> shift) & 3
                    out[b, n_start + j * 32 + l] = dl * qval - ml
                sc_val = int(sc[is_idx]); is_idx += 1
                dl = d_b * (sc_val & 0xF)
                ml = dmin_b * (sc_val >> 4)
                for l in range(16):
                    qval = (q_arr[q_off + l + 16] >> shift) & 3
                    out[b, n_start + j * 32 + 16 + l] = dl * qval - ml
                shift += 2

    return out.reshape(shape).astype(np.float16)


# ============================================================
# Q3_K: 110 байт на 256 элементов
#   hmask[32], qs[64], scales_raw[12], d(fp16)
#
# ВАЖНО: hmask всегда индексируется 0..31 (в llama.cpp hm не сдвигается).
# Разница между двумя n-итерациями — только в m_bit (1,2,4,8 / 16,32,64,128).
# ============================================================
def dequant_q3_k(raw_bytes, shape):
    QK = 256
    block_size = 110
    n_elements = int(np.prod(shape))
    n_blocks = n_elements // QK

    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    blocks = raw[: n_blocks * block_size].reshape(n_blocks, block_size)

    hmask = blocks[:, 0:32]
    qs = blocks[:, 32:96]
    scales_raw = blocks[:, 96:108]
    d = _read_fp16(blocks[:, 108:110])

    kmask1 = 0x03030303
    kmask2 = 0x0F0F0F0F

    out = np.empty((n_blocks, QK), dtype=np.float32)

    for b in range(n_blocks):
        aux = np.zeros(4, dtype=np.uint32)
        for i in range(3):
            aux[i] = int.from_bytes(
                scales_raw[b, i * 4:(i + 1) * 4].tobytes(), "little"
            )
        tmp = int(aux[2])
        aux[2] = ((int(aux[0]) >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
        aux[3] = ((int(aux[1]) >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
        aux[0] = (int(aux[0]) & kmask2) | (((tmp >> 0) & kmask1) << 4)
        aux[1] = (int(aux[1]) & kmask2) | (((tmp >> 2) & kmask1) << 4)

        scales_arr = np.frombuffer(aux.tobytes(), dtype=np.int8)

        d_b = float(d[b])
        q_arr = qs[b]
        hm_arr = hmask[b]
        is_idx = 0
        m_bit = 1

        for n_start in range(0, QK, 128):
            q_off = (n_start // 128) * 32
            shift = 0
            for j in range(4):
                sc_val = int(scales_arr[is_idx]); is_idx += 1
                dl = d_b * (sc_val - 32)
                for l in range(16):
                    qval = (int(q_arr[q_off + l]) >> shift) & 3
                    hval = 0 if (int(hm_arr[l]) & m_bit) else 4
                    out[b, n_start + j * 32 + l] = dl * (qval - hval)

                sc_val = int(scales_arr[is_idx]); is_idx += 1
                dl = d_b * (sc_val - 32)
                for l in range(16):
                    qval = (int(q_arr[q_off + l + 16]) >> shift) & 3
                    hval = 0 if (int(hm_arr[l + 16]) & m_bit) else 4
                    out[b, n_start + j * 32 + 16 + l] = dl * (qval - hval)

                shift += 2
                m_bit <<= 1

    return out.reshape(shape).astype(np.float16)


# ============================================================
# Q4_K: 144 байта на 256 элементов
# ============================================================
def _get_scale_min_k4(j, scales):
    if j < 4:
        sc = scales[:, j] & 0x3F
        mn = scales[:, j + 4] & 0x3F
    else:
        sc = (scales[:, j + 4] & 0x0F) | ((scales[:, j - 4] >> 6) << 4)
        mn = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
    return sc.astype(np.float32), mn.astype(np.float32)


def dequant_q4_k(raw_bytes, shape):
    QK = 256
    block_size = 144
    n_elements = int(np.prod(shape))
    n_blocks = n_elements // QK

    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    blocks = raw[: n_blocks * block_size].reshape(n_blocks, block_size)

    d = _read_fp16(blocks[:, 0:2])
    dmin = _read_fp16(blocks[:, 2:4])
    scales = blocks[:, 4:16]
    qs = blocks[:, 16:144]

    out = np.empty((n_blocks, QK), dtype=np.float32)

    for iter_j in range(4):
        for half in range(2):
            is_idx = iter_j * 2 + half
            sc, mn = _get_scale_min_k4(is_idx, scales)
            d1 = d * sc
            m1 = dmin * mn

            q_slice = qs[:, iter_j * 32:(iter_j + 1) * 32]
            if half == 0:
                vals = (q_slice & 0x0F).astype(np.float32)
            else:
                vals = ((q_slice >> 4) & 0x0F).astype(np.float32)

            start = iter_j * 64 + half * 32
            out[:, start:start + 32] = d1[:, None] * vals - m1[:, None]

    return out.reshape(shape).astype(np.float16)


# ============================================================
# Q5_K: 176 байт на 256 элементов
# ============================================================
def dequant_q5_k(raw_bytes, shape):
    QK = 256
    block_size = 176
    n_elements = int(np.prod(shape))
    n_blocks = n_elements // QK

    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    blocks = raw[: n_blocks * block_size].reshape(n_blocks, block_size)

    d = _read_fp16(blocks[:, 0:2])
    dmin = _read_fp16(blocks[:, 2:4])
    scales = blocks[:, 4:16]
    qh = blocks[:, 16:48]
    qs = blocks[:, 48:176]

    out = np.empty((n_blocks, QK), dtype=np.float32)

    for iter_j in range(4):
        for half in range(2):
            is_idx = iter_j * 2 + half
            sc, mn = _get_scale_min_k4(is_idx, scales)
            d1 = d * sc
            m1 = dmin * mn

            q_slice = qs[:, iter_j * 32:(iter_j + 1) * 32]
            qh_slice = qh[:, iter_j * 4:(iter_j + 1) * 4]
            qh_uint32 = qh_slice.view(np.uint32).reshape(-1)
            qh_bits = ((qh_uint32[:, None] >> np.arange(32)) & 1).astype(np.float32)

            if half == 0:
                vals = (q_slice & 0x0F).astype(np.float32)
            else:
                vals = ((q_slice >> 4) & 0x0F).astype(np.float32)

            vals = vals + qh_bits * 16.0

            start = iter_j * 64 + half * 32
            out[:, start:start + 32] = d1[:, None] * vals - m1[:, None]

    return out.reshape(shape).astype(np.float16)


# ============================================================
# Q6_K: 210 байт на 256 элементов
# ============================================================
def dequant_q6_k(raw_bytes, shape):
    QK = 256
    block_size = 210
    n_elements = int(np.prod(shape))
    n_blocks = n_elements // QK

    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    blocks = raw[: n_blocks * block_size].reshape(n_blocks, block_size)

    ql = blocks[:, 0:128]
    qh = blocks[:, 128:192]
    scales = blocks[:, 192:208].view(np.int8)
    d = _read_fp16(blocks[:, 208:210])

    out = np.empty((n_blocks, QK), dtype=np.float32)

    for n_block in range(2):
        ql_off = n_block * 64
        qh_off = n_block * 32
        sc_off = n_block * 8
        y_off = n_block * 128

        ql_slice = ql[:, ql_off:ql_off + 64]
        qh_slice = qh[:, qh_off:qh_off + 32]
        sc_slice = scales[:, sc_off:sc_off + 8].astype(np.float32)

        for l in range(32):
            is_idx = l // 16

            q1 = ((ql_slice[:, l] & 0x0F) | (((qh_slice[:, l] >> 0) & 0x03) << 4)).astype(np.int8) - 32
            q2 = ((ql_slice[:, l + 32] & 0x0F) | (((qh_slice[:, l] >> 2) & 0x03) << 4)).astype(np.int8) - 32
            q3 = ((ql_slice[:, l] >> 4) | (((qh_slice[:, l] >> 4) & 0x03) << 4)).astype(np.int8) - 32
            q4 = ((ql_slice[:, l + 32] >> 4) | (((qh_slice[:, l] >> 6) & 0x03) << 4)).astype(np.int8) - 32

            sc0 = sc_slice[:, is_idx]
            sc2 = sc_slice[:, is_idx + 2]
            sc4 = sc_slice[:, is_idx + 4]
            sc6 = sc_slice[:, is_idx + 6]

            out[:, y_off + l] = d * sc0 * q1.astype(np.float32)
            out[:, y_off + l + 32] = d * sc2 * q2.astype(np.float32)
            out[:, y_off + l + 64] = d * sc4 * q3.astype(np.float32)
            out[:, y_off + l + 96] = d * sc6 * q4.astype(np.float32)

    return out.reshape(shape).astype(np.float16)


# ============================================================
# Q8_K: 292 байта на 256 элементов
# ============================================================
def dequant_q8_k(raw_bytes, shape):
    QK = 256
    block_size = 292
    n_elements = int(np.prod(shape))
    n_blocks = n_elements // QK

    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    blocks = raw[: n_blocks * block_size].reshape(n_blocks, block_size)

    d = np.frombuffer(blocks[:, 0:4].tobytes(), dtype=np.float32)
    qs = blocks[:, 4:260].view(np.int8)

    out = qs.astype(np.float32) * d[:, None]
    return out.reshape(shape).astype(np.float16)


# ============================================================
# ДИСПЕТЧЕР
# ============================================================
DEQUANT_FUNCS = {
    "Q4_0": dequant_q4_0,
    "Q4_1": dequant_q4_1,
    "Q5_0": dequant_q5_0,
    "Q5_1": dequant_q5_1,
    "Q8_0": dequant_q8_0,
    "Q8_1": dequant_q8_1,
    "Q2_K": dequant_q2_k,
    "Q3_K": dequant_q3_k,
    "Q4_K": dequant_q4_k,
    "Q5_K": dequant_q5_k,
    "Q6_K": dequant_q6_k,
    "Q8_K": dequant_q8_k,
}


def dequantize(raw_bytes, tensor_type, shape):
    """Деквантует тензор GGUF по его типу."""
    ttype = tensor_type.upper()

    if ttype.startswith("Q2_K"):
        ttype = "Q2_K"
    elif ttype.startswith("Q3_K"):
        ttype = "Q3_K"
    elif ttype.startswith("Q4_K"):
        ttype = "Q4_K"
    elif ttype.startswith("Q5_K"):
        ttype = "Q5_K"
    elif ttype.startswith("Q6_K"):
        ttype = "Q6_K"
    elif ttype.startswith("Q8_K"):
        ttype = "Q8_K"
    elif ttype.startswith("IQ"):
        raise NotImplementedError(
            f"{tensor_type} (IQ-квантование) ещё не реализовано в meanflare. "
            f"Используйте Q4_K_M или Q8_0 модели."
        )
    elif ttype.startswith("TQ"):
        raise NotImplementedError(
            f"{tensor_type} (тернарное квантование) не реализовано в meanflare."
        )

    if ttype in ("F16", "FP16"):
        return np.frombuffer(raw_bytes, dtype=np.float16).reshape(shape).copy()
    if ttype in ("F32", "FP32"):
        return np.frombuffer(raw_bytes, dtype=np.float32).reshape(shape).astype(np.float16)
    if ttype in ("BF16",):
        from .dtype_ops import bf16_to_fp32
        arr_bf16 = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(shape)
        return bf16_to_fp32(arr_bf16).astype(np.float16)
    if ttype in ("I8", "INT8"):
        return np.frombuffer(raw_bytes, dtype=np.int8).reshape(shape).astype(np.float16)
    if ttype in ("I16", "INT16"):
        return np.frombuffer(raw_bytes, dtype=np.int16).reshape(shape).astype(np.float16)
    if ttype in ("I32", "INT32"):
        return np.frombuffer(raw_bytes, dtype=np.int32).reshape(shape).astype(np.float16)
    if ttype in ("I64", "INT64"):
        return np.frombuffer(raw_bytes, dtype=np.int64).reshape(shape).astype(np.float16)

    if ttype in DEQUANT_FUNCS:
        return DEQUANT_FUNCS[ttype](raw_bytes, shape)

    raise NotImplementedError(f"Тип {tensor_type} ещё не поддерживается в meanflare")


__all__ = [
    "dequant_q4_0", "dequant_q4_1", "dequant_q5_0", "dequant_q5_1",
    "dequant_q8_0", "dequant_q8_1",
    "dequant_q2_k", "dequant_q3_k",
    "dequant_q4_k", "dequant_q5_k", "dequant_q6_k", "dequant_q8_k",
    "DEQUANT_FUNCS", "dequantize",
]