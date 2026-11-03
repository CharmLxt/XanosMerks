"""Низкоуровневые numpy-операции с BF16, FP8, FP16, FP32, INT8."""
import numpy as np


# ============================================================
# BF16 (Brain Float 16) <-> FP32
# ============================================================
# BF16 — это усечённый FP32: он хранит только верхние 16 бит мантиссы.
# Layout: S EEEEEEEE MMMMMMM (1 sign + 8 exp + 7 mantissa)
# Формат НЕ поддерживается numpy нативно, поэтому храним как uint16.

def bf16_to_fp32(bf16_array):
    """
    BF16 (uint16) → FP32 (float32).
    Сдвигаем 16-битное значение влево на 16 и интерпретируем как float32.
    Это точное преобразование без потерь (bf16 — усечённый float32).
    """
    arr = np.asarray(bf16_array, dtype=np.uint16)
    # Переводим в uint32, сдвигаем влево на 16, смотрим как float32
    uint32_array = arr.astype(np.uint32) << 16
    return uint32_array.view(np.float32)


def fp32_to_bf16(fp32_array, rounding=True):
    """
    FP32 → BF16 (uint16).
    Берём верхние 16 бит float32. Если rounding=True — добавляем 0x8000 перед
    сдвигом, чтобы получить корректное округление «до ближайшего чётного»
    (round-to-nearest-even).
    """
    arr = np.asarray(fp32_array, dtype=np.float32)
    uint32_view = arr.view(np.uint32)
    if rounding:
        # round-to-nearest-even: добавляем половину ULА младшего разряда
        # Наивное округление с "half-up" даёт ошибки в крайних случаях.
        # Правильная формула: (u + 0x7FFF + ((u >> 16) & 1)) >> 16
        u = uint32_view
        rounding_bias = 0x7FFF + ((u >> 16) & 1)
        result = ((u + rounding_bias) >> 16).astype(np.uint16)
    else:
        # Простое усечение (truncate) — быстрее, но с погрешностью
        result = (uint32_view >> 16).astype(np.uint16)
    return result


def bf16_to_fp16(bf16_array):
    """BF16 → FP16. Через промежуточный FP32 (может терять точность)."""
    return bf16_to_fp32(bf16_array).astype(np.float16)


# ============================================================
# FP8 (8-bit floating point) <-> FP16 / FP32
# ============================================================
# Два варианта FP8: E4M3 (4 exp, 3 mantissa) и E5M2 (5 exp, 2 mantissa).
# У них разные bias: E4M3 имеет bias 7, E5M2 — bias 15.

# Таблица для E4M3: строим её один раз (256 записей, каждая = float32)
_FP8_E4M3_TABLE = None


def _build_fp8_e4m3_table():
    """Строит lookup-таблицу: uint8 → float32 для E4M3."""
    table = np.zeros(256, dtype=np.float32)
    for i in range(256):
        sign = (i >> 7) & 1
        exp = (i >> 3) & 0x0F
        mantissa = i & 0x07

        if exp == 0:
            # Субнормальное или ноль: value = mantissa * 2^(-6) / 8
            # = mantissa * 2^(-6) * (1/8) = mantissa * 2^(-9)
            value = (mantissa / 8.0) * (2.0 ** -6)
        elif exp == 15 and mantissa == 7:
            # NaN
            value = float("nan")
        else:
            # Нормальное: value = (1 + mantissa/8) * 2^(exp - 7)
            value = (1.0 + mantissa / 8.0) * (2.0 ** (exp - 7))

        if sign:
            value = -value
        table[i] = value
    return table


# Таблица для E5M2 (S EEEEE MM)
_FP8_E5M2_TABLE = None


def _build_fp8_e5m2_table():
    """Строит lookup-таблицу: uint8 → float32 для E5M2."""
    table = np.zeros(256, dtype=np.float32)
    for i in range(256):
        sign = (i >> 7) & 1
        exp = (i >> 2) & 0x1F
        mantissa = i & 0x03

        if exp == 0:
            value = (mantissa / 4.0) * (2.0 ** -14)
        elif exp == 31:
            # Inf / NaN
            if mantissa == 0:
                value = float("inf")
            else:
                value = float("nan")
        else:
            value = (1.0 + mantissa / 4.0) * (2.0 ** (exp - 15))

        if sign:
            value = -value
        table[i] = value
    return table


def _get_fp8_table(variant):
    global _FP8_E4M3_TABLE, _FP8_E5M2_TABLE
    if variant == "e4m3":
        if _FP8_E4M3_TABLE is None:
            _FP8_E4M3_TABLE = _build_fp8_e4m3_table()
        return _FP8_E4M3_TABLE
    elif variant == "e5m2":
        if _FP8_E5M2_TABLE is None:
            _FP8_E5M2_TABLE = _build_fp8_e5m2_table()
        return _FP8_E5M2_TABLE
    else:
        raise ValueError(f"Неизвестный вариант FP8: {variant}. Ожидалось 'e4m3' или 'e5m2'.")


def fp8_to_fp32(fp8_array, variant="e4m3"):
    """
    FP8 (uint8) → FP32 (float32).
    Использует lookup-таблицу (быстро и точно).
    """
    arr = np.asarray(fp8_array, dtype=np.uint8)
    table = _get_fp8_table(variant)
    return table[arr]


def fp8_to_fp16(fp8_array, variant="e4m3"):
    """FP8 (uint8) → FP16 (float16). Через FP32, потом astype."""
    return fp8_to_fp32(fp8_array, variant=variant).astype(np.float16)


# ============================================================
# INT8 → FP16 (со скейлом)
# ============================================================
def int8_to_fp16(int8_array, scale=1.0, zero_point=0):
    """
    INT8 → FP16 с линейным преобразованием:
    value = (int8_value - zero_point) * scale.
    """
    arr = np.asarray(int8_array, dtype=np.int8)
    return ((arr.astype(np.float32) - zero_point) * scale).astype(np.float16)


# ============================================================
# ВСПОМОГАТЕЛЬНОЕ: определение размера dtype
# ============================================================
DTYPE_SIZE = {
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "bfloat16": 2,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
}


def get_dtype_size(dtype_name):
    """Размер одного элемента в байтах по имени dtype."""
    return DTYPE_SIZE.get(dtype_name, 4)


def normalize_dtype_name(name):
    """Приводит строку dtype к каноничному виду."""
    name = name.lower().replace("-", "_").replace(".", "")
    aliases = {
        "f32": "float32", "f16": "float16", "f64": "float64",
        "fp16": "float16", "fp32": "float32",
        "bf16": "bfloat16",
        "fp8_e4m3fn": "fp8_e4m3", "fp8e4m3": "fp8_e4m3",
        "fp8_e5m2fn": "fp8_e5m2", "fp8e5m2": "fp8_e5m2",
    }
    return aliases.get(name, name)


__all__ = [
    "bf16_to_fp32", "fp32_to_bf16", "bf16_to_fp16",
    "fp8_to_fp32", "fp8_to_fp16",
    "int8_to_fp16",
    "DTYPE_SIZE", "get_dtype_size", "normalize_dtype_name",
]