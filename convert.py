"""Конвертация dtype тензоров и деквантование."""
import warnings
import numpy as np

from .support_tools.core_spr.exceptions import UnsupportedDTypeError
from .support_tools.core_spr.dtype_ops import (
    bf16_to_fp32 as _bf16_to_fp32,
    fp32_to_bf16 as _fp32_to_bf16,
    bf16_to_fp16 as _bf16_to_fp16,
    fp8_to_fp32 as _fp8_to_fp32,
    fp8_to_fp16 as _fp8_to_fp16,
    int8_to_fp16 as _int8_to_fp16,
    normalize_dtype_name,
    DTYPE_SIZE,
)
from .support_tools.core_spr.gguf_dequant import dequantize as _gguf_dequantize


# ============================================================
# ОБЁРТКИ НАД dtype_ops
# ============================================================
def bf16_to_fp32(array):
    return _bf16_to_fp32(array)


def fp32_to_bf16(array, rounding=True):
    return _fp32_to_bf16(array, rounding=rounding)


def bf16_to_fp16(array):
    return _bf16_to_fp16(array)


def fp8_to_fp32(array, variant="e4m3"):
    return _fp8_to_fp32(array, variant=variant)


def fp8_to_fp16(array, variant="e4m3"):
    return _fp8_to_fp16(array, variant=variant)


def int8_to_fp16(array, scale=1.0, zero_point=0):
    return _int8_to_fp16(array, scale=scale, zero_point=zero_point)


def dequantize_gguf_tensor(raw_bytes, tensor_type, shape):
    return _gguf_dequantize(raw_bytes, tensor_type, shape)


# ============================================================
# ОПРЕДЕЛЕНИЕ SOURCE DTYPE
# ============================================================
def _guess_source_dtype(array, tensor_name=None):
    """Определяет source dtype по numpy.dtype.

    uint16 → bfloat16, uint8 → fp8_e4m3 (наиболее вероятное).
    Это эвристика: если у вас в uint16/uint8 лежат не BF16/FP8,
    передайте source_dtype_map в convert_model или source_dtype в convert_dtype.

    Вызывает warnings.warn, если пришлось угадывать.
    """
    name = array.dtype.name
    if name == "uint16":
        warnings.warn(
            f"_guess_source_dtype: uint16 тензор "
            f"{tensor_name or '<unnamed>'} интерпретируется как bfloat16. "
            f"Если это не BF16 — передайте source_dtype_map / source_dtype.",
            UserWarning,
            stacklevel=3,
        )
        return "bfloat16"
    if name == "uint8":
        warnings.warn(
            f"_guess_source_dtype: uint8 тензор "
            f"{tensor_name or '<unnamed>'} интерпретируется как fp8_e4m3. "
            f"Если это не FP8 — передайте source_dtype_map / source_dtype.",
            UserWarning,
            stacklevel=3,
        )
        return "fp8_e4m3"
    return normalize_dtype_name(name)


# ============================================================
# КОНВЕРТАЦИЯ ОДНОГО ТЕНЗОРА
# ============================================================
def convert_dtype(array, target_dtype, source_dtype=None):
    """Конвертирует один numpy-массив в target_dtype.

    source_dtype: явный source ("bfloat16", "fp8_e4m3", "float32", ...).
    Если None — используется _guess_source_dtype (с предупреждением).
    """
    target = normalize_dtype_name(target_dtype) if target_dtype != "auto" else "auto"

    if target == "auto":
        return array

    source = (
        normalize_dtype_name(source_dtype)
        if source_dtype
        else _guess_source_dtype(array)
    )

    if source == target:
        return array

    if source == "bfloat16":
        if target == "float32":
            return bf16_to_fp32(array)
        if target == "float16":
            return bf16_to_fp16(array)
        raise UnsupportedDTypeError(f"BF16 → {target}: не поддерживается")

    if source in ("fp8_e4m3", "fp8_e5m2"):
        variant = "e4m3" if source == "fp8_e4m3" else "e5m2"
        if target == "float32":
            return fp8_to_fp32(array, variant=variant)
        if target == "float16":
            return fp8_to_fp16(array, variant=variant)
        raise UnsupportedDTypeError(f"{source} → {target}: не поддерживается")

    if source == "int8" and target in ("float16", "float32"):
        if target == "float16":
            return int8_to_fp16(array)
        return array.astype(np.float32)

    try:
        return array.astype(np.dtype(target))
    except Exception as e:
        raise UnsupportedDTypeError(f"{source} → {target}: {e}")


# ============================================================
# КОНВЕРТАЦИЯ ВСЕЙ МОДЕЛИ
# ============================================================
def convert_model(tensors, target_dtype, source_dtype_map=None):
    """Конвертирует все тензоры словаря в target_dtype.

    source_dtype_map: dict {name: source_dtype} — явный маппинг (например, из
        meta["source_dtypes"]). Если передан, эвристика не используется.
        Если None — для uint16/uint8 тензоров будет warnings.warn.

    Возвращает:
        (result_tensors, stats)
    """
    result = {}
    stats = {
        "converted": 0,
        "skipped": 0,
        "total_bytes_before": 0,
        "total_bytes_after": 0,
    }

    for name, arr in tensors.items():
        stats["total_bytes_before"] += arr.nbytes
        src = source_dtype_map.get(name) if source_dtype_map else None
        try:
            converted = convert_dtype(arr, target_dtype, source_dtype=src)
        except UnsupportedDTypeError:
            converted = arr
            stats["skipped"] += 1
        result[name] = converted
        stats["total_bytes_after"] += converted.nbytes
        if converted is not arr:
            stats["converted"] += 1

    if stats["total_bytes_before"] > 0:
        stats["compression_ratio"] = (
            stats["total_bytes_after"] / stats["total_bytes_before"]
        )
    else:
        stats["compression_ratio"] = 1.0

    return result, stats


def auto_cast(tensors, target_dtype, source_dtype_map=None):
    """Умный каст: конвертирует все тензоры в target_dtype."""
    return convert_model(tensors, target_dtype, source_dtype_map=source_dtype_map)[0]


# ============================================================
# АВТООПРЕДЕЛЕНИЕ ОПТИМАЛЬНОГО DTYPE
# ============================================================
def detect_optimal_dtype(tensors, source_dtype_map=None):
    """Анализирует dtype всех тензоров и предлагает оптимальный."""
    if not tensors:
        return "float16"

    dtypes = set()
    for name, arr in tensors.items():
        if source_dtype_map and name in source_dtype_map:
            dtypes.add(normalize_dtype_name(source_dtype_map[name]))
        else:
            dtypes.add(_guess_source_dtype(arr, tensor_name=name))

    if len(dtypes) == 1:
        only = dtypes.pop()
        if only == "float32":
            return "float16"
        return only

    return "float16"


# ============================================================
# ОЦЕНКА РАЗМЕРА
# ============================================================
def estimate_converted_size(tensors, target_dtype):
    target = normalize_dtype_name(target_dtype)
    if target not in DTYPE_SIZE:
        raise UnsupportedDTypeError(f"Неизвестный dtype: {target}")
    total_elements = sum(arr.size for arr in tensors.values())
    return total_elements * DTYPE_SIZE[target]


def compare_dtype_sizes(tensors):
    result = {}
    for dt in ("float32", "float16", "bfloat16", "int8", "fp8_e4m3"):
        try:
            result[dt] = estimate_converted_size(tensors, dt)
        except UnsupportedDTypeError:
            continue
    return result


def describe_conversion(tensors, target_dtype):
    before = sum(arr.nbytes for arr in tensors.values())
    after = estimate_converted_size(tensors, target_dtype)
    ratio = after / before if before > 0 else 1.0

    def _human(n):
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n:.2f} {unit}"
            n /= 1024
        return f"{n:.2f} TB"

    return {
        "target_dtype": target_dtype,
        "before_bytes": before,
        "before_human": _human(before),
        "after_bytes": after,
        "after_human": _human(after),
        "ratio": round(ratio, 3),
        "savings_pct": round((1 - ratio) * 100, 1) if ratio < 1 else 0.0,
    }


__all__ = [
    "bf16_to_fp32", "fp32_to_bf16", "bf16_to_fp16",
    "fp8_to_fp32", "fp8_to_fp16", "int8_to_fp16",
    "dequantize_gguf_tensor",
    "convert_dtype", "convert_model", "auto_cast",
    "detect_optimal_dtype",
    "estimate_converted_size", "compare_dtype_sizes", "describe_conversion",
]