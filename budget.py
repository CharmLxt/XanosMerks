"""Выбор dtype по бюджету памяти и пакетная конвертация по политике."""
import numpy as np

from .quantize import Quantizer, _tolist
from .support_tools.core_spr.exceptions import BudgetError
from .support_tools.sup2_spr.schemes import (
    DEFAULT_BUDGET_CANDIDATES,
    DEFAULT_POLICY,
)

_QUANT_DTYPES = ("int8", "fp8_e4m3", "fp8_e5m2")


def estimate_quantized_size(tensors, target_dtype):
    if target_dtype in _QUANT_DTYPES:
        total = 0
        for t in tensors.values():
            arr = np.asarray(t)
            total += arr.size
            if arr.ndim >= 1:
                total += arr.shape[0] * 4 * 2
        return total

    try:
        from xanosmerks.meanflare.convert import estimate_converted_size
        return estimate_converted_size(tensors, target_dtype)
    except Exception:
        bytes_per = {
            "float32": 4, "float16": 2, "bfloat16": 2,
            "int64": 8, "int32": 4,
        }.get(target_dtype, 4)
        return sum(np.asarray(t).size * bytes_per for t in tensors.values())


def choose_dtype_for_budget(tensors, max_bytes, candidates=None):
    if candidates is None:
        candidates = DEFAULT_BUDGET_CANDIDATES
    sizes = {}
    for dt in candidates:
        sizes[dt] = estimate_quantized_size(tensors, dt)
        if sizes[dt] <= max_bytes:
            return dt
    min_dt = min(sizes, key=sizes.get)
    raise BudgetError(
        f"Ни один dtype из {candidates} не влезает в {max_bytes} байт. "
        f"Минимальный ({min_dt}) требует {sizes[min_dt]} байт."
    )


def _match_policy(name, policy):
    for pattern, dtype in policy:
        if pattern in name:
            return dtype
    return None


def convert_with_policy(tensors, policy=None, quantizer=None,
                        symmetric_int8=True):
    if policy is None:
        policy = DEFAULT_POLICY
    q = quantizer or Quantizer()

    new_tensors = {}
    stats = {
        "converted": 0,
        "skipped": 0,
        "per_dtype": {},
        "quant_params": {},
        "bytes_before": 0,
        "bytes_after": 0,
    }

    for name, t in tensors.items():
        arr = np.asarray(t)
        stats["bytes_before"] += arr.nbytes
        target = _match_policy(name, policy)
        if target is None:
            new_tensors[name] = t
            stats["skipped"] += 1
            stats["bytes_after"] += arr.nbytes
            continue

        try:
            if target == "int8":
                q_arr, scales, zp = q.quantize_int8(
                    arr, scheme="per_channel", symmetric=symmetric_int8,
                )
                restored = q.dequantize_int8(
                    q_arr, scales, zp, symmetric=symmetric_int8,
                )
                new_tensors[name] = restored.astype(np.float16)
                stats["quant_params"][name] = {
                    "target_dtype": "int8",
                    "scheme": "per_channel",
                    "symmetric": bool(symmetric_int8),
                    "scales": _tolist(scales),
                    "zero_points": _tolist(zp),
                }
            elif target in ("fp8_e4m3", "fp8_e5m2"):
                variant = "e4m3" if target == "fp8_e4m3" else "e5m2"
                q_arr, scales = q.quantize_fp8(arr, variant=variant)
                restored = q.dequantize_fp8(q_arr, scales, variant=variant)
                new_tensors[name] = restored.astype(np.float16)
                stats["quant_params"][name] = {
                    "target_dtype": target,
                    "variant": variant,
                    "symmetric": True,
                    "scales": float(scales),
                }
            elif target in ("float32", "float16", "bfloat16"):
                np_dtype = {
                    "float32": np.float32,
                    "float16": np.float16,
                    "bfloat16": np.uint16,
                }[target]
                new_tensors[name] = arr.astype(np_dtype)
            else:
                new_tensors[name] = t
                stats["skipped"] += 1
                stats["bytes_after"] += arr.nbytes
                continue

            stats["converted"] += 1
            stats["per_dtype"][target] = stats["per_dtype"].get(target, 0) + 1
            stats["bytes_after"] += new_tensors[name].nbytes
        except Exception:
            new_tensors[name] = t
            stats["skipped"] += 1
            stats["bytes_after"] += arr.nbytes

    return new_tensors, stats


__all__ = [
    "choose_dtype_for_budget",
    "convert_with_policy",
    "estimate_quantized_size",
]