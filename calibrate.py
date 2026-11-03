"""Калибровка: подбор scale/zero_point по всей модели."""
import json
from pathlib import Path

import numpy as np

from .quantize import Quantizer, _tolist
from .support_tools.core_spr.exceptions import (
    CalibrationError,
    UnsupportedSchemeError,
)


def calibrate(tensors, scheme="per_channel", target_dtype="int8",
              quantizer=None, symmetric=True, group_size=128):
    """Подбирает scale/zero_point для каждого тензора. Не квантизует.

    scheme: "per_tensor" | "per_channel" | "per_group"
    group_size: используется только при scheme="per_group"
    """
    if scheme not in ("per_tensor", "per_channel", "per_group"):
        raise UnsupportedSchemeError(f"scheme={scheme}")
    if target_dtype not in ("int8", "fp8_e4m3", "fp8_e5m2"):
        raise UnsupportedSchemeError(f"target_dtype={target_dtype}")

    q = quantizer or Quantizer()
    result = {}
    for name, t in tensors.items():
        try:
            if target_dtype == "int8":
                _, scales, zp = q.quantize_int8(
                    t, scheme=scheme, symmetric=symmetric, group_size=group_size,
                )
            else:
                variant = "e4m3" if target_dtype == "fp8_e4m3" else "e5m2"
                _, scales = q.quantize_fp8(t, variant=variant)
                zp = None
            result[name] = {
                "scales": _tolist(scales),
                "zero_points": _tolist(zp),
                "scheme": scheme,
                "target_dtype": target_dtype,
                "symmetric": bool(symmetric) if target_dtype == "int8" else True,
                "group_size": int(group_size) if scheme == "per_group" else None,
                "shape": list(t.shape),
            }
        except Exception as e:
            raise CalibrationError(f"tensor '{name}': {e}") from e

    return result


def save_calibration(calibration, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 3, "tensors": calibration}
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def load_calibration(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text("utf-8"))
    except Exception:
        return None
    return data.get("tensors", {})


__all__ = ["calibrate", "save_calibration", "load_calibration"]