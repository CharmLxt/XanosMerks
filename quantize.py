"""Quantizer — квантизация тензоров в INT8 и FP8 с автоматическим подбором scale."""
import threading
import numpy as np

from .support_tools.core_spr.exceptions import (
    QuantizationError,
    UnsupportedSchemeError,
)

_FP8_MAX_VALUES = {
    "e4m3": 448.0,
    "e5m2": 57344.0,
}

_ENCODE_TABLES = {}
_ENCODE_TABLES_LOCK = threading.Lock()


def _get_encode_table(variant):
    with _ENCODE_TABLES_LOCK:
        if variant in _ENCODE_TABLES:
            return _ENCODE_TABLES[variant]
        from xanosmerks.meanflare.support_tools.core_spr.dtype_ops import (
            _get_fp8_table,
        )
        table = _get_fp8_table(variant)
        codes = np.arange(256)
        mask = np.isfinite(table)
        values = table[mask]
        codes = codes[mask]
        order = np.argsort(values)
        sorted_values = values[order].astype(np.float32)
        sorted_codes = codes[order].astype(np.uint8)
        _ENCODE_TABLES[variant] = (sorted_values, sorted_codes)
        return _ENCODE_TABLES[variant]


def _encode_fp8(x_scaled, variant):
    values, codes = _get_encode_table(variant)
    x_clip = np.clip(x_scaled, values[0], values[-1])
    pos = np.searchsorted(values, x_clip)
    pos = np.clip(pos, 1, len(values) - 1)
    left = pos - 1
    right = pos
    dist_l = np.abs(x_clip - values[left])
    dist_r = np.abs(x_clip - values[right])

    code_l = codes[left].astype(np.int32)
    code_r = codes[right].astype(np.int32)

    left_even = (code_l & 1) == 0
    right_even = (code_r & 1) == 0

    use_left = dist_l < dist_r
    use_right = dist_r < dist_l
    tie = ~use_left & ~use_right

    choose_left = use_left | (tie & left_even & ~right_even)
    best = np.where(choose_left, left, right)

    return codes[best]


def _tolist(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return x
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


class Quantizer:
    """Квантизация и деквантизация тензоров."""

    def __init__(self):
        pass

    # ---------- INT8 ----------

    def quantize_int8(self, tensor, scheme="per_channel",
                      symmetric=True, group_size=128):
        """Квантизует тензор в INT8.

        Схемы:
            per_tensor  — один scale на весь тензор
            per_channel — свой scale на каждую строку (axis=0)
            per_group   — свой scale на каждую группу из group_size элементов
                          вдоль последней оси. Для 2D тензора (out, in)
                          scales имеет форму (out, n_groups), где
                          n_groups = ceil(in / group_size).
                          Для 1D fallback на per_tensor.
        """
        if scheme not in ("per_tensor", "per_channel", "per_group"):
            raise UnsupportedSchemeError(f"scheme={scheme}")
        x = np.asarray(tensor, dtype=np.float32)
        if x.size == 0:
            return np.zeros(x.shape, dtype=np.int8), 1.0, 0

        if scheme == "per_channel" and x.ndim >= 2:
            return self._quantize_int8_per_channel(x, symmetric)
        if scheme == "per_group" and x.ndim >= 1 and group_size >= 1:
            return self._quantize_int8_per_group(x, group_size, symmetric)
        return self._quantize_int8_per_tensor(x, symmetric)

    def _quantize_int8_per_tensor(self, x, symmetric):
        if symmetric:
            amax = float(np.max(np.abs(x)))
            scale = amax / 127.0 if amax > 0 else 1.0
            q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
            return q, float(scale), 0

        xmin = float(np.min(x))
        xmax = float(np.max(x))
        if xmax == xmin:
            return np.full(x.shape, -128, dtype=np.int8), 1.0, 0
        scale = (xmax - xmin) / 255.0
        zero_point = int(np.clip(-128 - round(xmin / scale), -128, 127))
        q = np.clip(np.round(x / scale) + zero_point, -128, 127).astype(np.int8)
        return q, float(scale), zero_point

    def _quantize_int8_per_channel(self, x, symmetric):
        axes = tuple(range(1, x.ndim))
        shape_s = (x.shape[0],) + (1,) * (x.ndim - 1)

        if symmetric:
            amax = np.max(np.abs(x), axis=axes)
            scales = np.where(amax > 0, amax / 127.0, np.float32(1.0)).astype(np.float32)
            zero_points = np.zeros_like(scales, dtype=np.int32)
            q = np.clip(np.round(x / scales.reshape(shape_s)), -127, 127).astype(np.int8)
            return q, scales, zero_points

        xmin = np.min(x, axis=axes)
        xmax = np.max(x, axis=axes)
        rng = xmax - xmin
        safe_rng = np.where(rng > 0, rng, np.float32(1.0))
        scales = (safe_rng / 255.0).astype(np.float32)
        zero_points = np.clip(
            -128 - np.round(xmin / scales), -128, 127
        ).astype(np.int32)
        q = np.clip(
            np.round(x / scales.reshape(shape_s)) + zero_points.reshape(shape_s),
            -128, 127,
        ).astype(np.int8)
        return q, scales, zero_points

    def _quantize_int8_per_group(self, x, group_size, symmetric):
        """Per-group: свой scale на каждую группу из group_size элементов
        вдоль последней оси. Для 2D формы (out, in) → scales (out, n_groups)."""
        orig_shape = x.shape
        in_features = orig_shape[-1]

        if in_features < 2:
            return self._quantize_int8_per_tensor(x, symmetric)

        n_groups = (in_features + group_size - 1) // group_size
        pad = n_groups * group_size - in_features

        flat = x.reshape(-1, in_features).astype(np.float32)
        if pad > 0:
            padded = np.pad(flat, ((0, 0), (0, pad)), mode="constant")
        else:
            padded = flat

        grouped = padded.reshape(flat.shape[0], n_groups, group_size)

        if symmetric:
            amax = np.max(np.abs(grouped), axis=-1)
            scales = np.where(amax > 0, amax / 127.0, np.float32(1.0)).astype(np.float32)
            zero_points = np.zeros_like(scales, dtype=np.int32)
            q = np.clip(
                np.round(grouped / scales[:, :, None]), -127, 127
            ).astype(np.int8)
        else:
            xmin = np.min(grouped, axis=-1)
            xmax = np.max(grouped, axis=-1)
            rng = xmax - xmin
            safe_rng = np.where(rng > 0, rng, np.float32(1.0))
            scales = (safe_rng / 255.0).astype(np.float32)
            zero_points = np.clip(
                -128 - np.round(xmin / scales), -128, 127
            ).astype(np.int32)
            q = np.clip(
                np.round(grouped / scales[:, :, None]) + zero_points[:, :, None],
                -128, 127,
            ).astype(np.int8)

        q = q.reshape(flat.shape[0], n_groups * group_size)
        if pad > 0:
            q = q[:, :in_features]
        q = q.reshape(orig_shape)

        if len(orig_shape) == 1:
            return q, scales[0], zero_points[0]
        return q, scales, zero_points

    def dequantize_int8(self, q_array, scales, zero_points, symmetric=True):
        """INT8 → float32. Поддерживает per_tensor / per_channel / per_group."""
        q = np.asarray(q_array, dtype=np.float32)

        if np.ndim(scales) == 0:
            if symmetric or zero_points is None:
                zp_val = 0.0
            else:
                zp_arr = np.asarray(zero_points)
                if zp_arr.ndim != 0:
                    raise QuantizationError(
                        f"scalar scale требует scalar zero_point, получено {zp_arr.shape}"
                    )
                zp_val = float(zp_arr)
            return (q - zp_val) * float(scales)

        scales_arr = np.asarray(scales, dtype=np.float32)

        if scales_arr.ndim == 2:
            # per_group
            out_features, n_groups = scales_arr.shape
            in_features = q.shape[-1]
            group_size = (in_features + n_groups - 1) // n_groups

            flat = q.reshape(-1, in_features)
            if flat.shape[0] != out_features:
                raise QuantizationError(
                    f"per_group scales.shape={scales_arr.shape} не совпадает с q {q.shape}"
                )

            padded_len = n_groups * group_size
            padded = np.zeros((out_features, padded_len), dtype=np.float32)
            padded[:, :in_features] = flat
            grouped = padded.reshape(out_features, n_groups, group_size)

            if symmetric or zero_points is None:
                zp_arr = np.zeros_like(scales_arr)
            else:
                zp_arr = np.asarray(zero_points, dtype=np.float32)
                if zp_arr.shape != scales_arr.shape:
                    raise QuantizationError(
                        f"per_group zero_points.shape={zp_arr.shape} "
                        f"!= scales.shape={scales_arr.shape}"
                    )

            restored = (grouped - zp_arr[:, :, None]) * scales_arr[:, :, None]
            restored = restored.reshape(out_features, padded_len)[:, :in_features]
            return restored.reshape(q.shape)

        # per_channel
        if scales_arr.shape[0] != q.shape[0]:
            raise QuantizationError(
                f"per_channel scales.shape[0]={scales_arr.shape[0]} != q.shape[0]={q.shape[0]}"
            )

        if symmetric or zero_points is None:
            zp_arr = np.zeros_like(scales_arr)
        else:
            zp_arr = np.asarray(zero_points, dtype=np.float32)
            if zp_arr.shape != scales_arr.shape:
                raise QuantizationError(
                    f"per_channel zero_points.shape={zp_arr.shape} "
                    f"!= scales.shape={scales_arr.shape}"
                )

        shape_s = (q.shape[0],) + (1,) * (q.ndim - 1)
        return (q - zp_arr.reshape(shape_s)) * scales_arr.reshape(shape_s)

    # ---------- FP8 ----------

    def quantize_fp8(self, tensor, variant="e4m3"):
        if variant not in _FP8_MAX_VALUES:
            raise UnsupportedSchemeError(f"variant={variant}")
        x = np.asarray(tensor, dtype=np.float32)
        max_val = _FP8_MAX_VALUES[variant]
        if x.size == 0:
            return np.zeros(x.shape, dtype=np.uint8), 1.0
        amax = float(np.max(np.abs(x)))
        scale = amax / max_val if amax > 0 else 1.0
        x_scaled = x / scale
        q = _encode_fp8(x_scaled, variant)
        return q, float(scale)

    def dequantize_fp8(self, q_array, scales, variant="e4m3"):
        from xanosmerks.meanflare.support_tools.core_spr.dtype_ops import (
            fp8_to_fp32,
        )
        q = np.asarray(q_array, dtype=np.uint8)
        decoded = fp8_to_fp32(q, variant=variant)
        return decoded * float(scales)

    # ---------- Универсальный ----------

    def quantize(self, tensor, target_dtype, scheme="per_channel",
                 symmetric=True, group_size=128):
        if target_dtype == "int8":
            q, scales, zp = self.quantize_int8(
                tensor, scheme=scheme, symmetric=symmetric, group_size=group_size,
            )
            return q, {
                "scales": scales, "zero_points": zp,
                "symmetric": symmetric, "scheme": scheme,
                "group_size": group_size if scheme == "per_group" else None,
            }
        if target_dtype == "fp8_e4m3":
            q, scales = self.quantize_fp8(tensor, variant="e4m3")
            return q, {"scales": scales, "zero_points": None, "symmetric": True}
        if target_dtype == "fp8_e5m2":
            q, scales = self.quantize_fp8(tensor, variant="e5m2")
            return q, {"scales": scales, "zero_points": None, "symmetric": True}
        raise UnsupportedSchemeError(f"target_dtype={target_dtype}")

    def dequantize(self, q_array, target_dtype, params):
        if target_dtype == "int8":
            return self.dequantize_int8(
                q_array,
                params["scales"],
                params.get("zero_points", 0),
                symmetric=params.get("symmetric", True),
            )
        if target_dtype == "fp8_e4m3":
            return self.dequantize_fp8(q_array, params["scales"], variant="e4m3")
        if target_dtype == "fp8_e5m2":
            return self.dequantize_fp8(q_array, params["scales"], variant="e5m2")
        raise UnsupportedSchemeError(f"target_dtype={target_dtype}")

    def __repr__(self):
        return "<Quantizer int8=sym/asym/per_group fp8=e4m3/e5m2 rne>"


__all__ = ["Quantizer", "_tolist"]