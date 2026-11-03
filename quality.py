"""Оценка качества квантизации: MSE, MAE, cosine, SNR, max_abs_error."""
import numpy as np

from .support_tools.core_spr.exceptions import (
    QualityBelowThresholdError,
    VofcfError,
)
from .support_tools.sup2_spr.schemes import DEFAULT_COSINE_THRESHOLD


def evaluate_quality(original, reconstructed):
    """Сравнивает исходный и восстановленный тензор."""
    o = np.asarray(original)
    r = np.asarray(reconstructed)
    if o.shape != r.shape:
        raise VofcfError(f"shape mismatch: {o.shape} vs {r.shape}")
    shape = o.shape
    o = o.astype(np.float64).flatten()
    r = r.astype(np.float64).flatten()

    if o.size == 0:
        return {
            "mse": 0.0,
            "mae": 0.0,
            "max_abs_error": 0.0,
            "cosine": 1.0,
            "snr_db": float("inf"),
            "shape": shape,
        }

    diff = o - r
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))

    norm_o = float(np.linalg.norm(o))
    norm_r = float(np.linalg.norm(r))
    if norm_o == 0.0 and norm_r == 0.0:
        cosine = 1.0
    elif norm_o == 0.0 or norm_r == 0.0:
        cosine = 0.0
    else:
        cosine = float(np.dot(o, r) / (norm_o * norm_r))

    signal_power = float(np.mean(o ** 2))
    if mse > 0.0 and signal_power > 0.0:
        snr_db = float(10.0 * np.log10(signal_power / mse))
    elif mse == 0.0:
        snr_db = float("inf")
    else:
        snr_db = float("-inf")

    return {
        "mse": mse,
        "mae": mae,
        "max_abs_error": max_abs,
        "cosine": cosine,
        "snr_db": snr_db,
        "shape": shape,
    }


def quality_report(tensors_original, tensors_reconstructed):
    """Оценка по всей модели."""
    per_tensor = {}
    cosines = []
    worst_name = None
    worst_cos = 1.0
    for name in tensors_original:
        if name not in tensors_reconstructed:
            continue
        q = evaluate_quality(
            tensors_original[name], tensors_reconstructed[name]
        )
        per_tensor[name] = q
        cosines.append(q["cosine"])
        if q["cosine"] < worst_cos:
            worst_cos = q["cosine"]
            worst_name = name
    return {
        "per_tensor": per_tensor,
        "mean_cosine": float(np.mean(cosines)) if cosines else 1.0,
        "min_cosine": float(min(cosines)) if cosines else 1.0,
        "worst_tensor": worst_name,
    }


def check_quality(quality_dict, cosine_threshold=None):
    """Проверяет, что cosine >= порога. Иначе QualityBelowThresholdError."""
    if cosine_threshold is None:
        cosine_threshold = DEFAULT_COSINE_THRESHOLD
    cos = quality_dict.get("cosine")
    if cos is None:
        return True
    if cos < cosine_threshold:
        raise QualityBelowThresholdError(
            f"cosine={cos:.6f} < threshold={cosine_threshold:.6f}"
        )
    return True


__all__ = ["evaluate_quality", "quality_report", "check_quality"]