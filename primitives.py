"""Математические примитивы для forward-pass."""
import numpy as np

try:
    from ninewas9 import fast_softmax, log_softmax
except ImportError:
    def fast_softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        x_max = np.where(np.isfinite(x_max), x_max, 0.0)
        exp_x = np.exp(x - x_max)
        return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-12)
    def log_softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        x_max = np.where(np.isfinite(x_max), x_max, 0.0)
        return (x - x_max) - np.log(
            np.sum(np.exp(x - x_max), axis=axis, keepdims=True) + 1e-12
        )


def rms_norm(x, weight, eps=1e-5):
    """RMSNorm: x / sqrt(mean(x^2) + eps) * weight."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def layer_norm(x, weight, bias=None, eps=1e-5):
    """LayerNorm с центрированием."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    normed = (x - mean) / np.sqrt(var + eps)
    out = normed * weight
    if bias is not None:
        out = out + bias
    return out


def silu(x):
    """SiLU (Swish): x * sigmoid(x)."""
    return x / (1.0 + np.exp(-x))


def swiglu(gate, up):
    """SwiGLU: silu(gate) * up."""
    return silu(gate) * up


def gelu(x, approximate=True):
    """GELU активация (tanh-приближение)."""
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def apply_activation(x, name="silu"):
    """Универсальный вызов активации по имени."""
    if name in ("silu", "swish"):
        return silu(x)
    if name == "gelu":
        return gelu(x)
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "identity":
        return x
    raise ValueError(f"Неизвестная активация: {name}")


__all__ = [
    "fast_softmax", "log_softmax",
    "rms_norm", "layer_norm",
    "silu", "swiglu", "gelu", "apply_activation",
]