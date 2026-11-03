"""Операции quacdualc: базовые и fused (operator fusion)."""
from abc import ABC, abstractmethod
import numpy as np


class Op(ABC):
    """Базовый класс операции.

    op_type — короткое имя ("rmsnorm", "linear", ...) — по нему работает fusion.
    forward(*inputs) — обязательный метод, реализуется в наследниках.
    Прямое создание Op() запрещено — это абстрактный класс.
    """
    op_type = "identity"

    @abstractmethod
    def forward(self, *inputs, **params):
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__} op_type={self.op_type}>"


class RMSNormOp(Op):
    op_type = "rmsnorm"

    def __init__(self, weight, eps=1e-5):
        self.weight = np.asarray(weight, dtype=np.float32)
        self.eps = eps

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


class LayerNormOp(Op):
    op_type = "layer_norm"

    def __init__(self, weight, bias=None, eps=1e-5):
        self.weight = np.asarray(weight, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32) if bias is not None else None
        self.eps = eps

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normed = (x - mean) / np.sqrt(var + self.eps)
        out = normed * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class LinearOp(Op):
    op_type = "linear"

    def __init__(self, weight, bias=None):
        self.weight = np.asarray(weight, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32) if bias is not None else None

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class SiluOp(Op):
    op_type = "silu"

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        return x / (1.0 + np.exp(-x))


class GeluOp(Op):
    op_type = "gelu"

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


class MulOp(Op):
    op_type = "mul"

    def forward(self, a, b):
        return np.asarray(a, dtype=np.float32) * np.asarray(b, dtype=np.float32)


class AddOp(Op):
    op_type = "add"

    def forward(self, a, b):
        return np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)


class IdentityOp(Op):
    op_type = "identity"

    def forward(self, x):
        return x


class FusedRMSNormLinearOp(Op):
    op_type = "fused_rmsnorm_linear"

    def __init__(self, norm_weight, linear_weight, linear_bias=None, eps=1e-5):
        self.norm_weight = np.asarray(norm_weight, dtype=np.float32)
        self.linear_weight = np.asarray(linear_weight, dtype=np.float32)
        self.linear_bias = np.asarray(linear_bias, dtype=np.float32) if linear_bias is not None else None
        self.eps = eps

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        normed = (x / rms) * self.norm_weight
        out = normed @ self.linear_weight.T
        if self.linear_bias is not None:
            out = out + self.linear_bias
        return out


class FusedLayerNormLinearOp(Op):
    op_type = "fused_layernorm_linear"

    def __init__(self, norm_weight, norm_bias, linear_weight, linear_bias=None, eps=1e-5):
        self.norm_weight = np.asarray(norm_weight, dtype=np.float32)
        self.norm_bias = np.asarray(norm_bias, dtype=np.float32) if norm_bias is not None else None
        self.linear_weight = np.asarray(linear_weight, dtype=np.float32)
        self.linear_bias = np.asarray(linear_bias, dtype=np.float32) if linear_bias is not None else None
        self.eps = eps

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normed = (x - mean) / np.sqrt(var + self.eps)
        normed = normed * self.norm_weight
        if self.norm_bias is not None:
            normed = normed + self.norm_bias
        out = normed @ self.linear_weight.T
        if self.linear_bias is not None:
            out = out + self.linear_bias
        return out


class FusedResidualNormOp(Op):
    op_type = "fused_residual_norm"

    def __init__(self, norm_weight, eps=1e-5):
        self.norm_weight = np.asarray(norm_weight, dtype=np.float32)
        self.eps = eps

    def forward(self, x, residual):
        x = np.asarray(x, dtype=np.float32)
        residual = np.asarray(residual, dtype=np.float32)
        h = x + residual
        rms = np.sqrt(np.mean(h ** 2, axis=-1, keepdims=True) + self.eps)
        return (h / rms) * self.norm_weight


class FusedSwiGLUOp(Op):
    op_type = "fused_swiglu"

    def forward(self, gate, up):
        gate = np.asarray(gate, dtype=np.float32)
        up = np.asarray(up, dtype=np.float32)
        return (gate / (1.0 + np.exp(-gate))) * up


class FusedLinearSiluOp(Op):
    op_type = "fused_linear_silu"

    def __init__(self, weight, bias=None):
        self.weight = np.asarray(weight, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32) if bias is not None else None

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out / (1.0 + np.exp(-out))


__all__ = [
    "Op",
    "RMSNormOp", "LayerNormOp", "LinearOp",
    "SiluOp", "GeluOp", "MulOp", "AddOp", "IdentityOp",
    "FusedRMSNormLinearOp", "FusedLayerNormLinearOp",
    "FusedResidualNormOp", "FusedSwiGLUOp", "FusedLinearSiluOp",
]