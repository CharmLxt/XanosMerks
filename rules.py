"""Константы fusion-правил quacdualc.

Правило — пара (op_type_1, op_type_2), которая может быть склеена в одну
операцию fused_op_name. Порядок важен: сначала ищется самое длинное
совпадение (3 ops → 2 ops → 1 op).
"""

FUSABLE_PAIRS = {
    ("rmsnorm", "linear"): "fused_rmsnorm_linear",
    ("add", "rmsnorm"): "fused_residual_norm",
    ("silu", "mul"): "fused_swiglu",
    ("linear", "silu"): "fused_linear_silu",
    ("layer_norm", "linear"): "fused_layernorm_linear",
}

KNOWN_OPS = (
    "input",
    "output",
    "rmsnorm",
    "layer_norm",
    "linear",
    "silu",
    "gelu",
    "mul",
    "add",
    "swiglu",
    "fused_rmsnorm_linear",
    "fused_layernorm_linear",
    "fused_residual_norm",
    "fused_swiglu",
    "fused_linear_silu",
    "softmax",
    "attention",
    "embedding",
    "matmul",
    "split",
    "concat",
    "identity",
)

# Оценка ускорения для отчёта (на основе бенчмарков на типичных размерах).
# Это НЕ точное предсказание, а ориентир для сортировки паттернов в отчёте.
ESTIMATED_SPEEDUP = {
    "fused_rmsnorm_linear": 0.05,
    "fused_layernorm_linear": 0.05,
    "fused_residual_norm": 0.03,
    "fused_swiglu": 0.02,
    "fused_linear_silu": 0.02,
}

# Порог: считаем операцию "тяжёлой", если её выходной тензор больше N байт.
HEAVY_OP_THRESHOLD_BYTES = 64 * 1024


__all__ = [
    "FUSABLE_PAIRS",
    "KNOWN_OPS",
    "ESTIMATED_SPEEDUP",
    "HEAVY_OP_THRESHOLD_BYTES",
]