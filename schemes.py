"""Константы схем квантизации и наборов dtype для 64vofcf32."""

SCHEMES = ("per_tensor", "per_channel")

FP8_VARIANTS = ("e4m3", "e5m2")

QUANT_DTYPES = ("int8", "fp8_e4m3", "fp8_e5m2")

DEFAULT_BUDGET_CANDIDATES = (
    "float32",
    "float16",
    "bfloat16",
    "fp8_e4m3",
    "int8",
)

DEFAULT_POLICY = (
    ("embed_tokens", "float16"),
    ("lm_head", "float16"),
    ("norm", "float16"),
    ("layernorm", "float16"),
    ("self_attn", "int8"),
    ("mlp", "int8"),
    ("block_sparse_moe", "int8"),
)

DEFAULT_COSINE_THRESHOLD = 0.99


__all__ = [
    "SCHEMES",
    "FP8_VARIANTS",
    "QUANT_DTYPES",
    "DEFAULT_BUDGET_CANDIDATES",
    "DEFAULT_POLICY",
    "DEFAULT_COSINE_THRESHOLD",
]