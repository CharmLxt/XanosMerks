"""Маппинги типов данных для всех форматов моделей."""
import numpy as np

# ============================================================
# SAFETENSORS dtype -> numpy dtype
# ============================================================
SAFETENSORS_TO_NUMPY = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,   # BFloat16 — читаем как uint16, при необходимости конвертим
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8":  np.int8,
    "U8":  np.uint8,
    "BOOL": np.bool_,
    "FP8_E4M3": np.uint8,   # FP8 читаем как uint8, конвертер отдельно
    "FP8_E5M2": np.uint8,
}

# Размер dtype в байтах (для safetensors и gguf)
DTYPE_SIZE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
    "BOOL": 1, "FP8_E4M3": 1, "FP8_E5M2": 1,
}

# ============================================================
# GGUF tensor types (числа из спецификации GGML)
# ============================================================
# Все квантованные типы требуют декодирования в float32 для дальнейшей работы.
GGUF_TENSOR_TYPES = {
    0:  "F32",        # 4 байта на элемент
    1:  "F16",        # 2 байта
    2:  "Q4_0",       # 18 байт на 32 элемента
    3:  "Q4_1",       # 20 байт на 32 элемента
    6:  "Q5_0",       # 22 байта на 32 элемента
    7:  "Q5_1",       # 24 байта на 32 элемента
    8:  "Q8_0",       # 34 байта на 32 элемента
    9:  "Q8_1",       # 40 байт на 32 элемента
    10: "Q2_K",       # квантование K-quants
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
}

# Размеры в байтах для GGUF-типов (на один "блок")
GGUF_TYPE_BLOCK_SIZE = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "F64": (1, 8),
    "I8": (1, 1), "I16": (1, 2), "I32": (1, 4), "I64": (1, 8),
    "Q4_0": (32, 18), "Q4_1": (32, 20),
    "Q5_0": (32, 22), "Q5_1": (32, 24),
    "Q8_0": (32, 34), "Q8_1": (32, 40),
    "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144),
    "Q5_K": (256, 176), "Q6_K": (256, 210), "Q8_K": (256, 292),
    "IQ2_XXS": (256, 66), "IQ2_XS": (256, 74), "IQ3_XXS": (256, 98),
    "IQ1_S": (256, 50), "IQ4_NL": (32, 18), "IQ3_S": (256, 110),
    "IQ2_S": (256, 82), "IQ4_XS": (256, 136), "IQ1_M": (256, 56),
    "TQ1_0": (256, 54), "TQ2_0": (256, 66),
}

# ============================================================
# TORCH dtype (строки из HF config) -> numpy
# ============================================================
TORCH_TO_NUMPY = {
    "torch.float32": np.float32,
    "torch.float16": np.float16,
    "torch.bfloat16": np.uint16,
    "torch.float64": np.float64,
    "torch.int64": np.int64,
    "torch.int32": np.int32,
    "torch.int16": np.int16,
    "torch.int8": np.int8,
    "torch.uint8": np.uint8,
    "torch.bool": np.bool_,
    # Короткие алиасы из config.json
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": np.uint16,
    "float64": np.float64,
}


def get_numpy_dtype(dtype_str):
    """Универсальный маппинг: строка dtype → numpy-тип. Пробует все источники."""
    if dtype_str in SAFETENSORS_TO_NUMPY:
        return SAFETENSORS_TO_NUMPY[dtype_str]
    if dtype_str in TORCH_TO_NUMPY:
        return TORCH_TO_NUMPY[dtype_str]
    raise ValueError(f"Неизвестный dtype: {dtype_str}")


def get_dtype_size(dtype_str):
    """Размер одного элемента в байтах."""
    if dtype_str in DTYPE_SIZE_BYTES:
        return DTYPE_SIZE_BYTES[dtype_str]
    raise ValueError(f"Неизвестный размер для dtype: {dtype_str}")


def gguf_type_name(type_id):
    """Возвращает читаемое имя для числового ID GGUF-типа."""
    return GGUF_TENSOR_TYPES.get(type_id, f"UNKNOWN_{type_id}")


def gguf_tensor_size_in_bytes(type_id, num_elements):
    """Вычисляет, сколько байт занимает тензор в GGUF с учётом блочного квантования."""
    name = gguf_type_name(type_id)
    if name not in GGUF_TYPE_BLOCK_SIZE:
        raise ValueError(f"Неизвестный GGUF-тип: {name}")
    block_size, block_bytes = GGUF_TYPE_BLOCK_SIZE[name]
    num_blocks = (num_elements + block_size - 1) // block_size
    return num_blocks * block_bytes