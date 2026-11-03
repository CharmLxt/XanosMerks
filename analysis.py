"""Анализ моделей: архитектура, валидация, профиль памяти, рекомендации."""
import json
import re
from pathlib import Path

from .support_tools.core_spr.exceptions import (
    ArchitectureError, DivisionMismatchError, KVMismatchError,
    TensorMismatchError, VocabMismatchError, MoEConfigError,
    ProfilerError, UnknownArchitectureError,
)


# ============================================================
# 1. ОПРЕДЕЛЕНИЕ КЛАССА АРХИТЕКТУРЫ
# ============================================================
ARCH_CLASS_MAP = {
    "llama": "LlamaForCausalLM",
    "mistral": "MistralForCausalLM",
    "mixtral": "MixtralForCausalLM",
    "qwen": "QWenLMHeadModel",
    "qwen2": "Qwen2ForCausalLM",
    "qwen2_moe": "Qwen2MoeForCausalLM",
    "gemma": "GemmaForCausalLM",
    "gemma2": "Gemma2ForCausalLM",
    "phi": "PhiForCausalLM",
    "phi3": "Phi3ForCausalLM",
    "falcon": "FalconForCausalLM",
    "mpt": "MptForCausalLM",
    "gpt2": "GPT2LMHeadModel",
    "gpt_neox": "GPTNeoXForCausalLM",
    "gptj": "GPTJForCausalLM",
    "command_r": "CohereForCausalLM",
    "cohere": "CohereForCausalLM",
    "yi": "YiForCausalLM",
    "zephyr": "MistralForCausalLM",
    "stablelm": "StableLmForCausalLM",
    "starcoder2": "Starcoder2ForCausalLM",
    "deepseek_v1": "LlamaForCausalLM",
    "deepseek_moe": "MixtralForCausalLM",
    "deepseek_v2": "DeepseekV2ForCausalLM",
    "deepseek_v3": "DeepseekV3ForCausalLM",
    "deepseek_r1": "DeepseekV3ForCausalLM",
    "mamba": "MambaForCausalLM",
    "rwkv": "RwkvForCausalLM",
    "llava": "LlavaForConditionalGeneration",
    "qwen2_vl": "Qwen2VLForConditionalGeneration",
    "phi3_v": "Phi3VForCausalLM",
    "idefics3": "Idefics3ForConditionalGeneration",
}


def detect_architecture_class(config):
    """Возвращает строковый идентификатор класса архитектуры."""
    arch_list = config.get("architectures", [])
    if isinstance(arch_list, list) and arch_list:
        return arch_list[0]
    if isinstance(arch_list, str):
        return arch_list

    model_type = (config.get("model_type") or "").lower()
    if model_type in ARCH_CLASS_MAP:
        return ARCH_CLASS_MAP[model_type]

    for key, cls in ARCH_CLASS_MAP.items():
        if key in model_type:
            return cls

    if config.get("is_moe"):
        return "MixtralForCausalLM"
    if config.get("is_vision"):
        return "LlavaForConditionalGeneration"
    if config.get("is_ssm"):
        return "MambaForCausalLM"

    return "UnknownForCausalLM"


# ============================================================
# 2. ВАЛИДАЦИЯ АРХИТЕКТУРЫ
# ============================================================
def validate_architecture(config):
    """Проверяет математическую и логическую целостность конфига."""
    errors = []
    warnings = []

    if config["num_heads"] > 0:
        if config["hidden_size"] % config["num_heads"] != 0:
            errors.append(
                f"hidden_size ({config['hidden_size']}) не делится на "
                f"num_heads ({config['num_heads']})."
            )
        head_dim = config.get("head_dim", 0)
        if head_dim > 0 and head_dim % 2 != 0:
            warnings.append(
                f"head_dim = {head_dim} нечётный. RoPE может работать некорректно."
            )

    if config["num_kv_heads"] > config["num_heads"]:
        errors.append(
            f"num_kv_heads ({config['num_kv_heads']}) > num_heads ({config['num_heads']})."
        )
    if config["num_kv_heads"] > 0 and config["num_heads"] > 0:
        if config["num_heads"] % config["num_kv_heads"] != 0:
            errors.append(
                f"num_heads ({config['num_heads']}) не делится на "
                f"num_kv_heads ({config['num_kv_heads']}). GQA некорректен."
            )

    if config["vocab_size"] == 0:
        errors.append("vocab_size = 0.")
    if config["num_layers"] == 0:
        errors.append("num_layers = 0.")
    if config["intermediate_size"] == 0 and not config["is_ssm"]:
        warnings.append("intermediate_size = 0, FFN может не работать.")

    if config["is_moe"]:
        if config["num_experts_per_tok"] > config["num_experts"]:
            raise MoEConfigError(
                f"num_experts_per_tok ({config['num_experts_per_tok']}) > "
                f"num_experts ({config['num_experts']})."
            )
        if config["num_experts_per_tok"] == 0:
            warnings.append("MoE: num_experts_per_tok = 0, модель не сможет выбрать экспертов.")

    if config["max_position_embeddings"] < 128:
        warnings.append(
            f"max_position_embeddings = {config['max_position_embeddings']} — очень мало."
        )

    if config["rope_theta"] <= 1.0:
        errors.append(f"rope_theta = {config['rope_theta']} должен быть > 1.0.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def validate_safetensors_match_config(config, safetensors_info):
    """Сверяет количество и размеры тензоров с ожидаемыми по конфигу."""
    errors = []
    warnings = []

    if not safetensors_info:
        return {"valid": True, "errors": [], "warnings": []}

    tensor_names = list(safetensors_info.get("tensors", {}).keys())

    has_embed = any("embed_tokens" in n or "wte" in n for n in tensor_names)
    has_lm_head = any("lm_head" in n or "output" in n for n in tensor_names)

    if not has_embed:
        warnings.append("Не найден тензор эмбеддингов (embed_tokens/wte).")
    if not has_lm_head and not config.get("tie_word_embeddings", False):
        warnings.append("Не найден lm_head (выходной слой).")

    layer_indices = set()
    for n in tensor_names:
        m = re.search(r"layers?\.(\d+)\.", n) or re.search(r"h\.(\d+)\.", n)
        if m:
            layer_indices.add(int(m.group(1)))
    if layer_indices:
        found_layers = len(layer_indices)
        if found_layers != config["num_layers"]:
            errors.append(
                f"В весах {found_layers} слоёв, а в конфиге {config['num_layers']}."
            )

    for name, info in safetensors_info.get("tensors", {}).items():
        if "embed_tokens" in name and len(info["shape"]) == 2:
            vocab_dim = info["shape"][0]
            if vocab_dim != config["vocab_size"]:
                errors.append(
                    f"{name}: vocab_size в весах = {vocab_dim}, "
                    f"в конфиге = {config['vocab_size']}."
                )
        if "lm_head" in name and len(info["shape"]) == 2:
            vocab_dim = info["shape"][0]
            if vocab_dim != config["vocab_size"]:
                errors.append(
                    f"{name}: vocab_size в весах = {vocab_dim}, "
                    f"в конфиге = {config['vocab_size']}."
                )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def validate_tokenizer_match_config(config, tokenizer_info):
    """Сверяет vocab_size в конфиге и токенизаторе."""
    errors = []
    warnings = []

    if not tokenizer_info:
        return {"valid": True, "errors": [], "warnings": []}

    tok_vocab = tokenizer_info.get("vocab_size", 0)
    cfg_vocab = config.get("vocab_size", 0)

    if tok_vocab == 0:
        warnings.append("Токенизатор без vocab_size — не могу сверить.")
        return {"valid": True, "errors": [], "warnings": warnings}

    if cfg_vocab == 0:
        warnings.append("config.vocab_size = 0, сверка невозможна.")
        return {"valid": True, "errors": [], "warnings": warnings}

    if tok_vocab != cfg_vocab:
        if abs(tok_vocab - cfg_vocab) <= 100:
            warnings.append(
                f"Небольшое расхождение vocab_size: токенизатор = {tok_vocab}, "
                f"конфиг = {cfg_vocab}."
            )
        else:
            errors.append(
                f"vocab_size не совпадает: токенизатор = {tok_vocab}, "
                f"конфиг = {cfg_vocab}."
            )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# ============================================================
# 3. ОЦЕНКА ПАМЯТИ И СКОРОСТИ
# ============================================================
DTYPE_BYTES = {
    "float32": 4, "float16": 2, "bfloat16": 2,
    "float64": 8, "int8": 1, "uint8": 1,
    "fp8": 1, "int4": 0.5, "q4": 0.5,
}


def estimate_parameters(config):
    """Оценивает количество параметров модели (для трансформера).

    Для MLA (DeepSeek V2/V3): точные формулы по реальным весам
    из genof11mo.mla.MultiHeadLatentAttention.
    """
    h = config["hidden_size"]
    L = config["num_layers"]
    V = config["vocab_size"]
    I = config["intermediate_size"]
    n_heads = config["num_heads"]
    n_kv_heads = config["num_kv_heads"]
    head_dim = config.get("head_dim") or (h // max(n_heads, 1))

    params_embed = V * h
    params_lm_head = 0 if config.get("tie_word_embeddings") else V * h

    if config["attention_type"] == "mla":
        kv_lora = config.get("kv_lora_rank", 512)
        qk_rope = config.get("qk_rope_head_dim", 64)
        qk_nope = config.get("qk_nope_head_dim", 128)
        v_head_dim = config.get("v_head_dim", qk_nope)

        # w_q: (n_heads * (qk_nope + qk_rope), h)
        p_q = h * (n_heads * (qk_nope + qk_rope))
        # w_kv_a: (kv_lora + qk_rope, h) — сжатие в c_kv и k_rope
        p_kv_a = h * (kv_lora + qk_rope)
        # kv_a_layernorm: (kv_lora,)
        p_norm = kv_lora
        # w_kv_b: (n_heads * (qk_nope + v_head_dim), kv_lora)
        p_kv_b = kv_lora * (n_heads * (qk_nope + v_head_dim))
        # w_o: (h, n_heads * v_head_dim)
        p_o = h * (n_heads * v_head_dim)

        params_attn = p_q + p_kv_a + p_norm + p_kv_b + p_o
    else:
        q = h * n_heads * head_dim
        k = h * n_kv_heads * head_dim
        v = h * n_kv_heads * head_dim
        o = n_heads * head_dim * h
        params_attn = q + k + v + o

    if config["is_moe"]:
        params_ffn = config["num_experts"] * (3 * h * I)
        if config.get("num_shared_experts", 0) > 0:
            params_ffn += config["num_shared_experts"] * (3 * h * I)
    else:
        params_ffn = 3 * h * I

    params_norm = 2 * h

    params_per_layer = params_attn + params_ffn + params_norm
    params_total = params_embed + params_lm_head + L * params_per_layer

    return {
        "total": params_total,
        "embedding": params_embed,
        "lm_head": params_lm_head,
        "per_layer": params_per_layer,
        "attention": L * params_attn,
        "ffn": L * params_ffn,
        "human": _human_params(params_total),
    }


def _human_params(n):
    if n >= 1e12:
        return f"{n/1e12:.2f}T"
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def estimate_memory_profile(config, dtype_str="float16", context_len=2048, batch_size=1):
    """Оценивает память: веса + KV-cache + активации."""
    params = estimate_parameters(config)
    total_params = params["total"]

    dtype_bytes = DTYPE_BYTES.get(dtype_str.lower(), 2)
    weights_bytes = total_params * dtype_bytes

    h = config["hidden_size"]
    L = config["num_layers"]
    n_kv_heads = config["num_kv_heads"]
    head_dim = config.get("head_dim") or (h // max(config["num_heads"], 1))

    kv_cache_bytes = (
        2 * L * n_kv_heads * head_dim * context_len * batch_size * dtype_bytes
    )

    activations_bytes = (
        batch_size * context_len * h * 2 * 4 * dtype_bytes
    )

    total_bytes = weights_bytes + kv_cache_bytes + activations_bytes

    return {
        "dtype": dtype_str,
        "context_len": context_len,
        "batch_size": batch_size,
        "params_total": params["human"],
        "weights_gb": round(weights_bytes / (1024**3), 2),
        "kv_cache_gb": round(kv_cache_bytes / (1024**3), 2),
        "activations_gb": round(activations_bytes / (1024**3), 2),
        "total_gb": round(total_bytes / (1024**3), 2),
        "total_mb": round(total_bytes / (1024**2), 1),
    }


def recommend_quantization(config, available_vram_gb=None):
    """Рекомендует квантование под доступную VRAM."""
    recommendations = []

    base = estimate_memory_profile(config, "float16", context_len=2048)
    fp16_gb = base["total_gb"]

    variants = [
        ("FP16", fp16_gb, "максимальная точность"),
        ("INT8", fp16_gb * 0.5, "хороший баланс"),
        ("INT4", fp16_gb * 0.25, "экономия памяти"),
        ("Q4_K_M (GGUF)", fp16_gb * 0.28, "стандарт для CPU"),
        ("Q5_K_M (GGUF)", fp16_gb * 0.33, "компромисс"),
        ("Q8_0 (GGUF)", fp16_gb * 0.5, "почти без потерь"),
    ]

    if available_vram_gb is None:
        for name, gb, note in variants:
            recommendations.append({
                "quant": name,
                "estimated_gb": round(gb, 2),
                "note": note,
            })
    else:
        for name, gb, note in variants:
            if gb <= available_vram_gb * 0.9:
                recommendations.append({
                    "quant": name,
                    "estimated_gb": round(gb, 2),
                    "fits": True,
                    "note": note,
                })
            else:
                recommendations.append({
                    "quant": name,
                    "estimated_gb": round(gb, 2),
                    "fits": False,
                    "note": note,
                })

    return recommendations


# ============================================================
# 4. ИЗВЛЕЧЕНИЕ МЕТАДАННЫХ ИЗ README
# ============================================================
def extract_metadata_from_readme(folder_path):
    """Парсит README.md (model card) и вытаскивает метаданные."""
    folder = Path(folder_path)
    if not folder.exists():
        return {}

    readme = None
    for candidate in ("README.md", "README.rst", "readme.md", "MODEL_CARD.md"):
        p = folder / candidate
        if p.exists():
            readme = p
            break

    if not readme:
        return {}

    try:
        text = readme.read_text("utf-8", errors="replace")
    except Exception:
        return {}

    metadata = {
        "readme_file": readme.name,
        "readme_size_bytes": readme.stat().st_size,
    }

    if text.startswith("---"):
        end = text.find("---", 3)
        if end > 0:
            frontmatter = text[3:end]
            for line in frontmatter.split("\n"):
                if ":" in line:
                    k, _, v = line.partition(":")
                    metadata[k.strip()] = v.strip()

    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if title_match:
        metadata["title"] = title_match.group(1).strip()

    lang_pattern = re.findall(r"\[([^\]]+)\]|language[:\s]+(\w+)", text.lower())
    metadata["has_language_tags"] = bool(lang_pattern)

    lic_match = re.search(r"license[:\s]+([\w\-\.]+)", text, re.IGNORECASE)
    if lic_match:
        metadata["license"] = lic_match.group(1)

    return metadata


# ============================================================
# 5. АГРЕГИРУЮЩИЙ АНАЛИЗ
# ============================================================
def analyze_model(config, tokenizer_info=None, safetensors_info=None, gguf_info=None):
    """Агрегирует весь анализ в единый отчёт."""
    result = {
        "arch_class": "UnknownForCausalLM",
        "validation": {"valid": True, "errors": [], "warnings": []},
        "tensor_validation": {"valid": True, "errors": [], "warnings": []},
        "tokenizer_validation": {"valid": True, "errors": [], "warnings": []},
        "parameter_estimate": None,
        "memory_profile": {},
        "quantization_recommendations": [],
    }

    if not config or "warning" in config:
        return result

    try:
        result["arch_class"] = detect_architecture_class(config)
    except Exception as e:
        result["validation"]["errors"].append(f"Ошибка определения архитектуры: {e}")

    try:
        result["validation"] = validate_architecture(config)
    except ArchitectureError as e:
        result["validation"]["valid"] = False
        result["validation"]["errors"].append(str(e))

    if safetensors_info:
        try:
            result["tensor_validation"] = validate_safetensors_match_config(
                config, safetensors_info
            )
        except Exception as e:
            result["tensor_validation"]["valid"] = False
            result["tensor_validation"]["errors"].append(str(e))

    if tokenizer_info:
        try:
            result["tokenizer_validation"] = validate_tokenizer_match_config(
                config, tokenizer_info
            )
        except Exception as e:
            result["tokenizer_validation"]["valid"] = False
            result["tokenizer_validation"]["errors"].append(str(e))

    if not config.get("is_ssm"):
        try:
            result["parameter_estimate"] = estimate_parameters(config)
        except Exception as e:
            result["parameter_estimate"] = {"error": str(e)}

    try:
        dtype_str = config.get("torch_dtype", "float16")
        result["memory_profile"] = {
            "fp16": estimate_memory_profile(config, "float16"),
            "fp32": estimate_memory_profile(config, "float32"),
            "int8": estimate_memory_profile(config, "int8"),
            "int4": estimate_memory_profile(config, "int4"),
        }
    except ProfilerError as e:
        result["memory_profile"] = {"error": str(e)}
    except Exception as e:
        result["memory_profile"] = {"error": str(e)}

    try:
        result["quantization_recommendations"] = recommend_quantization(config)
    except Exception:
        result["quantization_recommendations"] = []

    return result


__all__ = [
    "detect_architecture_class",
    "validate_architecture",
    "validate_safetensors_match_config",
    "validate_tokenizer_match_config",
    "estimate_parameters",
    "estimate_memory_profile",
    "recommend_quantization",
    "extract_metadata_from_readme",
    "analyze_model",
    "ARCH_CLASS_MAP",
]