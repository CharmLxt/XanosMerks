"""Инспектор модели: собирает полный паспорт модели из всех парсеров и анализов."""
import json
from pathlib import Path

from .readers import (
    parse_config, parse_tokenizer, parse_safetensors, parse_gguf,
)
from .analysis import (
    analyze_model, extract_metadata_from_readme,
)
from .support_tools.sup2_spr.cache import ModelCache
from .support_tools.sup2_spr.path_utils import (
    find_config, find_tokenizer, find_shards, human_size,
)
from .support_tools.core_spr.exceptions import (
    WeightNotFoundError, ConfigNotFoundError, GamtypeError,
)


# Глобальный кэш (один на процесс)
_cache = ModelCache()


def inspect_model(folder_path, use_cache=True):
    """
    Полный осмотр модели. Возвращает паспорт модели.
    Кэширует результат на диск и в памяти.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Папка не существует: {folder_path}")

    # ============================================================
    # 0. Проверяем кэш
    # ============================================================
    if use_cache:
        cached = _cache.get(str(folder))
        if cached is not None:
            return cached

    # ============================================================
    # 1. Ищем файлы
    # ============================================================
    config_path = find_config(folder)
    tokenizer_path = find_tokenizer(folder)
    gguf_files = list(folder.rglob("*.gguf"))
    st_files = list(folder.rglob("*.safetensors"))

    if not gguf_files and not st_files:
        raise WeightNotFoundError(f"Не найдено ни GGUF, ни safetensors в {folder}")

    # ============================================================
    # 2. Парсим config
    # ============================================================
    config = {}
    config_error = None
    if config_path:
        try:
            config = parse_config(config_path)
        except GamtypeError as e:
            config_error = str(e)
            config = {"model_type": "unknown", "warning": config_error}

    # ============================================================
    # 3. Парсим токенизатор
    # ============================================================
    tokenizer_info = None
    if tokenizer_path:
        try:
            tokenizer_info = parse_tokenizer(tokenizer_path)
        except GamtypeError:
            tokenizer_info = None

    # ============================================================
    # 4. Парсим веса
    # ============================================================
    weights_info = None
    weights_format = None
    weights_error = None

    if gguf_files:
        # GGUF приоритетнее — обычно standalone
        try:
            weights_info = parse_gguf(gguf_files[0])
            weights_format = "gguf"
        except GamtypeError as e:
            weights_error = str(e)

    if st_files and not weights_info:
        try:
            weights_info = parse_safetensors(folder)
            weights_format = "safetensors"
        except GamtypeError as e:
            weights_error = str(e)

    # ============================================================
    # 5. Анализ
    # ============================================================
    analysis = analyze_model(
        config=config,
        tokenizer_info=tokenizer_info,
        safetensors_info=weights_info if weights_format == "safetensors" else None,
        gguf_info=weights_info if weights_format == "gguf" else None,
    )

    # ============================================================
    # 6. Метаданные из README
    # ============================================================
    readme_metadata = extract_metadata_from_readme(folder)

    # ============================================================
    # 7. Собираем паспорт
    # ============================================================
    # Очищаем config от несериализуемых полей
    clean_config = None
    if config and "warning" not in config:
        clean_config = {k: v for k, v in config.items()
                       if k not in ("raw_config", "numpy_dtype_class")}

    passport = {
        "path": str(folder),
        "model_type": config.get("model_type", "unknown"),
        "arch_class": analysis.get("arch_class", "UnknownForCausalLM"),
        "weights_format": weights_format,
        "config": clean_config,
        "config_error": config_error,
        "tokenizer": {
            "type": tokenizer_info.get("type") if tokenizer_info else None,
            "vocab_size": tokenizer_info.get("vocab_size", 0) if tokenizer_info else 0,
            "bos_token": tokenizer_info.get("bos_token") if tokenizer_info else None,
            "eos_token": tokenizer_info.get("eos_token") if tokenizer_info else None,
            "chat_template": bool(tokenizer_info.get("chat_template")) if tokenizer_info else False,
        } if tokenizer_info else None,
        "weights": {
            "format": weights_format,
            "num_files": weights_info.get("num_shards", 1) if weights_info and weights_format == "safetensors"
                         else (1 if weights_format == "gguf" else 0),
            "is_sharded": weights_info.get("is_sharded", False) if weights_info else False,
            "num_tensors": (weights_info.get("total_tensors", 0)
                            if weights_format == "safetensors"
                            else weights_info.get("tensor_count", 0) if weights_info else 0),
            "total_size_bytes": weights_info.get("total_size_bytes", 0) if weights_info else 0,
            "total_size_human": weights_info.get("total_size_human", "0 B") if weights_info else "0 B",
        },
        "gguf_meta": {
            "architecture": weights_info.get("architecture") if weights_info and weights_format == "gguf" else None,
            "model_name": weights_info.get("model_name") if weights_info and weights_format == "gguf" else None,
            "version": weights_info.get("version") if weights_info and weights_format == "gguf" else None,
            "num_metadata_keys": len(weights_info.get("metadata", {})) if weights_info and weights_format == "gguf" else 0,
        } if weights_format == "gguf" else None,
        "validation": analysis.get("validation", {}),
        "tensor_validation": analysis.get("tensor_validation", {}),
        "tokenizer_validation": analysis.get("tokenizer_validation", {}),
        "parameter_estimate": analysis.get("parameter_estimate"),
        "memory_profile": analysis.get("memory_profile", {}),
        "quantization_recommendations": analysis.get("quantization_recommendations", []),
        "readme_metadata": readme_metadata,
        "weights_error": weights_error,
    }

    # ============================================================
    # 8. Кэшируем
    # ============================================================
    if use_cache:
        try:
            _cache.set(str(folder), passport)
        except Exception:
            pass  # кэш не критичен

    return passport


def invalidate_cache(folder_path=None):
    """Сбрасывает кэш для папки или полностью."""
    if folder_path:
        _cache.invalidate(str(folder_path))
    else:
        _cache.memory.clear()
        _cache.disk.clear()


__all__ = ["inspect_model", "invalidate_cache"]