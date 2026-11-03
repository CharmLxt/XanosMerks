"""Парсеры файлов моделей. Полная реализация: config, tokenizer, safetensors, gguf."""
import base64
import json
import struct
import re
from pathlib import Path

from .support_tools.core_spr.exceptions import (
    InvalidConfigError, MissingRequiredFieldError, UnknownArchitectureError,
    TokenizerNotFoundError, InvalidTokenizerError, TokenizerFormatError,
    CorruptedSafetensorsError, CorruptedGGUFError, ShardMismatchError,
    ShardIncompleteError, WeightFormatError,
)
from .support_tools.core_spr.dtype_map import (
    get_numpy_dtype, get_dtype_size, gguf_type_name,
    gguf_tensor_size_in_bytes, SAFETENSORS_TO_NUMPY, GGUF_TENSOR_TYPES,
)
from .support_tools.core_spr.byte_utils import (
    ByteReader, file_size, is_safetensors, is_gguf,
)
from .support_tools.sup2_spr.path_utils import (
    find_shards, is_shard_complete, total_size_bytes, human_size, SHARD_PATTERN,
)


SUPPORTED_FAMILIES = {
    "llama", "mistral", "mixtral", "qwen", "qwen2", "qwen2_moe",
    "gemma", "gemma2", "phi", "phi3", "falcon", "mpt", "gpt2", "gpt_neox", "gptj",
    "command-r", "cohere", "yi", "zephyr", "stablelm", "starcoder2",
    "deepseek_v1", "deepseek_v2", "deepseek_v3", "deepseek_r1",
    "mamba", "rwkv", "rwkv4", "rwkv5", "rwkv6",
    "llava", "qwen2_vl", "phi3_v", "idefics3",
}


def _normalize_model_type(raw):
    if not raw:
        return "unknown"
    return raw.lower().replace("-", "_").replace(".", "_")


# ============================================================
# ПАРСЕР 1: CONFIG.JSON
# ============================================================
def parse_config(config_path):
    """Читает config.json и возвращает нормализованный словарь."""
    path = Path(config_path)
    try:
        raw = json.loads(path.read_text("utf-8"))
    except FileNotFoundError:
        raise InvalidConfigError(f"Файл не найден: {config_path}")
    except json.JSONDecodeError as e:
        raise InvalidConfigError(f"Некорректный JSON в {path.name}: {e}")

    def _get(*keys, default=None, cast=None):
        for k in keys:
            if k in raw and raw[k] is not None:
                val = raw[k]
                if cast is not None:
                    try:
                        return cast(val)
                    except (ValueError, TypeError):
                        continue
                return val
        return default

    raw_type = _get("model_type", "architecture", default="unknown")
    model_type = _normalize_model_type(raw_type)

    config = {
        "raw_config": raw,
        "model_type": model_type,
        "architectures": raw.get("architectures", []),
    }

    config["num_layers"] = _get(
        "num_hidden_layers", "n_layer", "num_layers", "n_layers", "depth",
        default=0, cast=int,
    )
    config["hidden_size"] = _get(
        "hidden_size", "n_embd", "d_model", "d_model_size", "model_dim",
        default=0, cast=int,
    )
    config["num_heads"] = _get(
        "num_attention_heads", "n_head", "num_heads", "n_heads",
        default=0, cast=int,
    )
    config["num_kv_heads"] = _get(
        "num_key_value_heads", "n_head_kv", "num_kv_heads", "num_kv_attention_heads",
        default=None, cast=int,
    )
    config["vocab_size"] = _get(
        "vocab_size", "n_vocab", "vocab", default=0, cast=int,
    )
    config["intermediate_size"] = _get(
        "intermediate_size", "n_inner", "ffn_dim", "ffn_hidden_size", "expert_hidden_size",
        default=0, cast=int,
    )
    config["max_position_embeddings"] = _get(
        "max_position_embeddings", "n_positions", "max_seq_len", "max_seq_length", "n_ctx",
        default=2048, cast=int,
    )

    config["hidden_act"] = _get(
        "hidden_act", "activation_function", "activation", default="silu",
    )
    config["norm_type"] = "rmsnorm" if ("rms_norm_eps" in raw or "rmsnorm" in str(raw.get("architectures", [])).lower()) else "layernorm"
    config["norm_eps"] = _get(
        "rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon", "norm_eps",
        default=1e-5, cast=float,
    )

    config["rope_theta"] = _get(
        "rope_theta", "rotary_emb_base", "rotary_theta",
        default=10000.0, cast=float,
    )
    config["rope_scaling"] = None
    if "rope_scaling" in raw and raw["rope_scaling"]:
        rs = raw["rope_scaling"]
        config["rope_scaling"] = {
            "type": rs.get("type", rs.get("rope_type", "linear")),
            "factor": rs.get("factor", 1.0),
            "original_max_position": rs.get("original_max_position_embeddings", config["max_position_embeddings"]),
        }

    raw_dtype = _get("torch_dtype", "dtype", "torch_dtype_str", default="float16")
    if isinstance(raw_dtype, str) and raw_dtype.startswith("torch."):
        raw_dtype = raw_dtype[6:]
    config["torch_dtype"] = raw_dtype
    try:
        np_dtype = get_numpy_dtype(raw_dtype)
        config["numpy_dtype"] = np_dtype.__name__
        config["numpy_dtype_class"] = np_dtype
    except ValueError:
        config["numpy_dtype"] = "float16"
        config["numpy_dtype_class"] = None

    if config["num_kv_heads"] is None:
        config["num_kv_heads"] = config["num_heads"]
        config["attention_type"] = "mha"
    elif config["num_kv_heads"] == 1:
        config["attention_type"] = "mqa"
    elif config["num_kv_heads"] < config["num_heads"]:
        config["attention_type"] = "gqa"
    else:
        config["attention_type"] = "mha"

    if "kv_lora_rank" in raw:
        config["attention_type"] = "mla"
        config["kv_lora_rank"] = raw.get("kv_lora_rank")
        config["qk_rope_head_dim"] = raw.get("qk_rope_head_dim")
        config["qk_nope_head_dim"] = raw.get("qk_nope_head_dim")

    config["is_moe"] = False
    config["num_experts"] = _get(
        "num_local_experts", "num_experts", "n_routed_experts", default=0, cast=int,
    )
    config["num_experts_per_tok"] = _get(
        "num_experts_per_tok", "n_experts_per_tok", "moe_topk", default=0, cast=int,
    )
    config["num_shared_experts"] = _get("n_shared_experts", "num_shared_experts", default=0, cast=int)
    if config["num_experts"] > 0:
        config["is_moe"] = True

    config["is_ssm"] = model_type in {"mamba", "mamba2", "rwkv", "rwkv4", "rwkv5", "rwkv6"}
    config["ssm_state_dim"] = _get("d_state", "ssm_state_dim", default=None, cast=int)
    config["ssm_conv_dim"] = _get("d_conv", "ssm_conv_dim", default=None, cast=int)
    config["ssm_expand"] = _get("expand", "ssm_expand", default=None, cast=int)

    if model_type.startswith("rwkv"):
        config["is_ssm"] = True
        config["wkv_head_size"] = _get("head_size", "wkv_head_size", default=0, cast=int)

    config["is_vision"] = False
    if "vision_config" in raw or "vision_tower" in raw:
        config["is_vision"] = True
        vc = raw.get("vision_config", {})
        config["vision_hidden_size"] = vc.get("hidden_size") or raw.get("vision_hidden_size")
        config["vision_num_layers"] = vc.get("num_hidden_layers") or raw.get("vision_num_layers")
        config["image_token_id"] = raw.get("image_token_id") or raw.get("image_token_index")

    config["tie_word_embeddings"] = _get("tie_word_embeddings", default=False, cast=bool)
    config["use_cache"] = _get("use_cache", default=True, cast=bool)
    config["sliding_window"] = _get("sliding_window", "window_size", default=None, cast=int)
    config["attention_bias"] = _get("attention_bias", default=False, cast=bool)
    config["mlp_bias"] = _get("mlp_bias", default=False, cast=bool)

    if config["num_layers"] <= 0:
        raise MissingRequiredFieldError(f"num_layers = {config['num_layers']} в {path}")
    if config["hidden_size"] <= 0:
        raise MissingRequiredFieldError(f"hidden_size = {config['hidden_size']} в {path}")
    if not config["is_ssm"] and config["num_heads"] <= 0:
        raise MissingRequiredFieldError(f"num_heads = {config['num_heads']} для трансформера в {path}")

    if config["num_heads"] > 0:
        config["head_dim"] = _get("head_dim", default=config["hidden_size"] // config["num_heads"], cast=int)
    else:
        config["head_dim"] = 0

    return config


# ============================================================
# ПАРСЕР 2: TOKENIZER
# ============================================================
def parse_tokenizer(tokenizer_path):
    """Читает токенизатор любого формата: BPE, WordPiece, SentencePiece, Tiktoken."""
    path = Path(tokenizer_path)
    if not path.exists():
        raise TokenizerNotFoundError(f"Токенизатор не найден: {tokenizer_path}")

    suffix = path.suffix.lower()

    if suffix == ".json":
        return _parse_tokenizer_json(path)
    if suffix == ".model" or path.name == "spiece.model":
        return _parse_sentencepiece(path)
    if suffix == ".tiktoken":
        return _parse_tiktoken(path)

    try:
        return _parse_tokenizer_json(path)
    except Exception:
        raise TokenizerFormatError(f"Неизвестный формат токенизатора: {path.name}")


def _parse_tokenizer_json(path):
    """Разбор HuggingFace tokenizer.json (BPE, WordPiece, Unigram)."""
    try:
        data = json.loads(path.read_text("utf-8"))
    except Exception as e:
        raise InvalidTokenizerError(f"Не удалось прочитать {path.name}: {e}")

    model = data.get("model", {})
    model_type = model.get("type", "BPE").upper()

    vocab = {}
    merges = []
    added_tokens = []

    if model_type == "BPE":
        raw_vocab = model.get("vocab", {})
        if isinstance(raw_vocab, dict):
            vocab = raw_vocab
        elif isinstance(raw_vocab, list):
            for item in raw_vocab:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    vocab[item[0]] = item[1]

        raw_merges = model.get("merges", [])
        if isinstance(raw_merges, list):
            for m in raw_merges:
                if isinstance(m, str):
                    merges.append(m.split())
                elif isinstance(m, (list, tuple)):
                    merges.append(list(m))

    elif model_type == "WORDPIECE":
        raw_vocab = model.get("vocab", {})
        if isinstance(raw_vocab, dict):
            vocab = raw_vocab
        elif isinstance(raw_vocab, list):
            for i, tok in enumerate(raw_vocab):
                vocab[tok] = i

    elif model_type == "UNIGRAM":
        raw_vocab = model.get("vocab", [])
        if isinstance(raw_vocab, list):
            for i, item in enumerate(raw_vocab):
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    vocab[item[0]] = i
                elif isinstance(item, str):
                    vocab[item] = i
        elif isinstance(raw_vocab, dict):
            vocab = raw_vocab

    for at in data.get("added_tokens", []):
        added_tokens.append({
            "id": at.get("id"),
            "content": at.get("content"),
            "special": at.get("special", False),
        })

    if not vocab:
        raise InvalidTokenizerError(f"В {path.name} не найден словарь")

    return {
        "type": model_type.lower(),
        "vocab": vocab,
        "vocab_size": len(vocab),
        "merges": merges,
        "merges_count": len(merges),
        "added_tokens": added_tokens,
        "bos_token": data.get("bos_token", "<s>"),
        "eos_token": data.get("eos_token", "</s>"),
        "pad_token": data.get("pad_token"),
        "unk_token": data.get("unk_token", "<unk>"),
        "add_bos_token": data.get("add_bos_token", True),
        "add_eos_token": data.get("add_eos_token", False),
        "clean_up_tokenization_spaces": data.get("clean_up_tokenization_spaces", True),
        "tokenizer_class": data.get("tokenizer_class", "PreTrainedTokenizer"),
        "chat_template": data.get("chat_template"),
    }


# ============================================================
# SENTENCEPIECE: реальный protobuf-парсер
# ============================================================
# SentencePiece .model — protobuf с известной схемой:
#   ModelProto:
#     field 1: pieces (repeated SentencePiece)
#     field 2: trainer_spec (TrainerSpec)
#     field 3: normalizer_spec
#     field 4: self_test_data
#     field 5: denormalizer_spec
#   SentencePiece:
#     field 1: piece (string)
#     field 2: score (float32, fixed32)
#     field 3: type (enum, varint) — 0 NORMAL, 1 UNKNOWN, 2 CONTROL,
#                                    3 USER_DEFINED, 4 UNUSED, 5 BYTE
#   TrainerSpec:
#     field 4: vocab_size (int32)
#     field 40: unk_id (int32)
#     field 41: bos_id (int32)
#     field 42: eos_id (int32)
#     field 43: pad_id (int32)

_PIECE_TYPES = {0: "NORMAL", 1: "UNKNOWN", 2: "CONTROL", 3: "USER_DEFINED",
                4: "UNUSED", 5: "BYTE"}


def _read_varint(buf, pos):
    """Читает varint из buf[pos:]. Возвращает (value, new_pos)."""
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise InvalidTokenizerError("Unexpected EOF в varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
        if shift > 63:
            raise InvalidTokenizerError("Varint слишком длинный")
    return result, pos


def _skip_protobuf_field(buf, pos, wire_type):
    """Пропускает поле, возвращает новый pos."""
    if wire_type == 0:  # varint
        _, pos = _read_varint(buf, pos)
    elif wire_type == 1:  # 64-bit
        pos += 8
    elif wire_type == 2:  # length-delimited
        length, pos = _read_varint(buf, pos)
        pos += length
    elif wire_type == 5:  # 32-bit
        pos += 4
    else:
        raise InvalidTokenizerError(f"Неизвестный wire_type: {wire_type}")
    return pos


def _parse_sentencepiece_piece(buf):
    """Парсит одно SentencePiece сообщение (buf = bytes одного piece)."""
    piece = None
    score = 0.0
    piece_type = 0
    pos = 0
    while pos < len(buf):
        tag, pos = _read_varint(buf, pos)
        field_num = tag >> 3
        wire_type = tag & 0x07

        if field_num == 1 and wire_type == 2:
            length, pos = _read_varint(buf, pos)
            piece = buf[pos:pos + length].decode("utf-8", errors="replace")
            pos += length
        elif field_num == 2 and wire_type == 5:
            score = struct.unpack("<f", buf[pos:pos + 4])[0]
            pos += 4
        elif field_num == 3 and wire_type == 0:
            piece_type, pos = _read_varint(buf, pos)
        else:
            pos = _skip_protobuf_field(buf, pos, wire_type)

    return piece, score, piece_type


def _parse_sentencepiece(path):
    """Разбор .model (SentencePiece protobuf) — настоящий protobuf."""
    try:
        raw = path.read_bytes()
    except Exception as e:
        raise InvalidTokenizerError(f"Не удалось прочитать {path.name}: {e}")

    pieces = []
    trainer_spec_bytes = None
    pos = 0

    try:
        while pos < len(raw):
            tag, pos = _read_varint(raw, pos)
            field_num = tag >> 3
            wire_type = tag & 0x07

            if field_num == 1 and wire_type == 2:
                length, pos = _read_varint(raw, pos)
                piece_buf = raw[pos:pos + length]
                pos += length
                piece, score, piece_type = _parse_sentencepiece_piece(piece_buf)
                if piece is not None:
                    pieces.append((piece, score, piece_type))
            elif field_num == 2 and wire_type == 2:
                length, pos = _read_varint(raw, pos)
                trainer_spec_bytes = raw[pos:pos + length]
                pos += length
            else:
                pos = _skip_protobuf_field(raw, pos, wire_type)
    except InvalidTokenizerError:
        raise
    except Exception as e:
        raise InvalidTokenizerError(f"Ошибка парсинга protobuf {path.name}: {e}")

    if not pieces:
        raise InvalidTokenizerError(f"В {path.name} нет pieces (не sentencepiece?)")

    vocab = {}
    piece_types = {}
    added_tokens = []
    unk_id = None
    bos_id = None
    eos_id = None
    pad_id = None
    unk_piece = None
    bos_piece = None
    eos_piece = None
    pad_piece = None

    for idx, (piece, _score, ptype) in enumerate(pieces):
        vocab[piece] = idx
        piece_types[piece] = _PIECE_TYPES.get(ptype, "UNKNOWN")
        if ptype == 1:
            unk_id = idx
            unk_piece = piece
        elif ptype == 2:
            if piece == "<s>":
                bos_id, bos_piece = idx, piece
            elif piece == "</s>":
                eos_id, eos_piece = idx, piece
            elif piece == "<pad>":
                pad_id, pad_piece = idx, piece
            else:
                added_tokens.append({"content": piece, "special": True, "id": idx})
        elif ptype == 3:
            added_tokens.append({"content": piece, "special": True, "id": idx})

    if trainer_spec_bytes:
        try:
            tpos = 0
            while tpos < len(trainer_spec_bytes):
                ttag, tpos = _read_varint(trainer_spec_bytes, tpos)
                tf = ttag >> 3
                tw = ttag & 0x07
                if tw == 0:
                    val, tpos = _read_varint(trainer_spec_bytes, tpos)
                    if tf == 40 and unk_id is None:
                        unk_id = val
                    elif tf == 41 and bos_id is None:
                        bos_id = val
                    elif tf == 42 and eos_id is None:
                        eos_id = val
                    elif tf == 43 and pad_id is None:
                        pad_id = val
                else:
                    tpos = _skip_protobuf_field(trainer_spec_bytes, tpos, tw)
        except Exception:
            pass

    return {
        "type": "sentencepiece",
        "vocab": vocab,
        "vocab_size": len(pieces),
        "merges": [],
        "merges_count": 0,
        "added_tokens": added_tokens,
        "bos_token": bos_piece if bos_piece else (vocab and list(vocab.keys())[bos_id] if bos_id is not None else "<s>"),
        "eos_token": eos_piece if eos_piece else (vocab and list(vocab.keys())[eos_id] if eos_id is not None else "</s>"),
        "pad_token": pad_piece,
        "unk_token": unk_piece if unk_piece else "<unk>",
        "bos_id": bos_id,
        "eos_id": eos_id,
        "pad_id": pad_id,
        "unk_id": unk_id,
        "add_bos_token": True,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": True,
        "tokenizer_class": "LlamaTokenizer",
        "chat_template": None,
        "piece_types": piece_types,
    }


# ============================================================
# TIKTOKEN: реальный формат — построчные base64 + rank
# ============================================================
# Реальный .tiktoken — не бинарный. Это текстовый файл:
#   <base64(token_bytes)> <rank>\n
#   <base64(token_bytes)> <rank>\n
#   ...
# Vocab = {token_str: rank}, vocab_size = max(rank) + 1.
# Special tokens (например <|endoftext|>) — те, что после base64-decode
# начинаются с "<|" и заканчиваются "|>".

def _parse_tiktoken(path):
    """Разбор .tiktoken (base64 построчно)."""
    try:
        text = path.read_text("utf-8")
    except Exception as e:
        raise InvalidTokenizerError(f"Не удалось прочитать {path.name}: {e}")

    vocab = {}
    specials = []
    max_rank = -1

    for line_num, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        b64_part, rank_str = parts
        try:
            rank = int(rank_str)
        except ValueError:
            continue
        try:
            token_bytes = base64.b64decode(b64_part, validate=True)
        except Exception:
            token_bytes = b64_part.encode("utf-8", errors="replace")

        token_str = token_bytes.decode("utf-8", errors="replace")
        vocab[token_str] = rank
        if rank > max_rank:
            max_rank = rank

        if token_bytes.startswith(b"<|") and token_bytes.endswith(b"|>"):
            specials.append({"content": token_str, "special": True, "id": rank})

    if not vocab:
        raise InvalidTokenizerError(f"{path.name}: не найдено ни одного токена")

    vocab_size = max_rank + 1 if max_rank >= 0 else len(vocab)

    return {
        "type": "tiktoken",
        "vocab": vocab,
        "vocab_size": vocab_size,
        "merges": [],
        "merges_count": 0,
        "added_tokens": specials,
        "bos_token": None,
        "eos_token": "<|endoftext|>" if "<|endoftext|>" in vocab else None,
        "pad_token": None,
        "unk_token": None,
        "add_bos_token": False,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "TiktokenTokenizer",
        "chat_template": None,
        "n_mergeable": len(vocab) - len(specials),
        "n_special": len(specials),
        "pattern": None,
        "warning": "pattern не встроен в .tiktoken — задавайте его явно для encode",
    }


# ============================================================
# ПАРСЕР 3: SAFETENSORS
# ============================================================
def _read_safetensors_header(filepath):
    path = Path(filepath)
    try:
        with open(path, "rb") as f:
            size_bytes = f.read(8)
            if len(size_bytes) != 8:
                raise CorruptedSafetensorsError(f"Файл {path.name} слишком мал")

            header_size = struct.unpack("<Q", size_bytes)[0]
            if header_size > 100 * 1024 * 1024:
                raise CorruptedSafetensorsError(
                    f"Размер заголовка {path.name} слишком велик ({header_size} байт)"
                )

            header_json = f.read(header_size)
            try:
                header = json.loads(header_json.decode("utf-8"))
            except json.JSONDecodeError as e:
                raise CorruptedSafetensorsError(f"Некорректный JSON в {path.name}: {e}")

            return header
    except FileNotFoundError:
        raise CorruptedSafetensorsError(f"Файл не найден: {filepath}")
    except OSError as e:
        raise CorruptedSafetensorsError(f"Ошибка чтения {path.name}: {e}")


def _extract_tensors_from_header(header, filename, shard_index=0):
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        shape = info.get("shape", [])
        dtype_str = info.get("dtype", "unknown")
        size = 1
        for d in shape:
            size *= d
        try:
            size_bytes = size * get_dtype_size(dtype_str)
        except ValueError:
            size_bytes = 0

        tensors[name] = {
            "shape": shape,
            "dtype": dtype_str,
            "size_bytes": size_bytes,
            "file": filename,
            "shard_index": shard_index,
        }
    return tensors


def parse_safetensors(path_or_dir):
    path = Path(path_or_dir)

    if not path.exists():
        raise CorruptedSafetensorsError(f"Путь не существует: {path}")

    if path.is_file():
        header = _read_safetensors_header(path)
        size_bytes_total = file_size(path)
        metadata = header.get("__metadata__", {})
        tensors = _extract_tensors_from_header(header, path.name, shard_index=1)

        return {
            "format": "safetensors",
            "files": [str(path)],
            "is_sharded": False,
            "num_shards": 1,
            "shard_status": "complete",
            "tensors": tensors,
            "total_tensors": len(tensors),
            "total_size_bytes": size_bytes_total,
            "total_size_human": human_size(size_bytes_total),
            "metadata": metadata,
        }

    all_st_files = list(path.rglob("*.safetensors"))
    if not all_st_files:
        raise CorruptedSafetensorsError(f"В {path} не найдено .safetensors файлов")

    sharded_groups = {}
    standalone_files = []

    for f in all_st_files:
        m = SHARD_PATTERN.match(f.name)
        if m:
            base, idx, total = m.group(1), int(m.group(2)), int(m.group(3))
            sharded_groups.setdefault(base, []).append((idx, total, f))
        else:
            standalone_files.append(f)

    tensors = {}
    all_files = []
    metadata = {}
    total_size = 0
    shard_status = "complete"

    for base_name, shard_list in sharded_groups.items():
        shard_list.sort(key=lambda x: x[0])
        if not is_shard_complete(shard_list):
            raise ShardIncompleteError(
                f"Набор шардов '{base_name}' неполный: {len(shard_list)} файлов"
            )
        for idx, total, shard_path in shard_list:
            header = _read_safetensors_header(shard_path)
            total_size += file_size(shard_path)
            all_files.append(str(shard_path))
            if "__metadata__" in header and not metadata:
                metadata = header["__metadata__"]
            shard_tensors = _extract_tensors_from_header(header, shard_path.name, shard_index=idx)
            for name, info in shard_tensors.items():
                if name in tensors:
                    raise ShardMismatchError(f"Тензор '{name}' встречается в нескольких шардах")
                tensors[name] = info

    for f in standalone_files:
        header = _read_safetensors_header(f)
        total_size += file_size(f)
        all_files.append(str(f))
        if "__metadata__" in header and not metadata:
            metadata = header["__metadata__"]
        file_tensors = _extract_tensors_from_header(header, f.name, shard_index=0)
        for name, info in file_tensors.items():
            if name in tensors:
                raise ShardMismatchError(
                    f"Тензор '{name}' встречается в нескольких файлах в {path}"
                )
            tensors[name] = info

    return {
        "format": "safetensors",
        "files": all_files,
        "is_sharded": len(sharded_groups) > 0,
        "num_shards": sum(len(v) for v in sharded_groups.values()) + len(standalone_files),
        "shard_status": shard_status,
        "tensors": tensors,
        "total_tensors": len(tensors),
        "total_size_bytes": total_size,
        "total_size_human": human_size(total_size),
        "metadata": metadata,
    }


# ============================================================
# ПАРСЕР 4: GGUF
# ============================================================
GGUF_META_TYPES = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16",
    4: "uint32", 5: "int32", 6: "float32", 7: "bool",
    8: "string", 9: "array", 10: "uint64", 11: "int64", 12: "float64",
}


def _read_gguf_meta_value(reader, value_type_id):
    type_name = GGUF_META_TYPES.get(value_type_id)
    if type_name is None:
        raise CorruptedGGUFError(f"Неизвестный тип metadata: {value_type_id}")

    if type_name == "uint8":   return reader.u8()
    if type_name == "int8":    return reader.i8()
    if type_name == "uint16":  return reader.u16()
    if type_name == "int16":   return reader.i16()
    if type_name == "uint32":  return reader.u32()
    if type_name == "int32":   return reader.i32()
    if type_name == "float32": return reader.f32()
    if type_name == "bool":    return bool(reader.u8())
    if type_name == "string":  return reader.read_gguf_string()
    if type_name == "uint64":  return reader.u64()
    if type_name == "int64":   return reader.i64()
    if type_name == "float64": return reader.f64()

    if type_name == "array":
        inner_type_id = reader.u32()
        array_len = reader.u64()
        if array_len > 10_000_000:
            raise CorruptedGGUFError(f"Аномальная длина массива: {array_len}")
        values = []
        for _ in range(array_len):
            values.append(_read_gguf_meta_value(reader, inner_type_id))
        return values

    raise CorruptedGGUFError(f"Неподдерживаемый тип metadata: {type_name}")


def parse_gguf(filepath):
    path = Path(filepath)
    if not path.exists():
        raise CorruptedGGUFError(f"Файл не найден: {filepath}")

    with ByteReader(path) as r:
        magic = r.read(4)
        if magic != b"GGUF":
            raise CorruptedGGUFError(f"{path.name} не является GGUF-файлом")

        version = r.u32()
        if version < 1 or version > 5:
            raise CorruptedGGUFError(f"Неподдерживаемая версия GGUF: {version}")

        tensor_count = r.u64()
        metadata_count = r.u64()

        metadata = {}
        for _ in range(metadata_count):
            key = r.read_gguf_string()
            type_id = r.u32()
            try:
                value = _read_gguf_meta_value(r, type_id)
            except Exception as e:
                raise CorruptedGGUFError(
                    f"Ошибка чтения metadata '{key}' ({GGUF_META_TYPES.get(type_id, '?')}): {e}"
                )
            metadata[key] = value

        alignment = metadata.get("general.alignment", 32)

        tensors = []
        for _ in range(tensor_count):
            name = r.read_gguf_string()
            n_dims = r.u32()
            if n_dims > 8:
                raise CorruptedGGUFError(f"Слишком много измерений у тензора '{name}': {n_dims}")

            dims = [r.u64() for _ in range(n_dims)]
            type_id = r.u32()
            offset = r.u64()

            type_name = gguf_type_name(type_id)
            try:
                num_elements = 1
                for d in dims:
                    num_elements *= d
                size_in_bytes = gguf_tensor_size_in_bytes(type_id, num_elements)
            except Exception:
                size_in_bytes = 0

            tensors.append({
                "name": name,
                "shape": dims,
                "type": type_name,
                "type_id": type_id,
                "offset": offset,
                "size_bytes": size_in_bytes,
                "num_elements": num_elements if dims else 0,
            })

        current_pos = r.tell()
        data_start = ((current_pos + alignment - 1) // alignment) * alignment

    file_size_bytes = file_size(path)
    data_size = file_size_bytes - data_start

    architecture = metadata.get("general.architecture", "unknown")
    model_name = metadata.get("general.name", "")

    return {
        "format": "gguf",
        "file": str(path),
        "version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "metadata": metadata,
        "tensors": tensors,
        "architecture": architecture,
        "model_name": model_name,
        "alignment": alignment,
        "data_start": data_start,
        "data_size_bytes": data_size,
        "total_size_bytes": file_size_bytes,
        "total_size_human": human_size(file_size_bytes),
    }


__all__ = [
    "parse_config",
    "parse_tokenizer",
    "parse_safetensors",
    "parse_gguf",
    "SUPPORTED_FAMILIES",
]