"""Утилиты генерации фейковых моделей для тестов."""
import json
import struct
import numpy as np
from pathlib import Path


SIZE_PRESETS = {
    "tiny": {
        "num_layers": 2, "hidden_size": 128, "num_heads": 4,
        "num_kv_heads": 2, "vocab_size": 512, "intermediate_size": 256,
        "max_position_embeddings": 512,
    },
    "small": {
        "num_layers": 4, "hidden_size": 512, "num_heads": 8,
        "num_kv_heads": 2, "vocab_size": 4096, "intermediate_size": 1024,
        "max_position_embeddings": 1024,
    },
    "medium": {
        "num_layers": 8, "hidden_size": 1024, "num_heads": 16,
        "num_kv_heads": 4, "vocab_size": 32000, "intermediate_size": 4096,
        "max_position_embeddings": 2048,
    },
    "large": {
        "num_layers": 32, "hidden_size": 4096, "num_heads": 32,
        "num_kv_heads": 8, "vocab_size": 32000, "intermediate_size": 11008,
        "max_position_embeddings": 8192,
    },
    "moe_tiny": {
        "num_layers": 2, "hidden_size": 128, "num_heads": 4,
        "num_kv_heads": 2, "vocab_size": 512, "intermediate_size": 256,
        "max_position_embeddings": 512,
        "num_local_experts": 4, "num_experts_per_tok": 2,
    },
    "mla_tiny": {
        "num_layers": 2, "hidden_size": 128, "num_heads": 4,
        "num_kv_heads": 4, "vocab_size": 512, "intermediate_size": 256,
        "max_position_embeddings": 512,
        "kv_lora_rank": 64,
        "qk_nope_head_dim": 32,
        "qk_rope_head_dim": 16,
    },
}


def _is_moe_size(size):
    return size.startswith("moe_")


def _is_mla_size(size):
    return size.startswith("mla_")


def create_test_config(path, size="tiny", model_type=None, **overrides):
    if size not in SIZE_PRESETS:
        raise ValueError(f"Неизвестный size '{size}'. Доступно: {list(SIZE_PRESETS)}")
    params = dict(SIZE_PRESETS[size])
    params.update(overrides)

    is_moe = _is_moe_size(size) or (params.get("num_local_experts", 0) > 0)
    is_mla = _is_mla_size(size) or ("kv_lora_rank" in params)

    if model_type is None:
        if is_mla:
            model_type = "deepseek_v3"
        elif is_moe:
            model_type = "mixtral"
        else:
            model_type = "llama"

    params["model_type"] = model_type
    if is_mla:
        params["architectures"] = ["DeepseekV3ForCausalLM"]
    elif is_moe:
        params["architectures"] = ["MixtralForCausalLM"]
    else:
        params["architectures"] = ["LlamaForCausalLM"]

    params["torch_dtype"] = "float16"
    params["rope_theta"] = 10000.0
    params["rms_norm_eps"] = 1e-5
    params["tie_word_embeddings"] = False

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return params


def create_test_tokenizer(path, vocab_size=512):
    """Word-level токенизатор (устаревший, для совместимости)."""
    vocab = {}
    for i in range(vocab_size):
        if i == 0:
            vocab["<unk>"] = 0
        elif i == 1:
            vocab["<s>"] = 1
        elif i == 2:
            vocab["</s>"] = 2
        elif i == 3:
            vocab["<pad>"] = 3
        else:
            vocab[f"token_{i}"] = i

    data = {
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "bos_token": "<s>", "eos_token": "</s>",
        "pad_token": "<pad>", "unk_token": "<unk>",
        "add_bos_token": True, "add_eos_token": False,
        "tokenizer_class": "LlamaTokenizer",
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def _gpt2_byte_encoder():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def create_byte_level_tokenizer(path, vocab_size=512):
    """Byte-level BPE токенизатор (для реальной работы с текстом)."""
    byte_encoder = _gpt2_byte_encoder()

    vocab = {}
    vocab["<unk>"] = 0
    vocab["<s>"] = 1
    vocab["</s>"] = 2
    vocab["<pad>"] = 3

    for b in range(256):
        vocab[byte_encoder[b]] = len(vocab)

    merges = ["t h", "h e", "i n", "e r", "a n", "r e", "o n", "a t", "e n", "n d",
              "o r", "l l", "t h e", "o u", "e s"]
    for merge in merges:
        parts = merge.split()
        if len(parts) == 2:
            merged = parts[0] + parts[1]
            if merged not in vocab:
                vocab[merged] = len(vocab)

    # Добить до vocab_size dummy токенами
    i = 0
    while len(vocab) < vocab_size:
        name = f"<extra_{i}>"
        if name not in vocab:
            vocab[name] = len(vocab)
        i += 1

    data = {
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
        "bos_token": "<s>", "eos_token": "</s>",
        "pad_token": "<pad>", "unk_token": "<unk>",
        "tokenizer_class": "LlamaTokenizer",
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def _write_safetensors(filepath, tensors):
    header = {}
    data_parts = []
    offset = 0
    DTYPE_MAP = {
        np.dtype("float32"): "F32",
        np.dtype("float16"): "F16",
        np.dtype("int32"): "I32",
        np.dtype("int64"): "I64",
        np.dtype("uint8"): "U8",
    }
    for name, arr in tensors.items():
        dtype_str = DTYPE_MAP.get(arr.dtype, "F32")
        raw = arr.tobytes()
        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        data_parts.append(raw)
        offset += len(raw)

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(filepath, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for raw in data_parts:
            f.write(raw)


def create_test_safetensors(folder, params, shards=1, random_weights=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    L = params["num_layers"]
    H = params["hidden_size"]
    I = params["intermediate_size"]
    V = params["vocab_size"]
    NH = params["num_heads"]

    is_moe = params.get("num_local_experts", 0) > 0
    is_mla = "kv_lora_rank" in params
    num_experts = params.get("num_local_experts", 0)

    tensors = {}

    def _rand(shape, dtype=np.float16):
        if random_weights:
            return np.random.randn(*shape).astype(dtype) * 0.02
        return np.zeros(shape, dtype=dtype)

    tensors["model.embed_tokens.weight"] = _rand((V, H))
    tensors["lm_head.weight"] = _rand((V, H))

    for i in range(L):
        prefix = f"model.layers.{i}"
        if is_mla:
            qk_nope = params["qk_nope_head_dim"]
            qk_rope = params["qk_rope_head_dim"]
            v_dim = params.get("v_head_dim", qk_nope)
            lora = params["kv_lora_rank"]
            head_dim = qk_nope + qk_rope
            tensors[f"{prefix}.self_attn.q_proj.weight"] = _rand((NH * head_dim, H))
            tensors[f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"] = _rand((lora + qk_rope, H))
            tensors[f"{prefix}.self_attn.kv_a_layernorm.weight"] = np.ones((lora,), dtype=np.float16)
            tensors[f"{prefix}.self_attn.kv_b_proj.weight"] = _rand((NH * (qk_nope + v_dim), lora))
            tensors[f"{prefix}.self_attn.o_proj.weight"] = _rand((H, NH * v_dim))
        else:
            NKV = params["num_kv_heads"]
            head_dim = H // NH
            tensors[f"{prefix}.self_attn.q_proj.weight"] = _rand((NH * head_dim, H))
            tensors[f"{prefix}.self_attn.k_proj.weight"] = _rand((NKV * head_dim, H))
            tensors[f"{prefix}.self_attn.v_proj.weight"] = _rand((NKV * head_dim, H))
            tensors[f"{prefix}.self_attn.o_proj.weight"] = _rand((H, NH * head_dim))

        if is_moe:
            tensors[f"{prefix}.block_sparse_moe.gate.weight"] = _rand((num_experts, H))
            for j in range(num_experts):
                exp_prefix = f"{prefix}.block_sparse_moe.experts.{j}"
                tensors[f"{exp_prefix}.w1.weight"] = _rand((I, H))
                tensors[f"{exp_prefix}.w2.weight"] = _rand((H, I))
                tensors[f"{exp_prefix}.w3.weight"] = _rand((I, H))
        else:
            tensors[f"{prefix}.mlp.gate_proj.weight"] = _rand((I, H))
            tensors[f"{prefix}.mlp.up_proj.weight"] = _rand((I, H))
            tensors[f"{prefix}.mlp.down_proj.weight"] = _rand((H, I))

        tensors[f"{prefix}.input_layernorm.weight"] = np.ones((H,), dtype=np.float16)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = np.ones((H,), dtype=np.float16)

    tensors["model.norm.weight"] = np.ones((H,), dtype=np.float16)

    files = []
    if shards <= 1:
        filepath = folder / "model.safetensors"
        _write_safetensors(filepath, tensors)
        files.append(filepath.name)
    else:
        names = sorted(tensors.keys())
        chunk = (len(names) + shards - 1) // shards
        for s in range(shards):
            batch = {n: tensors[n] for n in names[s * chunk:(s + 1) * chunk]}
            if not batch:
                continue
            filepath = folder / f"model-{s+1:05d}-of-{shards:05d}.safetensors"
            _write_safetensors(filepath, batch)
            files.append(filepath.name)

    return {"files": files, "num_tensors": len(tensors),
            "params": params, "is_moe": is_moe, "is_mla": is_mla}


def create_full_test_model(folder, size="tiny", overwrite=False,
                            tokenizer_type="word_level", random_weights=False):
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()) and not overwrite:
        raise FileExistsError(f"Папка {folder} не пуста. Используйте overwrite=True.")
    folder.mkdir(parents=True, exist_ok=True)
    for f in folder.iterdir():
        if f.is_file():
            f.unlink()

    params = create_test_config(folder / "config.json", size=size)

    if tokenizer_type == "byte_level":
        create_byte_level_tokenizer(folder / "tokenizer.json", vocab_size=params["vocab_size"])
    else:
        create_test_tokenizer(folder / "tokenizer.json", vocab_size=params["vocab_size"])

    result = create_test_safetensors(folder, params, random_weights=random_weights)
    total_bytes = sum((folder / f).stat().st_size for f in result["files"])

    return {
        "folder": str(folder), "size": size,
        "is_moe": result["is_moe"], "is_mla": result["is_mla"],
        "num_tensors": result["num_tensors"],
        "files": result["files"] + ["config.json", "tokenizer.json"],
        "total_bytes": total_bytes,
        "tokenizer_type": tokenizer_type,
        "params_summary": {
            k: params[k] for k in ("num_layers", "hidden_size", "vocab_size")
            if k in params
        },
    }


__all__ = [
    "SIZE_PRESETS",
    "create_test_config",
    "create_test_tokenizer",
    "create_byte_level_tokenizer",
    "create_test_safetensors",
    "create_full_test_model",
]