"""Загрузчики весов моделей в numpy-массивы."""
import json
import mmap
import struct
from pathlib import Path

import numpy as np

from .support_tools.core_spr.exceptions import (
    WeightLoadError, UnsupportedDTypeError, OOMError, TensorShapeError,
)
from .support_tools.core_spr.dtype_ops import (
    bf16_to_fp32, fp32_to_bf16, fp8_to_fp32,
    int8_to_fp16, normalize_dtype_name,
)
from .support_tools.core_spr.gguf_dequant import dequantize as gguf_dequantize
from .support_tools.sup2_spr.stats import MemoryStats

from ..gamtype.readers import parse_gguf as _gamtype_parse_gguf
from ..gamtype.support_tools.sup2_spr.path_utils import SHARD_PATTERN


SAFETENSORS_DTYPE = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
    "F8_E4M3": "fp8_e4m3",
    "F8_E5M2": "fp8_e5m2",
}

RAW_NUMPY_DTYPE = {
    "float64": np.float64,
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": np.uint16,
    "int64": np.int64,
    "int32": np.int32,
    "int16": np.int16,
    "int8": np.int8,
    "uint8": np.uint8,
    "bool": np.bool_,
    "fp8_e4m3": np.uint8,
    "fp8_e5m2": np.uint8,
}


def _convert_tensor(arr, source_dtype, target_dtype):
    target = normalize_dtype_name(target_dtype) if target_dtype != "auto" else "auto"

    if target == "auto":
        return arr

    source = normalize_dtype_name(source_dtype)

    if source == target:
        return arr

    if source == "bfloat16":
        if target == "float32":
            return bf16_to_fp32(arr)
        elif target == "float16":
            return bf16_to_fp32(arr).astype(np.float16)
        else:
            raise UnsupportedDTypeError(f"BF16 → {target}: не поддерживается")

    if source in ("fp8_e4m3", "fp8_e5m2"):
        variant = "e4m3" if source == "fp8_e4m3" else "e5m2"
        if target == "float32":
            return fp8_to_fp32(arr, variant=variant)
        elif target == "float16":
            return fp8_to_fp32(arr, variant=variant).astype(np.float16)
        else:
            raise UnsupportedDTypeError(f"{source} → {target}: не поддерживается")

    if source == "int8" and target in ("float16", "float32"):
        if target == "float16":
            return int8_to_fp16(arr)
        return arr.astype(np.float32)

    try:
        return arr.astype(target)
    except Exception as e:
        raise UnsupportedDTypeError(f"Не могу конвертировать {source} → {target}: {e}")


def _read_safetensors_header(filepath):
    path = Path(filepath)
    try:
        with open(path, "rb") as f:
            size_bytes = f.read(8)
            if len(size_bytes) != 8:
                raise WeightLoadError(f"{path.name} слишком мал для safetensors")
            header_size = struct.unpack("<Q", size_bytes)[0]
            if header_size > 100 * 1024 * 1024:
                raise WeightLoadError(
                    f"Заголовок {path.name} слишком велик ({header_size} байт)"
                )
            header_json = f.read(header_size)
            header = json.loads(header_json.decode("utf-8"))
            return header, header_size
    except FileNotFoundError:
        raise WeightLoadError(f"Файл не найден: {filepath}")
    except json.JSONDecodeError as e:
        raise WeightLoadError(f"Некорректный JSON в {path.name}: {e}")
    except OSError as e:
        raise WeightLoadError(f"Ошибка чтения {path.name}: {e}")


def _load_safetensors_file(filepath, mmap_mode=True, dtype="auto",
                           stats=None, source_label=None):
    path = Path(filepath)
    header, header_size = _read_safetensors_header(path)
    data_start = 8 + header_size

    tensors = {}
    metadata = header.get("__metadata__", {})
    total_bytes = path.stat().st_size
    source_dtypes = {}

    if mmap_mode:
        f = open(path, "rb")
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            f.close()
            raise

        try:
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                shape = info.get("shape", [])
                st_dtype = info.get("dtype", "unknown")
                start, end = info.get("data_offsets", [0, 0])

                if st_dtype not in SAFETENSORS_DTYPE:
                    raise UnsupportedDTypeError(
                        f"Неизвестный dtype '{st_dtype}' в {path.name}:{name}"
                    )

                our_dtype = SAFETENSORS_DTYPE[st_dtype]
                source_dtypes[name] = our_dtype
                np_dtype = RAW_NUMPY_DTYPE[our_dtype]

                n_elements = int(np.prod(shape)) if shape else 1
                expected_bytes = n_elements * np.dtype(np_dtype).itemsize
                actual_bytes = end - start
                if expected_bytes != actual_bytes:
                    raise TensorShapeError(
                        f"{path.name}:{name}: ожидалось {expected_bytes} байт, "
                        f"в файле {actual_bytes} байт"
                    )

                arr = np.frombuffer(
                    mm, dtype=np_dtype, count=n_elements, offset=data_start + start
                )
                if shape:
                    arr = arr.reshape(shape)

                converted = _convert_tensor(arr, our_dtype, dtype)

                if converted.base is not None:
                    converted = converted.copy()

                tensors[name] = converted

                if stats is not None:
                    stats.record_load(
                        converted.nbytes,
                        dtype=str(converted.dtype),
                        source=source_label or "safetensors",
                    )

        finally:
            try:
                mm.close()
            except Exception:
                pass
            f.close()
    else:
        raw = path.read_bytes()

        for name, info in header.items():
            if name == "__metadata__":
                continue
            shape = info.get("shape", [])
            st_dtype = info.get("dtype", "unknown")
            start, end = info.get("data_offsets", [0, 0])

            if st_dtype not in SAFETENSORS_DTYPE:
                raise UnsupportedDTypeError(f"Неизвестный dtype '{st_dtype}'")

            our_dtype = SAFETENSORS_DTYPE[st_dtype]
            source_dtypes[name] = our_dtype
            np_dtype = RAW_NUMPY_DTYPE[our_dtype]
            n_elements = int(np.prod(shape)) if shape else 1

            arr = np.frombuffer(
                raw, dtype=np_dtype, count=n_elements, offset=data_start + start
            )
            if shape:
                arr = arr.reshape(shape)

            converted = _convert_tensor(arr, our_dtype, dtype)
            tensors[name] = converted

            if stats is not None:
                stats.record_load(
                    converted.nbytes,
                    dtype=str(converted.dtype),
                    source=source_label or "safetensors",
                )

    meta = {
        "format": "safetensors",
        "file": str(path),
        "file_size_bytes": total_bytes,
        "header_size_bytes": header_size,
        "num_tensors": len(tensors),
        "metadata": metadata,
        "source_dtypes": source_dtypes,
    }
    return tensors, meta


def load_safetensors(path, mmap=True, dtype="auto", stats=None):
    path = Path(path)
    if not path.exists():
        raise WeightLoadError(f"Путь не существует: {path}")

    if path.is_file():
        return _load_safetensors_file(path, mmap_mode=mmap, dtype=dtype, stats=stats)

    all_files = list(path.rglob("*.safetensors"))
    if not all_files:
        raise WeightLoadError(f"В {path} не найдено .safetensors файлов")

    sharded_groups = {}
    standalone_files = []
    for f in all_files:
        m = SHARD_PATTERN.match(f.name)
        if m:
            base = m.group(1)
            sharded_groups.setdefault(base, []).append(f)
        else:
            standalone_files.append(f)

    tensors = {}
    file_metas = []
    seen_names = set()
    source_dtypes = {}

    for base, files in sharded_groups.items():
        files.sort(key=lambda p: int(SHARD_PATTERN.match(p.name).group(2)))
        for f in files:
            file_tensors, fmeta = _load_safetensors_file(
                f, mmap_mode=mmap, dtype=dtype, stats=stats,
            )
            for name, arr in file_tensors.items():
                if name in seen_names:
                    raise WeightLoadError(
                        f"Тензор '{name}' встречается в нескольких шардах"
                    )
                seen_names.add(name)
                tensors[name] = arr
                source_dtypes[name] = fmeta["source_dtypes"][name]
            file_metas.append(fmeta)

    for f in standalone_files:
        file_tensors, fmeta = _load_safetensors_file(
            f, mmap_mode=mmap, dtype=dtype, stats=stats,
        )
        for name, arr in file_tensors.items():
            if name in seen_names:
                raise WeightLoadError(
                    f"Тензор '{name}' встречается в нескольких файлах"
                )
            seen_names.add(name)
            tensors[name] = arr
            source_dtypes[name] = fmeta["source_dtypes"][name]
        file_metas.append(fmeta)

    meta = {
        "format": "safetensors",
        "path": str(path),
        "is_sharded": len(sharded_groups) > 0,
        "num_shards": sum(len(v) for v in sharded_groups.values()) + len(standalone_files),
        "num_tensors": len(tensors),
        "files": file_metas,
        "source_dtypes": source_dtypes,
    }
    return tensors, meta


def load_gguf(path, dtype="float16", stats=None):
    path = Path(path)
    if not path.exists():
        raise WeightLoadError(f"Файл не найден: {path}")

    try:
        gguf_info = _gamtype_parse_gguf(path)
    except Exception as e:
        raise WeightLoadError(f"Не удалось разобрать GGUF {path.name}: {e}")

    data_start = gguf_info.get("data_start", 0)
    file_size = path.stat().st_size

    f = open(path, "rb")
    try:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    except Exception:
        f.close()
        raise

    tensors = {}
    source_dtypes = {}

    try:
        for tensor_info in gguf_info["tensors"]:
            name = tensor_info["name"]
            shape = tensor_info["shape"]
            type_name = tensor_info["type"]
            offset = tensor_info["offset"]
            size_bytes = tensor_info["size_bytes"]

            abs_start = data_start + offset
            abs_end = abs_start + size_bytes
            if abs_end > file_size:
                raise WeightLoadError(
                    f"Тензор {name}: данные выходят за пределы файла"
                )

            raw_view = mm[abs_start:abs_end]

            try:
                arr = gguf_dequantize(raw_view, type_name, shape)
            except NotImplementedError as e:
                raise UnsupportedDTypeError(
                    f"Тензор {name}: тип {type_name} не поддерживается. {e}"
                )
            except Exception as e:
                raise WeightLoadError(
                    f"Ошибка деквантования {name} ({type_name}): {e}"
                )

            target = normalize_dtype_name(dtype)
            if target != arr.dtype.name:
                arr = arr.astype(np.dtype(target))

            tensors[name] = arr
            source_dtypes[name] = target

            if stats is not None:
                stats.record_load(
                    arr.nbytes,
                    dtype=str(arr.dtype),
                    source="gguf",
                )

    finally:
        try:
            mm.close()
        except Exception:
            pass
        f.close()

    meta = {
        "format": "gguf",
        "file": str(path),
        "file_size_bytes": file_size,
        "num_tensors": len(tensors),
        "architecture": gguf_info.get("architecture", "unknown"),
        "model_name": gguf_info.get("model_name", ""),
        "version": gguf_info.get("version"),
        "metadata": gguf_info.get("metadata", {}),
        "source_dtypes": source_dtypes,
    }
    return tensors, meta


def load_npy(path, stats=None):
    path = Path(path)
    if not path.exists():
        raise WeightLoadError(f"Файл не найден: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npy":
        try:
            arr = np.load(str(path), allow_pickle=False)
        except Exception as e:
            raise WeightLoadError(f"Ошибка чтения {path.name}: {e}")

        tensors = {"__single__": arr}
        if stats is not None:
            stats.record_load(arr.nbytes, dtype=str(arr.dtype), source="npy")

        meta = {
            "format": "npy",
            "file": str(path),
            "file_size_bytes": path.stat().st_size,
            "num_tensors": 1,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "source_dtypes": {"__single__": normalize_dtype_name(arr.dtype.name)},
        }
        return tensors, meta

    elif suffix == ".npz":
        try:
            data = np.load(str(path), allow_pickle=False)
            tensors = {k: data[k] for k in data.files}
            data.close()
        except Exception as e:
            raise WeightLoadError(f"Ошибка чтения {path.name}: {e}")

        if stats is not None:
            for arr in tensors.values():
                stats.record_load(arr.nbytes, dtype=str(arr.dtype), source="npy")

        meta = {
            "format": "npz",
            "file": str(path),
            "file_size_bytes": path.stat().st_size,
            "num_tensors": len(tensors),
            "source_dtypes": {
                k: normalize_dtype_name(v.dtype.name) for k, v in tensors.items()
            },
        }
        return tensors, meta

    else:
        raise WeightLoadError(
            f"{path.name}: неизвестный формат. Ожидалось .npy или .npz."
        )


def load_model_weights(folder_path, prefer="auto", dtype="auto",
                       mmap=True, stats=None):
    folder = Path(folder_path)
    if not folder.exists():
        raise WeightLoadError(f"Папка не существует: {folder}")

    if stats is None:
        stats = MemoryStats()

    st_files = list(folder.rglob("*.safetensors"))
    if prefer in ("auto", "safetensors") and st_files:
        return load_safetensors(folder, mmap=mmap, dtype=dtype, stats=stats)

    gguf_files = list(folder.rglob("*.gguf"))
    if prefer in ("auto", "gguf") and gguf_files:
        if len(gguf_files) > 1:
            raise WeightLoadError(
                f"Найдено {len(gguf_files)} .gguf файлов в {folder}."
            )
        target_dtype = "float16" if dtype == "auto" else dtype
        return load_gguf(gguf_files[0], dtype=target_dtype, stats=stats)

    npy_files = list(folder.rglob("*.npy")) + list(folder.rglob("*.npz"))
    if prefer in ("auto", "npy") and npy_files:
        if len(npy_files) > 1:
            all_tensors = {}
            all_metas = []
            combined_source_dtypes = {}
            for nf in npy_files:
                t, m = load_npy(nf, stats=stats)
                for k, v in t.items():
                    if k == "__single__":
                        all_tensors[nf.stem] = v
                        combined_source_dtypes[nf.stem] = m["source_dtypes"]["__single__"]
                    else:
                        all_tensors[k] = v
                        combined_source_dtypes[k] = m["source_dtypes"].get(k, "unknown")
                all_metas.append(m)
            return all_tensors, {
                "format": "npy_multi",
                "path": str(folder),
                "num_tensors": len(all_tensors),
                "files": all_metas,
                "source_dtypes": combined_source_dtypes,
            }
        else:
            return load_npy(npy_files[0], stats=stats)

    raise WeightLoadError(
        f"В {folder} не найдено ни .safetensors, ни .gguf, ни .npy."
    )


__all__ = [
    "load_safetensors",
    "load_gguf",
    "load_npy",
    "load_model_weights",
    "SAFETENSORS_DTYPE",
    "RAW_NUMPY_DTYPE",
]