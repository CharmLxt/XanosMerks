"""Управление памятью: lazy TensorHandle, MemoryManager с бюджетом, TensorIndex.

Порядок захвата локов (обязателен во избежание дедлока):
    1. Сначала захватывается MemoryManager._lock (если удерживается).
    2. Потом TensorHandle._lock (если нужен).
Никогда наоборот.
"""
import fnmatch
import threading
from pathlib import Path

import numpy as np

from .support_tools.core_spr.exceptions import (
    LazyHandleClosedError, MemoryBudgetExceeded, OOMError, WeightLoadError,
)
from .support_tools.sup2_spr.stats import MemoryStats


_DTYPE_BYTES = {
    "float64": 8, "float32": 4, "float16": 2, "bfloat16": 2,
    "int64": 8, "int32": 4, "int16": 2, "int8": 1,
    "uint8": 1, "bool": 1, "fp8_e4m3": 1, "fp8_e5m2": 1,
}


class TensorHandle:
    """Ленивый указатель на тензор в файле."""

    def __init__(self, name, file_path, offset, shape, dtype,
                 data_start=0, memory_manager=None, target_dtype="auto"):
        """target_dtype: dtype для load(). "auto" — как в файле.
        Влияет на ram_nbytes (оценка/факт размера в RAM).
        """
        self.name = name
        self.file_path = Path(file_path)
        self.offset = offset
        self.shape = tuple(shape)
        self.dtype = dtype
        self.data_start = data_start
        self.target_dtype = target_dtype
        self._memory_manager = memory_manager

        self._array = None
        self._closed = False
        self._lock = threading.Lock()

    @property
    def is_loaded(self):
        return self._array is not None

    @property
    def is_closed(self):
        return self._closed

    @property
    def disk_nbytes(self):
        """Сколько байт занимает тензор на диске (по исходному safetensors dtype)."""
        from .loaders import SAFETENSORS_DTYPE, RAW_NUMPY_DTYPE
        our_dtype = SAFETENSORS_DTYPE.get(self.dtype)
        if our_dtype is None:
            return 0
        np_dtype = RAW_NUMPY_DTYPE.get(our_dtype, np.float32)
        n_elements = 1
        for d in self.shape:
            n_elements *= d
        return n_elements * np.dtype(np_dtype).itemsize

    @property
    def nbytes(self):
        """Алиас disk_nbytes для обратной совместимости."""
        return self.disk_nbytes

    @property
    def ram_nbytes(self):
        """Сколько байт тензор занимает (или займёт) в RAM.

        - Если загружен — фактический self._array.nbytes.
        - Если не загружен — оценка по target_dtype (или исходному dtype,
          если target_dtype="auto").
        """
        n_elements = 1
        for d in self.shape:
            n_elements *= d

        if self._array is not None:
            return self._array.nbytes

        if self.target_dtype == "auto":
            from .loaders import SAFETENSORS_DTYPE, RAW_NUMPY_DTYPE
            our_dtype = SAFETENSORS_DTYPE.get(self.dtype)
            if our_dtype is None:
                return 0
            np_dtype = RAW_NUMPY_DTYPE.get(our_dtype, np.float32)
            return n_elements * np.dtype(np_dtype).itemsize

        bytes_per = _DTYPE_BYTES.get(self.target_dtype, 4)
        return n_elements * bytes_per

    def load(self):
        """Загружает тензор в RAM. Возвращает np.ndarray.

        Использует self.target_dtype для конвертации (по умолчанию "auto").
        """
        if self._closed:
            raise LazyHandleClosedError(f"TensorHandle '{self.name}' уже закрыт")

        if self._array is not None:
            return self._array

        with self._lock:
            if self._array is not None:
                return self._array

            from .loaders import SAFETENSORS_DTYPE, RAW_NUMPY_DTYPE, _convert_tensor

            if not self.file_path.exists():
                raise WeightLoadError(f"Файл {self.file_path} не найден")

            our_dtype = SAFETENSORS_DTYPE.get(self.dtype)
            if our_dtype is None:
                raise WeightLoadError(
                    f"Неизвестный dtype '{self.dtype}' в тензоре '{self.name}'"
                )

            np_dtype = RAW_NUMPY_DTYPE[our_dtype]
            n_elements = 1
            for d in self.shape:
                n_elements *= d
            n_bytes = n_elements * np.dtype(np_dtype).itemsize

            abs_offset = self.data_start + self.offset
            with open(self.file_path, "rb") as f:
                f.seek(abs_offset)
                raw = f.read(n_bytes)
                if len(raw) < n_bytes:
                    raise WeightLoadError(
                        f"Тензор '{self.name}': неполные данные в файле "
                        f"(получено {len(raw)}, ожидалось {n_bytes})"
                    )

            arr = np.frombuffer(raw, dtype=np_dtype, count=n_elements)
            if self.shape:
                arr = arr.reshape(self.shape)

            converted = _convert_tensor(arr, our_dtype, self.target_dtype)
            if converted.base is not None:
                converted = converted.copy()

            self._array = converted
            nbytes = converted.nbytes

        if self._memory_manager is not None:
            try:
                self._memory_manager._on_load(self.name, nbytes)
            except MemoryBudgetExceeded:
                with self._lock:
                    self._array = None
                raise

        return self._array

    def unload(self):
        """Освобождает память тензора."""
        nbytes = 0
        freed = False
        with self._lock:
            if self._array is not None:
                nbytes = self._array.nbytes
                self._array = None
                freed = True

        if freed and self._memory_manager is not None:
            self._memory_manager._on_free(self.name, nbytes)

    def close(self):
        self.unload()
        self._closed = True

    @property
    def value(self):
        return self.load()

    def __repr__(self):
        status = "loaded" if self.is_loaded else ("closed" if self._closed else "lazy")
        return (
            f"<TensorHandle {self.name} shape={self.shape} "
            f"disk={self.disk_nbytes}B ram={self.ram_nbytes}B {status}>"
        )


class MemoryManager:
    """Управляет бюджетом памяти и всеми зарегистрированными TensorHandle."""

    def __init__(self, max_bytes=None, eviction_policy="lru", safety_margin=0.0):
        self.max_bytes = max_bytes
        self.eviction_policy = eviction_policy
        self.safety_margin = safety_margin
        self.used_bytes = 0
        self.handles = {}
        self._lru_order = []
        self.stats = MemoryStats()
        self._lock = threading.RLock()

    @classmethod
    def auto_configure(cls, fraction=0.6, safety_margin=0.1):
        try:
            import psutil
            total = psutil.virtual_memory().total
            max_bytes = int(total * fraction)
        except ImportError:
            import os
            try:
                pages = os.sysconf("SC_PHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                total = pages * page_size
                max_bytes = int(total * fraction)
            except Exception:
                max_bytes = None

        return cls(max_bytes=max_bytes, eviction_policy="lru",
                   safety_margin=safety_margin)

    def register(self, handle):
        with self._lock:
            self.handles[handle.name] = handle
            handle._memory_manager = self

    def register_all(self, handles):
        if isinstance(handles, dict):
            handles = handles.values()
        for h in handles:
            self.register(h)

    def _on_load(self, name, nbytes):
        with self._lock:
            if self.max_bytes is not None and self.safety_margin > 0:
                effective_max = int(self.max_bytes * (1 - self.safety_margin))
            else:
                effective_max = self.max_bytes

            if effective_max is not None:
                while self.used_bytes + nbytes > effective_max:
                    if not self._evict_one_locked():
                        raise MemoryBudgetExceeded(
                            f"Недостаточно памяти: нужно {nbytes} байт, "
                            f"доступно {effective_max - self.used_bytes}"
                        )

            self.used_bytes += nbytes
            if name in self._lru_order:
                self._lru_order.remove(name)
            self._lru_order.append(name)

            self.stats.record_load(nbytes, dtype="auto", source="lazy")

    def _on_free(self, name, nbytes):
        with self._lock:
            self.used_bytes = max(0, self.used_bytes - nbytes)
            if name in self._lru_order:
                self._lru_order.remove(name)
            self.stats.record_free(nbytes)

    def _evict_one_locked(self):
        if self.eviction_policy == "none":
            return False

        for name in list(self._lru_order):
            h = self.handles.get(name)
            if h is not None and h.is_loaded:
                h.unload()
                return True
        return False

    def load(self, name):
        if name not in self.handles:
            raise KeyError(f"Тензор '{name}' не зарегистрирован")
        return self.handles[name].load()

    def unload(self, name):
        if name in self.handles:
            self.handles[name].unload()

    def unload_all(self):
        for h in self.handles.values():
            if h.is_loaded:
                h.unload()

    def free(self):
        self.unload_all()
        with self._lock:
            self.used_bytes = 0
            self._lru_order.clear()

    @property
    def available_bytes(self):
        if self.max_bytes is None:
            return None
        return max(0, self.max_bytes - self.used_bytes)

    def get_stats(self):
        return self.stats.snapshot()

    def summary(self):
        total_handles = len(self.handles)
        loaded = sum(1 for h in self.handles.values() if h.is_loaded)
        return {
            "handles": total_handles,
            "loaded": loaded,
            "lazy": total_handles - loaded,
            "used_bytes": self.used_bytes,
            "used_human": _human(self.used_bytes),
            "max_bytes": self.max_bytes,
            "available_bytes": self.available_bytes,
            "peak_bytes": self.stats.peak_usage,
            "peak_human": _human(self.stats.peak_usage),
        }

    def __repr__(self):
        s = self.summary()
        max_str = _human(s["max_bytes"]) if s["max_bytes"] else "unlimited"
        return f"<MemoryManager used={s['used_human']}/{max_str}, loaded={s['loaded']}/{s['handles']}>"


def _human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


class TensorIndex:
    """Удобный доступ к TensorHandle по имени, префиксу или glob-паттерну."""

    def __init__(self, handles):
        if isinstance(handles, list):
            self._handles = {h.name: h for h in handles}
        else:
            self._handles = dict(handles)

    def __getitem__(self, key):
        if "*" in key:
            return {
                name: h for name, h in self._handles.items()
                if fnmatch.fnmatch(name, key)
            }
        if key in self._handles:
            return self._handles[key]
        prefix = key.rstrip(".") + "."
        result = {
            name: h for name, h in self._handles.items()
            if name.startswith(prefix)
        }
        if result:
            return result
        raise KeyError(f"Тензор '{key}' не найден и префикс не дал результатов")

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __iter__(self):
        return iter(self._handles)

    def __len__(self):
        return len(self._handles)

    def keys(self):
        return self._handles.keys()

    def values(self):
        return self._handles.values()

    def items(self):
        return self._handles.items()

    def get(self, name, default=None):
        return self._handles.get(name, default)

    def load(self, name):
        if name not in self._handles:
            raise KeyError(f"Тензор '{name}' не найден")
        return self._handles[name].load()

    def load_prefix(self, prefix):
        prefix = prefix.rstrip(".") + "."
        result = {}
        for name, h in self._handles.items():
            if name.startswith(prefix):
                result[name] = h.load()
        return result

    def load_all(self):
        return {name: h.load() for name, h in self._handles.items()}

    def unload_all(self):
        for h in self._handles.values():
            if h.is_loaded:
                h.unload()

    def summary(self):
        total = len(self._handles)
        loaded = sum(1 for h in self._handles.values() if h.is_loaded)
        total_disk = sum(h.disk_nbytes for h in self._handles.values())
        total_ram = sum(h.ram_nbytes for h in self._handles.values())
        return {
            "total_handles": total,
            "loaded": loaded,
            "lazy": total - loaded,
            "total_disk_bytes": total_disk,
            "total_disk_human": _human(total_disk),
            "total_ram_bytes": total_ram,
            "total_ram_human": _human(total_ram),
            # backwards compat
            "total_bytes": total_disk,
            "total_human": _human(total_disk),
        }

    def __repr__(self):
        s = self.summary()
        return (
            f"<TensorIndex {s['loaded']}/{s['total_handles']} loaded, "
            f"disk={s['total_disk_human']}, ram={s['total_ram_human']}>"
        )


def create_handles_from_safetensors(path, memory_manager=None,
                                    target_dtype="auto"):
    """Создаёт TensorHandle из .safetensors без загрузки данных в RAM.

    Параметры:
        path: путь к .safetensors файлу или к папке с шардами
        memory_manager: если передан — все handles регистрируются в нём
        target_dtype: dtype для load() каждого handle. "auto" — как в файле.
            Влияет на ram_nbytes (оценку размера в RAM).
    """
    from .loaders import _read_safetensors_header, SAFETENSORS_DTYPE

    path = Path(path)
    if not path.exists():
        raise WeightLoadError(f"Путь не существует: {path}")

    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("*.safetensors"))
        if not files:
            raise WeightLoadError(f"В {path} нет .safetensors файлов")

    handles = {}
    for file_path in files:
        header, header_size = _read_safetensors_header(file_path)
        data_start = 8 + header_size

        for name, info in header.items():
            if name == "__metadata__":
                continue
            shape = info.get("shape", [])
            st_dtype = info.get("dtype", "F16")
            start, _ = info.get("data_offsets", [0, 0])

            if st_dtype not in SAFETENSORS_DTYPE:
                raise WeightLoadError(
                    f"Неизвестный dtype '{st_dtype}' в тензоре '{name}'"
                )

            if name in handles:
                raise WeightLoadError(
                    f"Тензор '{name}' встречается в нескольких файлах"
                )

            handle = TensorHandle(
                name=name,
                file_path=file_path,
                offset=start,
                shape=shape,
                dtype=st_dtype,
                data_start=data_start,
                memory_manager=memory_manager,
                target_dtype=target_dtype,
            )
            handles[name] = handle

            if memory_manager is not None:
                memory_manager.register(handle)

    return TensorIndex(handles)


__all__ = [
    "TensorHandle",
    "MemoryManager",
    "TensorIndex",
    "create_handles_from_safetensors",
]