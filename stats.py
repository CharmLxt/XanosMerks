"""Статистика использования памяти meanflare."""
import time
from collections import defaultdict


class MemoryStats:
    """Собирает и агрегирует информацию о загрузке/освобождении тензоров."""

    def __init__(self):
        self.total_loaded = 0       # сколько всего байт было загружено (суммарно, с повторами)
        self.total_freed = 0        # сколько всего освобождено
        self.peak_usage = 0         # пиковое использование (байт)
        self.current_usage = 0      # текущее использование
        self.load_count = 0
        self.free_count = 0
        self.per_dtype = defaultdict(int)  # bytes loaded per dtype
        self.per_source = defaultdict(int)  # bytes per source (safetensors/gguf/npy)
        self.timeline = []          # [(timestamp, event, size_bytes)]
        self._max_timeline = 1000   # ограничение, чтобы не жрать RAM

    def record_load(self, size_bytes, dtype="unknown", source="unknown"):
        """Фиксирует загрузку тензора в память."""
        self.total_loaded += size_bytes
        self.current_usage += size_bytes
        self.load_count += 1
        self.per_dtype[dtype] += size_bytes
        self.per_source[source] += size_bytes

        if self.current_usage > self.peak_usage:
            self.peak_usage = self.current_usage

        self._add_timeline("load", size_bytes)

    def record_free(self, size_bytes):
        """Фиксирует освобождение тензора."""
        self.total_freed += size_bytes
        self.current_usage = max(0, self.current_usage - size_bytes)
        self.free_count += 1
        self._add_timeline("free", size_bytes)

    def _add_timeline(self, event, size_bytes):
        self.timeline.append((time.time(), event, size_bytes))
        if len(self.timeline) > self._max_timeline:
            self.timeline = self.timeline[-self._max_timeline:]

    def snapshot(self):
        """Возвращает снимок текущего состояния."""
        return {
            "total_loaded_bytes": self.total_loaded,
            "total_loaded_human": _human(self.total_loaded),
            "total_freed_bytes": self.total_freed,
            "total_freed_human": _human(self.total_freed),
            "current_usage_bytes": self.current_usage,
            "current_usage_human": _human(self.current_usage),
            "peak_usage_bytes": self.peak_usage,
            "peak_usage_human": _human(self.peak_usage),
            "load_count": self.load_count,
            "free_count": self.free_count,
            "per_dtype": {k: _human(v) for k, v in self.per_dtype.items()},
            "per_source": {k: _human(v) for k, v in self.per_source.items()},
        }

    def reset(self):
        """Сбрасывает статистику."""
        self.__init__()

    def __repr__(self):
        return (
            f"<MemoryStats current={_human(self.current_usage)}, "
            f"peak={_human(self.peak_usage)}, loads={self.load_count}>"
        )


def _human(n_bytes):
    """Читаемое представление байт."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.2f} PB"


__all__ = ["MemoryStats"]