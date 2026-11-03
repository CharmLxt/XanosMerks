"""MemoryMonitor — кроссплатформенное чтение RSS и доступной RAM."""
import threading
import time

from .support_tools.core_spr.system_info import (
    get_system_memory,
    get_process_rss,
    get_process_peak_rss,
    human_bytes,
)


class MemorySnapshot:
    """Снимок памяти в один момент времени."""

    def __init__(self, rss=None, available=None, total=None,
                 peak_rss=None, percent_used=None):
        self.rss = rss
        self.available = available
        self.total = total
        self.peak_rss = peak_rss
        self.percent_used = percent_used

    @property
    def rss_human(self):
        return human_bytes(self.rss)

    @property
    def available_human(self):
        return human_bytes(self.available)

    @property
    def total_human(self):
        return human_bytes(self.total)

    @property
    def peak_rss_human(self):
        return human_bytes(self.peak_rss)

    def to_dict(self):
        return {
            "rss": self.rss,
            "rss_human": self.rss_human,
            "available": self.available,
            "available_human": self.available_human,
            "total": self.total,
            "total_human": self.total_human,
            "peak_rss": self.peak_rss,
            "peak_rss_human": self.peak_rss_human,
            "percent_used": self.percent_used,
        }

    def __repr__(self):
        return (
            f"<MemorySnapshot rss={self.rss_human} "
            f"available={self.available_human} "
            f"percent={self.percent_used}>"
        )


class MemoryMonitor:
    """Мониторинг памяти.

    Pull-режим с TTL-кэшем (по умолчанию 100 мс) + опциональный фоновый поток.

    Параметры:
        interval: секунды между проверками в фоне (если start()).
        cache_ttl: сколько секунд держать последний MemorySnapshot.
            0 — без кэша (каждый вызов читает системные счётчики).
    """

    def __init__(self, interval=1.0, cache_ttl=0.1):
        self.interval = interval
        self.cache_ttl = cache_ttl
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._cached_snapshot = None
        self._cached_at = 0.0

    def snapshot(self):
        """Возвращает MemorySnapshot (из кэша, если TTL не истёк)."""
        if self.cache_ttl > 0:
            with self._lock:
                now = time.time()
                if (self._cached_snapshot is not None
                        and (now - self._cached_at) < self.cache_ttl):
                    return self._cached_snapshot

        mem = get_system_memory() or {}
        snap = MemorySnapshot(
            rss=get_process_rss(),
            available=mem.get("available"),
            total=mem.get("total"),
            peak_rss=get_process_peak_rss(),
            percent_used=mem.get("used_percent"),
        )

        if self.cache_ttl > 0:
            with self._lock:
                self._cached_snapshot = snap
                self._cached_at = time.time()
        return snap

    def invalidate(self):
        """Сбрасывает кэш: следующий snapshot() прочитает счётчики заново."""
        with self._lock:
            self._cached_snapshot = None
            self._cached_at = 0.0

    def current_rss(self):
        return self.snapshot().rss

    def available(self):
        return self.snapshot().available

    def total(self):
        return self.snapshot().total

    def peak_rss(self):
        return self.snapshot().peak_rss

    def percent_used(self):
        return self.snapshot().percent_used

    def check_available(self, nbytes, safety_margin=0.0):
        avail = self.available()
        if avail is None:
            return None
        threshold = avail * (1.0 - safety_margin)
        return nbytes <= threshold

    def _loop(self):
        while not self._stop.is_set():
            snap = self.snapshot()
            with self._lock:
                self._latest = snap
            self._stop.wait(self.interval)

    def start(self):
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def is_running(self):
        return bool(self._thread and self._thread.is_alive())

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    def __repr__(self):
        return (
            f"<MemoryMonitor interval={self.interval} "
            f"cache_ttl={self.cache_ttl} running={self.is_running()}>"
        )


__all__ = ["MemoryMonitor", "MemorySnapshot"]