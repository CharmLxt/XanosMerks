"""Метрики nsofi40: eviction, абсолютная позиция, физический/виртуальный расход RAM."""


class NsofiMetrics:
    """Счётчики для NsofiKVCache и NsofiMLACache."""

    def __init__(self):
        self._total_appended = 0
        self._total_evicted = 0
        self._eviction_events = 0
        self._sink_filled_at = None
        self._window_full_at = None

    def record_append(self, n_tokens):
        self._total_appended += n_tokens

    def record_eviction(self, n_tokens):
        if n_tokens > 0:
            self._total_evicted += n_tokens
            self._eviction_events += 1

    def record_sink_filled(self, at_position):
        if self._sink_filled_at is None:
            self._sink_filled_at = at_position

    def record_window_full(self, at_position):
        if self._window_full_at is None:
            self._window_full_at = at_position

    def record_reset(self):
        self._total_appended = 0
        self._total_evicted = 0
        self._eviction_events = 0
        self._sink_filled_at = None
        self._window_full_at = None

    def snapshot(self):
        return {
            "total_appended": self._total_appended,
            "total_evicted": self._total_evicted,
            "eviction_events": self._eviction_events,
            "sink_filled_at": self._sink_filled_at,
            "window_full_at": self._window_full_at,
        }

    def __repr__(self):
        return (
            f"<NsofiMetrics appended={self._total_appended} "
            f"evicted={self._total_evicted} events={self._eviction_events}>"
        )


__all__ = ["NsofiMetrics"]