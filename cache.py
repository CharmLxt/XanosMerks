"""NsofiKVCache — KV-кэш со sliding window и attention sink."""
import threading
import numpy as np

from .metrics import NsofiMetrics
from .support_tools.core_spr.exceptions import (
    NsofiError,
    WindowOverflowError,
    SinkConfigError,
    EvictionError,
    CacheStateError,
)
from .support_tools.sup2_spr.ring import write_range


def _human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


class NsofiKVCache:
    """KV-кэш со sliding window, attention sink и H2O eviction."""

    def __init__(self, num_layers, num_kv_heads, head_dim, max_seq_len,
                 window_size=4096, sink_size=4, eviction="sink_window",
                 batch_size=1, dtype=np.float32):
        if window_size < 1:
            raise SinkConfigError(f"window_size должен быть >= 1, получено {window_size}")
        if sink_size < 0:
            raise SinkConfigError(f"sink_size должен быть >= 0, получено {sink_size}")
        if sink_size > window_size:
            raise SinkConfigError(
                f"sink_size ({sink_size}) > window_size ({window_size})"
            )
        if eviction not in ("sink_window", "fifo", "none", "h2o"):
            raise EvictionError(f"Неизвестная политика eviction: {eviction}")
        if max_seq_len < 1:
            raise NsofiError(f"max_seq_len должен быть >= 1, получено {max_seq_len}")
        if batch_size < 1:
            raise NsofiError(f"batch_size должен быть >= 1, получено {batch_size}")

        if eviction == "none":
            if max_seq_len > sink_size + window_size:
                window_size = max_seq_len
            sink_size = 0
        elif eviction in ("fifo", "h2o"):
            sink_size = 0

        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.sink_size = sink_size
        self.eviction = eviction
        self.batch_size = batch_size
        self.dtype = dtype

        self._use_ring = eviction in ("sink_window", "fifo", "none")

        if sink_size > 0:
            sink_shape = (num_layers, batch_size, num_kv_heads, sink_size, head_dim)
            self._sink_k = np.zeros(sink_shape, dtype=dtype)
            self._sink_v = np.zeros(sink_shape, dtype=dtype)
        else:
            self._sink_k = None
            self._sink_v = None

        window_capacity = 2 * window_size if self._use_ring else window_size
        window_shape = (num_layers, batch_size, num_kv_heads, window_capacity, head_dim)
        self._window_k = np.zeros(window_shape, dtype=dtype)
        self._window_v = np.zeros(window_shape, dtype=dtype)

        if eviction == "h2o":
            self._accumulated_scores = np.zeros(
                (num_layers, batch_size, num_kv_heads, window_size),
                dtype=np.float32,
            )
        else:
            self._accumulated_scores = None

        self._sink_count = 0
        self._window_count = 0
        self._head = 0
        self._absolute_position = 0
        self._metrics = NsofiMetrics()
        self._lock = threading.RLock()

    @property
    def absolute_position(self):
        with self._lock:
            return self._absolute_position

    @property
    def cur_len(self):
        with self._lock:
            return self._sink_count + self._window_count

    @property
    def needs_attention_scores(self):
        return self.eviction == "h2o"

    @property
    def k(self):
        with self._lock:
            return self._get_logical(self._sink_k, self._window_k)

    @property
    def v(self):
        with self._lock:
            return self._get_logical(self._sink_v, self._window_v)

    def _get_logical(self, sink_buf, window_buf):
        parts = []
        if self._sink_count > 0 and sink_buf is not None:
            parts.append(sink_buf[:, :, :, :self._sink_count, :])
        if self._window_count > 0:
            if self._use_ring:
                start = (self._head - self._window_count) % self.window_size
                parts.append(window_buf[:, :, :, start:start + self._window_count, :])
            else:
                parts.append(window_buf[:, :, :, :self._window_count, :])
        if len(parts) == 1:
            return parts[0]
        if len(parts) == 0:
            return np.zeros(
                (self.num_layers, self.batch_size, self.num_kv_heads, 0, self.head_dim),
                dtype=self.dtype,
            )
        return np.concatenate(parts, axis=-2)

    def append(self, layer_idx, k_new, v_new):
        with self._lock:
            if not (0 <= layer_idx < self.num_layers):
                raise CacheStateError(f"layer_idx {layer_idx} вне [0, {self.num_layers})")
            if k_new.shape != v_new.shape:
                raise CacheStateError(
                    f"k_new.shape {k_new.shape} != v_new.shape {v_new.shape}"
                )
            n_new = k_new.shape[2]
            if n_new < 1:
                return
            if k_new.shape[0] != self.batch_size:
                raise CacheStateError(
                    f"batch_size {k_new.shape[0]} != {self.batch_size}"
                )
            if k_new.shape[1] != self.num_kv_heads:
                raise CacheStateError(
                    f"num_kv_heads {k_new.shape[1]} != {self.num_kv_heads}"
                )
            if k_new.shape[3] != self.head_dim:
                raise CacheStateError(
                    f"head_dim {k_new.shape[3]} != {self.head_dim}"
                )

            if self.eviction == "none":
                self._append_none(layer_idx, k_new, v_new, n_new)
            elif self.eviction == "h2o":
                self._append_h2o(layer_idx, k_new, v_new, n_new)
            else:
                self._append_ring(layer_idx, k_new, v_new, n_new)

    def _append_none(self, layer_idx, k_new, v_new, n_new):
        if self._window_count + n_new > self.window_size:
            raise WindowOverflowError(
                f"eviction='none': window_count={self._window_count} + "
                f"n={n_new} > W={self.window_size}"
            )
        w0 = self._window_count
        w1 = w0 + n_new
        ws0 = w0 + self.window_size
        ws1 = ws0 + n_new
        self._window_k[layer_idx, :, :, w0:w1, :] = k_new
        self._window_v[layer_idx, :, :, w0:w1, :] = v_new
        self._window_k[layer_idx, :, :, ws0:ws1, :] = k_new
        self._window_v[layer_idx, :, :, ws0:ws1, :] = v_new
        self._window_count = w1
        self._head = self._window_count % self.window_size
        self._metrics.record_append(n_new)

    def _append_ring(self, layer_idx, k_new, v_new, n_new):
        if n_new > self.window_size:
            raise WindowOverflowError(
                f"За один append нельзя записать {n_new} токенов при "
                f"window_size={self.window_size}."
            )

        sink_remaining = self.sink_size - self._sink_count
        n_to_sink = min(n_new, sink_remaining)
        n_to_window = n_new - n_to_sink

        if n_to_sink > 0 and self._sink_k is not None:
            s_start = self._sink_count
            s_end = s_start + n_to_sink
            self._sink_k[layer_idx, :, :, s_start:s_end, :] = k_new[:, :, :n_to_sink, :]
            self._sink_v[layer_idx, :, :, s_start:s_end, :] = v_new[:, :, :n_to_sink, :]
            self._sink_count = s_end

        if n_to_window > 0:
            write_range(
                self._window_k[layer_idx], self._head, self.window_size,
                k_new[:, :, n_to_sink:, :],
            )
            write_range(
                self._window_v[layer_idx], self._head, self.window_size,
                v_new[:, :, n_to_sink:, :],
            )
            old_window = self._window_count
            evicted = max(0, old_window + n_to_window - self.window_size)
            self._window_count = min(self.window_size, old_window + n_to_window)
            self._head = (self._head + n_to_window) % self.window_size
            if evicted > 0:
                self._metrics.record_eviction(evicted)

        self._metrics.record_append(n_new)

    def _append_h2o(self, layer_idx, k_new, v_new, n_new):
        """H2O: за один append выбираем n_evict РАЗНЫХ позиций через argsort.

        Это гарантирует, что один и тот же слот не вытесняется дважды
        в рамках одного append. Score вытесненного слота сбрасывается в 0
        (новый токен ещё никто не смотрел).
        """
        if n_new > self.window_size:
            raise WindowOverflowError(
                f"h2o: append(n={n_new}) > window_size={self.window_size}"
            )

        space = self.window_size - self._window_count
        n_fill = min(n_new, space)
        n_evict = n_new - n_fill

        if n_fill > 0:
            w0 = self._window_count
            w1 = w0 + n_fill
            self._window_k[layer_idx, :, :, w0:w1, :] = k_new[:, :, :n_fill, :]
            self._window_v[layer_idx, :, :, w0:w1, :] = v_new[:, :, :n_fill, :]
            self._window_count = w1

        if n_evict > 0:
            scores = self._accumulated_scores[
                layer_idx, :, :, :self._window_count
            ]  # (batch, kv_heads, window_count)
            order = np.argsort(scores, axis=-1)  # (batch, kv_heads, W)
            evict_indices = order[:, :, :n_evict]  # (batch, kv_heads, n_evict)

            src_k = k_new[:, :, n_fill:n_fill + n_evict, :]
            src_v = v_new[:, :, n_fill:n_fill + n_evict, :]

            for b in range(self.batch_size):
                for h in range(self.num_kv_heads):
                    for k in range(n_evict):
                        pos = int(evict_indices[b, h, k])
                        self._window_k[layer_idx, b, h, pos, :] = src_k[b, h, k, :]
                        self._window_v[layer_idx, b, h, pos, :] = src_v[b, h, k, :]
                        self._accumulated_scores[layer_idx, b, h, pos] = 0.0

            self._metrics.record_eviction(n_evict)

        self._metrics.record_append(n_new)

    def advance(self, n=1):
        with self._lock:
            if n < 0:
                raise CacheStateError(f"advance({n}) с отрицательным n")
            if n == 0:
                return
            self._absolute_position += n
            if self._sink_count >= self.sink_size and self.sink_size > 0:
                self._metrics.record_sink_filled(self._absolute_position)
            if self._window_count >= self.window_size:
                self._metrics.record_window_full(self._absolute_position)

    def reset(self):
        with self._lock:
            self._sink_count = 0
            self._window_count = 0
            self._head = 0
            self._absolute_position = 0
            if self._accumulated_scores is not None:
                self._accumulated_scores.fill(0.0)
            self._metrics.record_reset()

    def record_attention(self, layer_idx, scores):
        with self._lock:
            if self.eviction != "h2o":
                return
            if not (0 <= layer_idx < self.num_layers):
                return
            cur = self._sink_count + self._window_count
            if scores.shape[-1] != cur:
                return
            if self._window_count == 0:
                return
            window_scores = scores[..., self._sink_count:]
            if window_scores.shape[-1] != self._window_count:
                return
            self._accumulated_scores[layer_idx, :, :, :self._window_count] += window_scores

    def get(self, layer_idx):
        with self._lock:
            if not (0 <= layer_idx < self.num_layers):
                raise CacheStateError(f"layer_idx {layer_idx} вне [0, {self.num_layers})")
            empty = np.zeros(
                (self.batch_size, self.num_kv_heads, 0, self.head_dim),
                dtype=self.dtype,
            )
            parts_k, parts_v = [], []
            if self._sink_count > 0 and self._sink_k is not None:
                parts_k.append(self._sink_k[layer_idx, :, :, :self._sink_count, :])
                parts_v.append(self._sink_v[layer_idx, :, :, :self._sink_count, :])
            if self._window_count > 0:
                if self._use_ring:
                    start = (self._head - self._window_count) % self.window_size
                    parts_k.append(self._window_k[layer_idx, :, :, start:start + self._window_count, :])
                    parts_v.append(self._window_v[layer_idx, :, :, start:start + self._window_count, :])
                else:
                    parts_k.append(self._window_k[layer_idx, :, :, :self._window_count, :])
                    parts_v.append(self._window_v[layer_idx, :, :, :self._window_count, :])
            if not parts_k:
                return empty, empty
            if len(parts_k) == 1:
                return parts_k[0], parts_v[0]
            return (
                np.concatenate(parts_k, axis=-2),
                np.concatenate(parts_v, axis=-2),
            )

    def get_parts(self, layer_idx):
        with self._lock:
            if not (0 <= layer_idx < self.num_layers):
                raise CacheStateError(f"layer_idx {layer_idx} вне [0, {self.num_layers})")
            empty = np.zeros(
                (self.batch_size, self.num_kv_heads, 0, self.head_dim), dtype=self.dtype
            )
            if self._sink_count > 0 and self._sink_k is not None:
                sink_k = self._sink_k[layer_idx, :, :, :self._sink_count, :]
                sink_v = self._sink_v[layer_idx, :, :, :self._sink_count, :]
            else:
                sink_k = empty
                sink_v = empty
            if self._window_count > 0:
                if self._use_ring:
                    start = (self._head - self._window_count) % self.window_size
                    window_k = self._window_k[layer_idx, :, :, start:start + self._window_count, :]
                    window_v = self._window_v[layer_idx, :, :, start:start + self._window_count, :]
                else:
                    window_k = self._window_k[layer_idx, :, :, :self._window_count, :]
                    window_v = self._window_v[layer_idx, :, :, :self._window_count, :]
            else:
                window_k = empty
                window_v = empty
            return sink_k, sink_v, window_k, window_v

    def memory_usage_bytes(self):
        with self._lock:
            total = self._window_k.nbytes + self._window_v.nbytes
            if self._sink_k is not None:
                total += self._sink_k.nbytes + self._sink_v.nbytes
            if self._accumulated_scores is not None:
                total += self._accumulated_scores.nbytes
            return total

    def memory_usage_human(self):
        return _human(self.memory_usage_bytes())

    def metrics(self):
        with self._lock:
            snap = self._metrics.snapshot()
            virtual_per_layer = self.batch_size * self.num_kv_heads * self.max_seq_len * self.head_dim
            virtual = 2 * virtual_per_layer * self.num_layers * np.dtype(self.dtype).itemsize
            physical = self.memory_usage_bytes()
            snap.update({
                "absolute_position": self._absolute_position,
                "cur_len": self._sink_count + self._window_count,
                "sink_count": self._sink_count,
                "window_count": self._window_count,
                "sink_size": self.sink_size,
                "window_size": self.window_size,
                "eviction": self.eviction,
                "physical_bytes": physical,
                "physical_human": self.memory_usage_human(),
                "virtual_bytes": virtual,
                "compression_ratio": (physical / virtual) if virtual > 0 else 1.0,
            })
            return snap

    def __repr__(self):
        return (
            f"<NsofiKVCache layers={self.num_layers} kv_heads={self.num_kv_heads} "
            f"head_dim={self.head_dim} batch={self.batch_size} "
            f"sink={self._sink_count}/{self.sink_size} "
            f"window={self._window_count}/{self.window_size} "
            f"abs_pos={self._absolute_position} evict={self.eviction}>"
        )


__all__ = ["NsofiKVCache"]