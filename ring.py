"""Ring buffer helpers для sliding window в nsofi40.

Физический буфер в 2 раза больше логического окна. Каждая запись
в позицию p дублируется в p + W. Тогда окно читается как непрерывный
view без np.roll и без копирований.
"""
import numpy as np


def next_head(head, window_size):
    """Возвращает следующую позицию записи после одного append."""
    return (head + 1) % window_size


def write_range(buf, head, window_size, data):
    """Записывает `data` (..., n, ...) в буфер начиная с `head`, с дублированием.

    Параметры:
        buf: (..., 2 * window_size, ...) — физический двойной буфер.
        head: позиция записи в [0, window_size).
        window_size: W.
        data: (..., n, ...), n <= window_size.

    Возвращает:
        новый head (в [0, window_size)).
    """
    n = data.shape[-2]
    if n > window_size:
        raise ValueError(f"write_range: n={n} > window_size={window_size}")
    if n == 0:
        return head

    first = min(n, window_size - head)
    buf[..., head:head + first, :] = data[..., :first, :]
    buf[..., head + window_size:head + window_size + first, :] = data[..., :first, :]

    if n > first:
        rest = n - first
        buf[..., 0:rest, :] = data[..., first:, :]
        buf[..., window_size:window_size + rest, :] = data[..., first:, :]

    return (head + n) % window_size


def read_window(buf, head, window_size, count):
    """Возвращает view последних `count` записей (от старых к новым).

    Параметры:
        buf: (..., 2 * window_size, ...).
        head: позиция следующей записи в [0, window_size).
        window_size: W.
        count: сколько записей прочитать, 0 <= count <= W.

    Формула: start = (head - count) % W; читаем buf[start : start + count].
    """
    if count == 0:
        return buf[..., head:head, :]
    start = (head - count) % window_size
    return buf[..., start:start + count, :]


__all__ = ["next_head", "write_range", "read_window"]