"""Стратегии сэмплинга токенов: greedy, temperature, top-k, top-p, min-p, typical."""
import numpy as np
from ninewas9 import fast_softmax, top_k_filter, top_p_filter


def apply_repetition_penalty(logits, past_tokens, penalty=1.1):
    if penalty == 1.0 or not past_tokens:
        return logits
    logits = logits.copy()
    for tok in set(past_tokens):
        if 0 <= tok < len(logits):
            if logits[tok] > 0:
                logits[tok] /= penalty
            else:
                logits[tok] *= penalty
    return logits


def min_p_filter(logits, min_p=0.0):
    if min_p <= 0:
        return logits
    probs = fast_softmax(logits, axis=-1)
    max_prob = np.max(probs, axis=-1, keepdims=True)
    threshold = min_p * max_prob
    return np.where(probs >= threshold, logits, -np.inf)


def typical_filter(logits, mass=0.9):
    if mass <= 0 or mass >= 1.0:
        return logits
    logits_arr = np.asarray(logits, dtype=np.float32)
    probs = fast_softmax(logits_arr, axis=-1)
    log_probs = np.log(probs + 1e-12)
    entropy = -np.sum(probs * log_probs, axis=-1, keepdims=True)
    shifted = log_probs + entropy
    typicality = np.abs(shifted)

    sorted_indices = np.argsort(typicality, axis=-1)
    sorted_probs = np.take_along_axis(probs, sorted_indices, axis=-1)
    cumsum = np.cumsum(sorted_probs, axis=-1)

    mask_sorted = (cumsum - sorted_probs) >= mass
    sorted_logits = np.take_along_axis(logits_arr, sorted_indices, axis=-1)
    sorted_logits = np.where(mask_sorted, -np.inf, sorted_logits)

    result = np.empty_like(logits_arr)
    np.put_along_axis(result, sorted_indices, sorted_logits, axis=-1)
    return result


def sample_greedy(logits):
    return int(np.argmax(logits))


def sample_temperature(logits, temperature=1.0):
    if temperature <= 0:
        return sample_greedy(logits)
    probs = fast_softmax(logits / temperature)
    return int(np.random.choice(len(probs), p=probs))


def sample_top_k(logits, k=40, temperature=1.0):
    if temperature <= 0:
        return sample_greedy(logits)
    filtered = top_k_filter(logits / temperature, k)
    probs = fast_softmax(filtered)
    return int(np.random.choice(len(probs), p=probs))


def sample_top_p(logits, p=0.9, temperature=1.0):
    if temperature <= 0:
        return sample_greedy(logits)
    filtered = top_p_filter(logits / temperature, p)
    probs = fast_softmax(filtered)
    return int(np.random.choice(len(probs), p=probs))


def sample_min_p(logits, min_p=0.05, temperature=1.0):
    if temperature <= 0:
        return sample_greedy(logits)
    filtered = min_p_filter(logits / temperature, min_p)
    probs = fast_softmax(filtered)
    return int(np.random.choice(len(probs), p=probs))


def sample_typical(logits, mass=0.9, temperature=1.0):
    if temperature <= 0:
        return sample_greedy(logits)
    filtered = typical_filter(logits / temperature, mass)
    probs = fast_softmax(filtered)
    return int(np.random.choice(len(probs), p=probs))


def sample(logits, temperature=1.0, top_k=None, top_p=None,
           min_p=None, typical=None,
           repetition_penalty=1.0, past_tokens=None):
    logits = np.asarray(logits, dtype=np.float32)

    if repetition_penalty != 1.0 and past_tokens:
        logits = apply_repetition_penalty(logits, past_tokens, repetition_penalty)

    if temperature <= 0:
        return int(np.argmax(logits))

    scaled = logits / temperature

    if top_k is not None and top_k > 0:
        scaled = top_k_filter(scaled, top_k)

    if top_p is not None and top_p < 1.0:
        scaled = top_p_filter(scaled, top_p)

    if min_p is not None and min_p > 0:
        scaled = min_p_filter(scaled, min_p)

    if typical is not None and 0 < typical < 1.0:
        scaled = typical_filter(scaled, typical)

    probs = fast_softmax(scaled)
    if not np.isfinite(probs).all() or probs.sum() <= 0:
        return int(np.argmax(logits))

    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def sample_batch(logits, temperature=1.0, top_k=None, top_p=None,
                 min_p=None, typical=None,
                 repetition_penalty=1.0, past_tokens_list=None):
    """Сэмплинг для батча.

    logits: (batch, vocab)
    past_tokens_list: список списков (по одному на batch-элемент) или None

    Возвращает: np.ndarray формы (batch,), dtype int64.
    """
    logits_arr = np.asarray(logits)
    if logits_arr.ndim != 2:
        raise ValueError(
            f"sample_batch: ожидалось 2D logits, получено {logits_arr.ndim}D"
        )
    batch = logits_arr.shape[0]
    result = np.empty(batch, dtype=np.int64)
    for i in range(batch):
        past = past_tokens_list[i] if past_tokens_list is not None else None
        result[i] = sample(
            logits_arr[i],
            temperature=temperature, top_k=top_k, top_p=top_p,
            min_p=min_p, typical=typical,
            repetition_penalty=repetition_penalty, past_tokens=past,
        )
    return result


__all__ = [
    "sample", "sample_batch", "sample_greedy", "sample_temperature",
    "sample_top_k", "sample_top_p", "sample_min_p", "sample_typical",
    "min_p_filter", "typical_filter", "apply_repetition_penalty",
]