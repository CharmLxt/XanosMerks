"""Автогрессивная генерация: один промпт, батч с паддингом, MLA, stop, overflow."""
import numpy as np
from ninewas9 import KVCache, RoPECache
from .forward import TransformerModel
from .mla import MLACompressedCache
from .sampling import sample, sample_batch
from .tokenizer import BPETokenizer, _make_gpt2_byte_encoder


def _cache_position(cache):
    return getattr(cache, "absolute_position", cache.cur_len)


class Generator:
    def __init__(self, config, weights, tokenizer):
        self.config = config
        self.max_seq = config.get("max_position_embeddings", 2048)
        is_mla = config.get("attention_type") == "mla"

        if is_mla:
            self.kv_cache = MLACompressedCache(
                num_layers=config["num_layers"],
                kv_lora_rank=config["kv_lora_rank"],
                qk_rope_dim=config["qk_rope_head_dim"],
                max_seq_len=self.max_seq,
            )
            rope_dim = config["qk_rope_head_dim"]
        else:
            self.kv_cache = KVCache(
                num_layers=config["num_layers"],
                num_kv_heads=config["num_kv_heads"],
                head_dim=config["head_dim"],
                max_seq_len=self.max_seq,
            )
            rope_dim = config["head_dim"]

        self.rope_cache = RoPECache(
            head_dim=rope_dim,
            max_seq_len=self.max_seq,
            rope_theta=config.get("rope_theta", 10000.0),
            rope_scaling=config.get("rope_scaling"),
        )
        self.model = TransformerModel(
            config, weights, self.kv_cache, self.rope_cache,
        )

        vocab = tokenizer.get("vocab", {})
        if not vocab:
            raise ValueError("Токенизатор пуст: нет поля 'vocab'")

        byte_encoder = _make_gpt2_byte_encoder()
        is_byte_level = all(
            byte_encoder[b] in vocab for b in (32, 101, 116)
        ) and bool(tokenizer.get("merges"))

        if is_byte_level:
            self.tokenizer_obj = BPETokenizer(
                vocab=vocab,
                merges=tokenizer.get("merges", []),
                bos_token=tokenizer.get("bos_token", "<s>"),
                eos_token=tokenizer.get("eos_token", "</s>"),
                unk_token=tokenizer.get("unk_token", "<unk>"),
                pad_token=tokenizer.get("pad_token"),
            )
        else:
            self.tokenizer_obj = None

        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.unk_id = vocab.get(tokenizer.get("unk_token", "<unk>"), 0)
        self.eos_id = vocab.get(tokenizer.get("eos_token", "</s>"))
        pad_token = tokenizer.get("pad_token")
        self.pad_id = vocab.get(pad_token) if pad_token else None

    def set_kv_cache(self, cache):
        self.kv_cache = cache
        self.model.kv_cache = cache
        for layer in self.model.layers:
            layer.attn.kv_cache = cache

    def _recreate_cache(self, batch_size):
        old = self.kv_cache
        is_nsofi = hasattr(old, "window_size") and hasattr(old, "sink_size")
        is_mla = hasattr(old, "kv_lora_rank")

        if is_nsofi and is_mla:
            from xanosmerks.nsofi40 import NsofiMLACache
            new = NsofiMLACache(
                num_layers=old.num_layers,
                kv_lora_rank=old.kv_lora_rank,
                qk_rope_dim=old.qk_rope_dim,
                max_seq_len=old.max_seq_len,
                window_size=old.window_size,
                sink_size=old.sink_size,
                eviction=old.eviction,
                batch_size=batch_size,
                dtype=old.dtype,
            )
        elif is_nsofi:
            from xanosmerks.nsofi40 import NsofiKVCache
            new = NsofiKVCache(
                num_layers=old.num_layers,
                num_kv_heads=old.num_kv_heads,
                head_dim=old.head_dim,
                max_seq_len=old.max_seq_len,
                window_size=old.window_size,
                sink_size=old.sink_size,
                eviction=old.eviction,
                batch_size=batch_size,
                dtype=old.dtype,
            )
        elif is_mla:
            new = MLACompressedCache(
                num_layers=old.num_layers,
                kv_lora_rank=old.kv_lora_rank,
                qk_rope_dim=old.qk_rope_dim,
                max_seq_len=old.max_seq_len,
                batch_size=batch_size,
                dtype=old.dtype,
            )
        else:
            new = KVCache(
                num_layers=old.num_layers,
                num_kv_heads=old.num_kv_heads,
                head_dim=old.head_dim,
                max_seq_len=old.max_seq_len,
                batch_size=batch_size,
                dtype=old.dtype,
            )
        self.set_kv_cache(new)

    def encode(self, text):
        if self.tokenizer_obj is not None:
            ids = self.tokenizer_obj.encode(text, add_special_tokens=False)
            if not ids:
                raise ValueError(f"Токенизатор вернул пустой список для {text!r}.")
            return ids
        tokens = [self.vocab.get(word, self.unk_id) for word in text.split()]
        if not tokens:
            raise ValueError(f"Не удалось закодировать: {text!r}")
        return tokens

    def decode(self, token_ids):
        if self.tokenizer_obj is not None:
            return self.tokenizer_obj.decode(token_ids, skip_special_tokens=True)
        return " ".join(self.inv_vocab.get(t, "<unk>") for t in token_ids)

    def _decode_token(self, token_id):
        if self.tokenizer_obj is not None:
            return self.tokenizer_obj.decode([token_id], skip_special_tokens=True)
        return self.inv_vocab.get(token_id, "")

    def _check_nsofi_capacity(self, prompt_ids):
        window_size = getattr(self.kv_cache, "window_size", None)
        if window_size is None:
            return
        sink_size = getattr(self.kv_cache, "sink_size", 0)
        capacity = sink_size + window_size
        if len(prompt_ids) > capacity:
            raise ValueError(
                f"Prompt ({len(prompt_ids)}) > nsofi40 capacity={capacity}."
            )

    def generate(self, prompt, max_new_tokens=50,
                 temperature=0.7, top_k=40, top_p=0.9,
                 min_p=None, typical=None,
                 repetition_penalty=1.0,
                 stop=None, truncate="start",
                 on_token=None, verbose=False):
        stop = [s for s in (stop or []) if s]
        
        # Если кэш настроен на батч (например, после generate_batch), 
        # пересоздаем его под размер батча 1.
        cache_bs = getattr(self.kv_cache, "batch_size", 1)
        if cache_bs != 1:
            self._recreate_cache(1)
            
        self.kv_cache.reset()

        prompt_ids = self.encode(prompt)
        if not prompt_ids:
            raise ValueError(f"encode({prompt!r}) вернул пустой список.")

        self._check_nsofi_capacity(prompt_ids)

        if verbose:
            print(f"[encode] {prompt!r} -> {len(prompt_ids)} tokens")

        if len(prompt_ids) + max_new_tokens + 1 > self.max_seq:
            overflow = len(prompt_ids) + max_new_tokens + 1 - self.max_seq
            if truncate == "none":
                raise ValueError(f"Промпт + max_new + 1 > max_seq ({self.max_seq})")
            if truncate == "start":
                prompt_ids = prompt_ids[overflow:]
            elif truncate == "middle":
                half = overflow // 2
                keep_left = max(len(prompt_ids) - overflow - half, 0)
                left = prompt_ids[:half] if half > 0 else []
                right = prompt_ids[len(prompt_ids) - half:] if keep_left > 0 else []
                prompt_ids = list(left) + list(right)
            if len(prompt_ids) + 1 > self.max_seq:
                prompt_ids = prompt_ids[-(self.max_seq - 1):]
            max_new_tokens = min(max_new_tokens, self.max_seq - len(prompt_ids) - 1)
            if max_new_tokens < 1:
                raise ValueError("После truncation не осталось места.")

        input_ids = np.array([prompt_ids], dtype=np.int32)
        logits = self.model(input_ids, position_offset=0, last_only=True)
        self.kv_cache.advance(len(prompt_ids))

        generated_ids = []
        next_token = sample(
            logits[0], temperature=temperature, top_k=top_k, top_p=top_p,
            min_p=min_p, typical=typical,
            repetition_penalty=repetition_penalty,
            past_tokens=prompt_ids + generated_ids,
        )
        generated_ids.append(next_token)

        if on_token is not None:
            on_token(self._decode_token(next_token), next_token)

        if any(s in self.decode(generated_ids) for s in stop):
            return prompt + self.decode(generated_ids)

        for step in range(2, max_new_tokens + 1):
            if self.eos_id is not None and next_token == self.eos_id:
                break
            input_ids = np.array([[next_token]], dtype=np.int32)
            logits = self.model(
                input_ids,
                position_offset=_cache_position(self.kv_cache),
                last_only=True,
            )
            self.kv_cache.advance(1)
            next_token = sample(
                logits[0], temperature=temperature, top_k=top_k, top_p=top_p,
                min_p=min_p, typical=typical,
                repetition_penalty=repetition_penalty,
                past_tokens=prompt_ids + generated_ids,
            )
            generated_ids.append(next_token)
            if on_token is not None:
                on_token(self._decode_token(next_token), next_token)
            if any(s in self.decode(generated_ids) for s in stop):
                return prompt + self.decode(generated_ids)

        return prompt + self.decode(generated_ids)

    def generate_batch(self, prompts, max_new_tokens=50,
                       temperature=0.7, top_k=40, top_p=0.9,
                       min_p=None, typical=None,
                       repetition_penalty=1.0,
                       stop=None, on_token=None, verbose=False,
                       pad_token_id=None):
        """Параллельная генерация с поддержкой разной длины промптов (padding)."""
        if not prompts:
            return []

        stop = [s for s in (stop or []) if s]
        batch_size = len(prompts)

        cache_bs = getattr(self.kv_cache, "batch_size", 1)
        if cache_bs != batch_size:
            if cache_bs == 1:
                self._recreate_cache(batch_size)
            else:
                raise ValueError(
                    f"batch={batch_size} != kv_cache.batch_size={cache_bs}"
                )

        self.kv_cache.reset()

        prompts_ids = [self.encode(p) for p in prompts]
        lengths = [len(p) for p in prompts_ids]
        if any(l == 0 for l in lengths):
            raise ValueError("Пустые промпты после encode.")

        if pad_token_id is None:
            pad_token_id = self.pad_id if self.pad_id is not None else self.unk_id

        max_len = max(lengths)
        is_ragged = any(l != max_len for l in lengths)

        if is_ragged and hasattr(self.kv_cache, "window_size"):
            raise ValueError(
                "nsofi40 не поддерживает padding (разная длина промптов). "
                "Сделайте промпты одинаковыми или используйте KVCache/MLACompressedCache."
            )

        if max_len + max_new_tokens + 1 > self.max_seq:
            raise ValueError(
                f"max_len ({max_len}) + max_new ({max_new_tokens}) + 1 > "
                f"max_seq ({self.max_seq})."
            )

        if is_ragged:
            input_ids_arr = np.full(
                (batch_size, max_len), pad_token_id, dtype=np.int32
            )
            attention_mask = np.zeros((batch_size, max_len), dtype=np.int32)
            for b, ids in enumerate(prompts_ids):
                input_ids_arr[b, :len(ids)] = ids
                attention_mask[b, :len(ids)] = 1
        else:
            input_ids_arr = np.array(prompts_ids, dtype=np.int32)
            attention_mask = None

        if verbose:
            print(f"[batch] {batch_size} prompts, max_len={max_len}, ragged={is_ragged}")

        # Prefill
        if is_ragged:
            logits_full = self.model(
                input_ids_arr, position_offset=0, last_only=False,
                kv_attention_mask=attention_mask,
            )
            last_idx = np.array(
                [int(np.sum(attention_mask[b])) - 1 for b in range(batch_size)]
            )
            logits = logits_full[np.arange(batch_size), last_idx, :]
        else:
            logits = self.model(
                input_ids_arr, position_offset=0, last_only=True,
            )

        self.kv_cache.advance(max_len)

        generated_ids = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        next_tokens = sample_batch(
            logits,
            temperature=temperature, top_k=top_k, top_p=top_p,
            min_p=min_p, typical=typical,
            repetition_penalty=repetition_penalty,
            past_tokens_list=prompts_ids,
        )

        for i in range(batch_size):
            tok = int(next_tokens[i])
            generated_ids[i].append(tok)
            if self.eos_id is not None and tok == self.eos_id:
                finished[i] = True
            if on_token is not None:
                on_token(self._decode_token(tok), tok, i)
            if stop and any(s in self.decode(generated_ids[i]) for s in stop):
                finished[i] = True

        for step in range(2, max_new_tokens + 1):
            if all(finished):
                break

            input_ids = np.array(
                [[int(next_tokens[i])] for i in range(batch_size)],
                dtype=np.int32,
            )

            current_kv_mask = None
            if is_ragged:
                cur_len = self.kv_cache.cur_len
                mask_len = cur_len + 1
                current_kv_mask = np.ones((batch_size, mask_len), dtype=np.int32)
                current_kv_mask[:, :max_len] = attention_mask

            logits = self.model(
                input_ids,
                position_offset=_cache_position(self.kv_cache),
                last_only=True,
                kv_attention_mask=current_kv_mask,
            )
            self.kv_cache.advance(1)

            past_for_sample = [
                prompts_ids[i] + generated_ids[i] for i in range(batch_size)
            ]
            new_tokens = sample_batch(
                logits,
                temperature=temperature, top_k=top_k, top_p=top_p,
                min_p=min_p, typical=typical,
                repetition_penalty=repetition_penalty,
                past_tokens_list=past_for_sample,
            )

            for i in range(batch_size):
                if finished[i]:
                    continue
                tok = int(new_tokens[i])
                generated_ids[i].append(tok)
                if self.eos_id is not None and tok == self.eos_id:
                    finished[i] = True
                    continue
                if on_token is not None:
                    on_token(self._decode_token(tok), tok, i)
                if stop and any(s in self.decode(generated_ids[i]) for s in stop):
                    finished[i] = True
            next_tokens = new_tokens

        return [prompts[i] + self.decode(generated_ids[i]) for i in range(batch_size)]


__all__ = ["Generator"]