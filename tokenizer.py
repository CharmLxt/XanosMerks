"""BPE-токенизатор с byte-level encoding (совместим с GPT-2/Llama-стилем)."""
import re
import json
from pathlib import Path


def _make_gpt2_byte_encoder():
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


# GPT-2-style pre-tokenization:
#   - апостроф-суффиксы ('s, 't, 're, 've, 'm, 'll, 'd)
#   - буквы отдельно от цифр ([^\W\d_]+ = Unicode letters, \d+ = digits)
#   - прочее (пунктуация) и пробелы
_PRE_TOKENIZE_RE = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+",
    re.IGNORECASE,
)


class BPETokenizer:
    """Полный byte-level BPE-токенизатор."""

    def __init__(self, vocab, merges, bos_token="<s>", eos_token="</s>",
                 unk_token="<unk>", pad_token=None):
        self.vocab = dict(vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        self.merges = {}
        for rank, m in enumerate(merges):
            if isinstance(m, str):
                parts = m.split()
            else:
                parts = list(m)
            if len(parts) == 2:
                self.merges[tuple(parts)] = rank

        self.bos_id = self.vocab.get(bos_token) if bos_token else None
        self.eos_id = self.vocab.get(eos_token) if eos_token else None
        self.unk_id = self.vocab.get(unk_token, 0) if unk_token else 0
        self.pad_id = self.vocab.get(pad_token) if pad_token else None

        self.byte_encoder = _make_gpt2_byte_encoder()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    def _bytes_to_unicode(self, text):
        utf8 = text.encode("utf-8")
        return "".join(self.byte_encoder[b] for b in utf8)

    def _unicode_to_bytes(self, text):
        out = bytearray()
        for ch in text:
            if ch in self.byte_decoder:
                out.append(self.byte_decoder[ch])
            else:
                out.extend(ch.encode("utf-8", errors="replace"))
        return bytes(out)

    def _bpe(self, word):
        if len(word) < 2:
            return (word,)
        parts = tuple(word)
        while True:
            best_rank = float("inf")
            best_pair = None
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.merges.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            a, b = best_pair
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
                    new_parts.append(a + b)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = tuple(new_parts)

        return parts

    def encode(self, text, add_special_tokens=True):
        if not text:
            if add_special_tokens and self.bos_id is not None:
                return [self.bos_id]
            return []

        chunks = _PRE_TOKENIZE_RE.findall(text)
        all_ids = []
        for chunk in chunks:
            unicode_chunk = self._bytes_to_unicode(chunk)
            bpe_parts = self._bpe(unicode_chunk)
            ids = [self.vocab.get(p, self.unk_id) for p in bpe_parts]
            all_ids.extend(ids)

        if add_special_tokens and self.bos_id is not None:
            all_ids = [self.bos_id] + all_ids

        return all_ids

    def decode(self, ids, skip_special_tokens=True):
        specials = {self.bos_id, self.eos_id, self.pad_id}
        specials.discard(None)

        tokens = []
        for i in ids:
            if skip_special_tokens and i in specials:
                continue
            tokens.append(self.inv_vocab.get(i, ""))

        text = "".join(tokens)
        try:
            return self._unicode_to_bytes(text).decode("utf-8", errors="replace")
        except Exception:
            return text

    def __call__(self, text):
        return self.encode(text)

    def __len__(self):
        return len(self.vocab)


def load_tokenizer_from_json(path):
    path = Path(path)
    data = json.loads(path.read_text("utf-8"))
    model = data.get("model", {})
    vocab = model.get("vocab", {})
    merges = model.get("merges", [])

    if isinstance(vocab, list):
        vocab = {tok: i for i, tok in enumerate(vocab)}

    return BPETokenizer(
        vocab=vocab,
        merges=merges,
        bos_token=data.get("bos_token", "<s>"),
        eos_token=data.get("eos_token", "</s>"),
        unk_token=data.get("unk_token", "<unk>"),
        pad_token=data.get("pad_token"),
    )


__all__ = ["BPETokenizer", "load_tokenizer_from_json"]