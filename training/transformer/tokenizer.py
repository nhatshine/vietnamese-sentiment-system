"""Vietnamese whitespace/syllable tokenizer built entirely from TRAIN data."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
from preprocess import clean_text

SPECIAL_TOKENS = ("<pad>", "<unk>", "<cls>")
PAD_ID, UNK_ID, CLS_ID = 0, 1, 2


def tokenize(text: str) -> list[str]:
    return clean_text(text).split()


def build_vocab(train_texts, min_freq: int = 2, max_vocab_size: int = 25000) -> dict[str, int]:
    if min_freq < 1 or max_vocab_size < len(SPECIAL_TOKENS):
        raise ValueError("min_freq / max_vocab_size không hợp lệ")
    counts = Counter(token for text in train_texts for token in tokenize(text))
    # Frequency descending and alphabetical tiebreaker -> deterministic vocabulary.
    items = sorted(((t, n) for t, n in counts.items() if n >= min_freq and t not in SPECIAL_TOKENS),
                   key=lambda pair: (-pair[1], pair[0]))
    vocab = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
    for token, _ in items[:max_vocab_size - len(SPECIAL_TOKENS)]:
        vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int):
    if max_len < 2:
        raise ValueError("max_len phải >= 2")
    ids = [CLS_ID] + [vocab.get(tok, UNK_ID) for tok in tokenize(text)[:max_len - 1]]
    mask = [1] * len(ids)
    n_pad = max_len - len(ids)
    ids.extend([PAD_ID] * n_pad)
    mask.extend([0] * n_pad)
    return ids, mask


def save_vocab(vocab: dict[str, int], path: str | Path):
    Path(path).write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")


def load_vocab(path: str | Path) -> dict[str, int]:
    vocab = json.loads(Path(path).read_text(encoding="utf-8"))
    for i, t in enumerate(SPECIAL_TOKENS):
        if vocab.get(t) != i:
            raise ValueError(f"Special token {t} must map to {i}")
    return vocab
