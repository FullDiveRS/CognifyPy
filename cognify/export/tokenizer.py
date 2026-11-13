"""Hugging Face tokenizer.json exporter (WordPiece), built from the mind's
own corpus — knowledge-base facts, memories, name — plus the companion
``vocab.json`` and ``special_tokens_map.json``.

The result loads with the ``tokenizers`` library:
``tok = tokenizers.Tokenizer.from_file("tokenizer.json")``.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..core.mind import Mind

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]


def _tokenize(text: str) -> List[str]:
    """Lowercased alphanumeric word tokens."""
    tokens, current = [], []
    for ch in (text or "").lower():
        if ch.isalnum():
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def collect_corpus(mind: "Mind") -> List[str]:
    """The mind's text corpus: name, facts, and memory contents."""
    corpus: List[str] = [mind.name]
    for fact in mind.kb.facts():
        corpus.append(f"{fact.subject} {fact.predicate} {fact.object}")
    for record in mind.memory.all():
        corpus.append(record.content)
    return [line for line in corpus if line and line.strip()]


def build_vocab(corpus: Iterable[str], max_size: int) -> List[str]:
    """Frequency-ranked word list, special tokens first (ids 0..4)."""
    counts: Counter = Counter()
    for line in corpus:
        counts.update(_tokenize(line))
    vocab = list(SPECIAL_TOKENS)
    for word, _count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if len(vocab) >= max_size:
            break
        if word not in vocab:
            vocab.append(word)
    return vocab


def write(directory: str, vocab: List[str], mind_name: str) -> int:
    """Writes tokenizer.json + vocab.json + special_tokens_map.json.

    Returns the vocabulary size.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    added = [{
        "id": i, "content": tok, "single_word": False, "lstrip": False,
        "rstrip": False, "normalized": False, "special": True,
    } for i, tok in enumerate(SPECIAL_TOKENS)]

    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordPiece",
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": {token: i for i, token in enumerate(vocab)},
        },
    }
    (out / "tokenizer.json").write_text(
        json.dumps(tokenizer, ensure_ascii=False, indent=2), encoding="utf-8")

    (out / "vocab.json").write_text(
        json.dumps(tokenizer["model"]["vocab"], ensure_ascii=False, indent=2),
        encoding="utf-8")

    (out / "special_tokens_map.json").write_text(json.dumps({
        "pad_token": "[PAD]", "unk_token": "[UNK]", "bos_token": "[BOS]",
        "eos_token": "[EOS]", "mask_token": "[MASK]",
    }, indent=2), encoding="utf-8")
    return len(vocab)
