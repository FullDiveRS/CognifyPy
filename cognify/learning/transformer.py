"""TransformerLM: a decoder-only transformer language model — the neural
half of the mind's next-token prediction.

The same architecture family as GPT (token + learned positional embeddings,
pre-LayerNorm blocks of multi-head causal self-attention and GELU
feed-forwards, residual stream, weight-tied output head), implemented from
scratch in PURE PYTHON with full backpropagation and Adam — zero
dependencies, mirroring the C# ``Cognify.Learning.TransformerLM``.

Word-level vocabulary (same tokenizer as the n-gram model) and the same
observation buffer: the mind feeds it everything she reads and says, and
training is lazy — automatic once enough new language accumulates, or
explicit via ``Mind.train_deep_model()``.
"""
from __future__ import annotations

import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..util import tokenize as _util_tokenize
from .adam import AdamOptimizer
from .deep_net import gelu, gelu_grad
from .next_token import TokenPrediction

_PAD, _UNK, _BOS, _EOS = "<pad>", "<unk>", "<bos>", "<eos>"
_PAD_ID, _UNK_ID, _BOS_ID, _EOS_ID = 0, 1, 2, 3
_SPECIAL_COUNT = 4

_SENTENCE_SPLIT = re.compile(r"[.!?\;\n\r]+")

AUTO_TRAIN_TOKENS = 800
AUTO_TRAIN_EPOCHS = 3
MAX_BUFFERED_SENTENCES = 1000


@dataclass
class TransformerConfig:
    """Architecture + training knobs.

    The defaults are sized for a PURE-PYTHON CPU budget (the C# mirror
    defaults to dim 32 / 4 heads / 2 layers): still a real causal
    transformer with attention, but small enough that a lazy training pass
    inside a cognitive cycle stays in the seconds range. Pass a bigger
    config when you have the patience — the architecture is identical.
    """

    dim: int = 16
    heads: int = 2
    layers: int = 1
    context_len: int = 12
    ffn_mult: int = 2
    dropout: float = 0.1
    learning_rate: float = 3e-3
    seed: int = 42
    max_vocab: int = 2048
    batch_size: int = 16
    max_windows_per_epoch: int = 96
    grad_clip: float = 1.0

    def validate(self) -> None:
        if self.dim <= 0 or self.heads <= 0 or self.layers <= 0 or self.context_len <= 2:
            raise ValueError("transformer dimensions must be positive (context_len > 2)")
        if self.dim % self.heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by heads ({self.heads})")


@dataclass
class TransformerTrainingReport:
    """Everything one training pass produced (mirrors the C# record)."""

    epochs: int
    windows: int
    initial_loss: float
    final_loss: float
    parameters: int
    vocabulary: int
    duration: float  # seconds

    def __str__(self) -> str:  # pragma: no cover
        return (f"{self.epochs} epoch(s) on {self.windows} window(s): "
                f"loss {self.initial_loss:.3f} -> {self.final_loss:.3f} "
                f"({self.parameters} parameters, {self.vocabulary} words, "
                f"{self.duration * 1000:.0f} ms)")


class _BlockParams:
    """One transformer block's parameters (mirrors the C# class)."""

    __slots__ = ("ln1g", "ln1b", "wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo",
                 "ln2g", "ln2b", "f1", "bf1", "f2", "bf2")

    def __init__(self, d: int, f: int) -> None:
        self.ln1g = [1.0] * d
        self.ln1b = [0.0] * d
        self.wq = [[0.0] * d for _ in range(d)]
        self.bq = [0.0] * d
        self.wk = [[0.0] * d for _ in range(d)]
        self.bk = [0.0] * d
        self.wv = [[0.0] * d for _ in range(d)]
        self.bv = [0.0] * d
        self.wo = [[0.0] * d for _ in range(d)]
        self.bo = [0.0] * d
        self.ln2g = [1.0] * d
        self.ln2b = [0.0] * d
        self.f1 = [[0.0] * d for _ in range(f)]
        self.bf1 = [0.0] * f
        self.f2 = [[0.0] * f for _ in range(d)]
        self.bf2 = [0.0] * d


class TransformerLM:
    """Decoder-only transformer LM: tiny enough for a cognitive cycle,
    real enough to generalize beyond the n-gram's exact tables."""

    def __init__(self, config: Optional[TransformerConfig] = None) -> None:
        self.cfg = config or TransformerConfig()
        self.cfg.validate()
        self._rng = random.Random(self.cfg.seed)
        self._adam = AdamOptimizer()

        self._vocab: List[str] = [_PAD, _UNK, _BOS, _EOS]
        self._index: Dict[str, int] = {t: i for i, t in enumerate(self._vocab)}
        self._buffer: List[str] = []
        self._tokens_observed = 0
        self._tokens_since_train = 0
        self._epochs_trained = 0
        self._last_loss = 0.0

        d, t, f = self.cfg.dim, self.cfg.context_len, self.cfg.dim * self.cfg.ffn_mult
        self._wte: List[List[float]] = []
        self._wpe = self._gauss_matrix(t, d)
        self._blocks = [_BlockParams(d, f) for _ in range(self.cfg.layers)]
        self._lnfg = [1.0] * d
        self._lnfb = [0.0] * d
        self._init_vocab_embeddings()

    # ------------------------------------------------------------------ init
    def _gauss_matrix(self, rows: int, cols: int, std: float = 0.02) -> List[List[float]]:
        rng = self._rng
        return [[rng.gauss(0.0, std) for _ in range(cols)] for _ in range(rows)]

    def _init_vocab_embeddings(self) -> None:
        # Keep already-learned rows when the vocabulary grows.
        d = self.cfg.dim
        while len(self._wte) < len(self._vocab):
            self._wte.append([self._rng.gauss(0.0, 0.02) for _ in range(d)])

    # ------------------------------------------------------------------ stats
    @property
    def is_trained(self) -> bool:
        return self._epochs_trained > 0

    @property
    def vocabulary_size(self) -> int:
        return len(self._vocab)

    def export_vocab(self) -> List[str]:
        """Vocabulary strings in id order (specials first: <pad>, <unk>, <bos>,
        <eos>) — used by the export pipeline to write tokenizer metadata."""
        return list(self._vocab)

    def export_tensors(self) -> List[Tuple[str, Optional[List[List[float]]], Optional[List[float]]]]:
        """Named weight tensors for the export pipeline (safetensors/GGUF):
        the token/positional embeddings, every block's projections and
        LayerNorm parameters, and the final LayerNorm — mirrors the C#
        ``ExportTensors``. Each entry is ``(name, matrix_or_None,
        vector_or_None)``; matrices are row-major ``[out][in]``."""
        out: List[Tuple[str, Optional[List[List[float]]], Optional[List[float]]]] = [
            ("cognify.transformer.wte", [row[:] for row in self._wte], None),
            ("cognify.transformer.wpe", [row[:] for row in self._wpe], None),
        ]
        for li, b in enumerate(self._blocks):
            p = f"cognify.transformer.l{li}."
            out += [
                (p + "attn.wq", [row[:] for row in b.wq], None),
                (p + "attn.wk", [row[:] for row in b.wk], None),
                (p + "attn.wv", [row[:] for row in b.wv], None),
                (p + "attn.wo", [row[:] for row in b.wo], None),
                (p + "attn.bq", None, list(b.bq)),
                (p + "attn.bk", None, list(b.bk)),
                (p + "attn.bv", None, list(b.bv)),
                (p + "attn.bo", None, list(b.bo)),
                (p + "ln1.g", None, list(b.ln1g)),
                (p + "ln1.b", None, list(b.ln1b)),
                (p + "ln2.g", None, list(b.ln2g)),
                (p + "ln2.b", None, list(b.ln2b)),
                (p + "ffn.w1", [row[:] for row in b.f1], None),
                (p + "ffn.w2", [row[:] for row in b.f2], None),
                (p + "ffn.b1", None, list(b.bf1)),
                (p + "ffn.b2", None, list(b.bf2)),
            ]
        out += [
            ("cognify.transformer.lnf.g", None, list(self._lnfg)),
            ("cognify.transformer.lnf.b", None, list(self._lnfb)),
        ]
        return out

    @property
    def tokens_observed(self) -> int:
        return self._tokens_observed

    @property
    def buffered_sentences(self) -> int:
        return len(self._buffer)

    @property
    def epochs_trained(self) -> int:
        return self._epochs_trained

    @property
    def last_loss(self) -> float:
        return self._last_loss

    @property
    def n_params(self) -> int:
        d, t = self.cfg.dim, self.cfg.context_len
        f = d * self.cfg.ffn_mult
        total = len(self._wte) * d + t * d + 2 * d
        for b in self._blocks:
            total += 4 * d * d + 4 * d          # wq/wk/wv/wo + biases
            total += 2 * d * f + f + d          # f1/f2 + biases
            total += 4 * d                      # two LayerNorms (g + b)
        return total

    @property
    def should_auto_train(self) -> bool:
        return (self._tokens_since_train >= AUTO_TRAIN_TOKENS
                and len(self._buffer) >= 8)

    # ------------------------------------------------------------------ observe
    def observe(self, text: Optional[str]) -> None:
        """Feeds text to the rolling training buffer + grows the vocabulary."""
        if not text or not text.strip():
            return
        for sentence in _SENTENCE_SPLIT.split(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            self._buffer.append(sentence)
            words = self._tokenize(sentence)
            self._tokens_observed += len(words)
            self._tokens_since_train += len(words)
            grew = False
            for w in words:
                if w not in self._index and len(self._vocab) < self.cfg.max_vocab:
                    self._index[w] = len(self._vocab)
                    self._vocab.append(w)
                    grew = True
            if grew:
                self._init_vocab_embeddings()
                self._adam.reset()
        if len(self._buffer) > MAX_BUFFERED_SENTENCES:
            del self._buffer[:len(self._buffer) - MAX_BUFFERED_SENTENCES]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return _util_tokenize(text) if text else []

    def _encode_word(self, word: str) -> int:
        return self._index.get(word, _UNK_ID)

    def _encode_context(self, context: str) -> List[int]:
        ids = [self._encode_word(w) for w in self._tokenize(context or "")]
        if not ids:
            ids = [_BOS_ID]
        return ids[-self.cfg.context_len:]

    def _pad(self, tokens: List[int]) -> List[int]:
        t = self.cfg.context_len
        if len(tokens) >= t:
            return tokens[-t:]
        return [_BOS_ID] * (t - len(tokens)) + list(tokens)

    # ------------------------------------------------------------------ training
    def auto_train(self) -> Optional[TransformerTrainingReport]:
        """Runs a lazy automatic pass when enough language accumulated."""
        if not self.should_auto_train:
            return None
        return self.train(AUTO_TRAIN_EPOCHS)

    def train(self, epochs: int = 8) -> TransformerTrainingReport:
        """Trains on everything in the observation buffer; returns the report."""
        start = time.time()
        epochs = max(1, epochs)
        self._tokens_since_train = 0

        total_tokens = sum(len(self._tokenize(s)) for s in self._buffer)
        if total_tokens < 2:
            return TransformerTrainingReport(0, 0, self._last_loss, self._last_loss,
                                             self.n_params, len(self._vocab),
                                             time.time() - start)

        # Training windows: EVERY contiguous subsequence of every sentence,
        # left-padded with <bos> when shorter than the context — covering all
        # (start, end) pairs means every layout inference can produce
        # (prefixes, mid-sentence fragments like "sat on the", long windows,
        # the empty all-<bos> prompt) appears in training exactly as _pad
        # builds it.
        xs: List[List[int]] = []
        ys: List[List[int]] = []
        for sentence in self._buffer:
            ids = [_BOS_ID] + [self._encode_word(w) for w in self._tokenize(sentence)] + [_EOS_ID]
            length = len(ids)
            for st in range(length - 1):
                max_e = min(length, st + 1 + self.cfg.context_len)
                for e in range(st + 1, max_e):
                    x = self._pad(ids[st:e])
                    y = x[1:] + [ids[e]]
                    xs.append(x)
                    ys.append(y)

        epoch_window_count = min(len(xs), self.cfg.max_windows_per_epoch)
        order = list(range(len(xs)))
        rng = random.Random(self.cfg.seed + self._epochs_trained * 7919)
        initial_loss = math.nan
        final_loss = math.nan

        for _ in range(epochs):
            rng.shuffle(order)
            loss_sum, loss_n = 0.0, 0
            for start_i in range(0, epoch_window_count, self.cfg.batch_size):
                idx = order[start_i:start_i + self.cfg.batch_size]
                loss = self._train_batch([xs[i] for i in idx], [ys[i] for i in idx])
                if math.isnan(initial_loss):
                    initial_loss = loss
                loss_sum += loss * len(idx)
                loss_n += len(idx)
            final_loss = loss_sum / max(1, loss_n)
            self._last_loss = final_loss

        self._epochs_trained += epochs
        return TransformerTrainingReport(epochs, epoch_window_count,
                                         initial_loss if not math.isnan(initial_loss) else final_loss,
                                         final_loss, self.n_params, len(self._vocab),
                                         time.time() - start)

    def _train_batch(self, xs: List[List[int]], ys: List[List[int]]) -> float:
        grads = self._new_grads()
        loss_sum = 0.0
        for x, y in zip(xs, ys):
            cache = self._forward(x, training=True)
            probs = cache["probs"]
            v = len(self._vocab)
            t = self.cfg.context_len
            d_logits = [[0.0] * v for _ in range(t)]
            for p in range(t):
                row = d_logits[p]
                prow = probs[p]
                target = y[p]
                loss_row = 0.0
                for i in range(v):
                    row[i] = prow[i]
                    if i == target:
                        loss_row -= math.log(max(prow[i], 1e-12))
                        row[i] -= 1.0
                for i in range(v):
                    row[i] /= t
                loss_sum += loss_row
            self._backward(cache, y, d_logits, grads)
        batch = len(xs)
        self._apply_grads(grads, batch)
        return loss_sum / t / batch

    # ------------------------------------------------------------------ forward
    def _layer_norm(self, x: List[List[float]], g: List[float], b: List[float]):
        t, d = len(x), len(x[0])
        out, hat, means, inv_stds = [], [], [], []
        for row in x:
            m = sum(row) / d
            var = sum((v - m) ** 2 for v in row) / d
            inv = 1.0 / math.sqrt(var + 1e-5)
            hrow = [(v - m) * inv for v in row]
            hat.append(hrow)
            out.append([hrow[c] * g[c] + b[c] for c in range(d)])
            means.append(m)
            inv_stds.append(inv)
        return out, hat, means, inv_stds

    @staticmethod
    def _matmul_add(a: List[List[float]], w: List[List[float]], bias: List[float]) -> List[List[float]]:
        out_dim, in_dim = len(w), len(a[0])
        out = []
        for row in a:
            orow = list(bias)
            for r in range(out_dim):
                wr = w[r]
                s = orow[r]
                for c in range(in_dim):
                    s += row[c] * wr[c]
                orow[r] = s
            out.append(orow)
        return out

    @staticmethod
    def _softmax(x: List[float]) -> List[float]:
        m = max(x) if x else 0.0
        exps = [math.exp(v - m) for v in x]
        total = sum(exps) or 1.0
        return [e / total for e in exps]

    def _forward(self, tokens: List[int], training: bool) -> Dict[str, Any]:
        d, t, h = self.cfg.dim, self.cfg.context_len, self.cfg.heads
        dh = d // h
        v_size = len(self._vocab)

        x0 = [[self._wte[tokens[p]][c] + self._wpe[p][c] for c in range(d)] for p in range(t)]
        layers: List[Dict[str, Any]] = []
        a = x0
        for block in self._blocks:
            lc: Dict[str, Any] = {"xin": a}
            a = self._forward_block(block, lc, training)
            layers.append(lc)

        lnf_out, lnf_hat, lnf_mean, lnf_inv = self._layer_norm(a, self._lnfg, self._lnfb)
        logits = [[sum(lnf_out[p][c] * self._wte[i][c] for c in range(d))
                   for i in range(v_size)] for p in range(t)]
        probs = [self._softmax(row) for row in logits]
        return {"tokens": tokens, "x0": x0, "layers": layers,
                "lnf_out": lnf_out, "lnf_hat": lnf_hat, "lnf_inv": lnf_inv,
                "logits": logits, "probs": probs}

    def _forward_block(self, b: _BlockParams, lc: Dict[str, Any], training: bool) -> List[List[float]]:
        d, t, h = self.cfg.dim, self.cfg.context_len, self.cfg.heads
        dh = d // h
        f = d * self.cfg.ffn_mult
        scale = 1.0 / math.sqrt(dh)

        ln1_out, ln1_hat, ln1_mean, ln1_inv = self._layer_norm(lc["xin"], b.ln1g, b.ln1b)
        lc["ln1_out"], lc["ln1_hat"], lc["ln1_inv"] = ln1_out, ln1_hat, ln1_inv
        q = self._matmul_add(ln1_out, b.wq, b.bq)
        k = self._matmul_add(ln1_out, b.wk, b.bk)
        vv = self._matmul_add(ln1_out, b.wv, b.bv)
        lc["q"], lc["k"], lc["v"] = q, k, vv

        attn_raw, attn_used, attn_mask = [], [], None
        if training and self.cfg.dropout > 0:
            attn_mask = []
        for head in range(h):
            raw_rows, used_rows, mask_rows = [], [], []
            for i in range(t):
                scores = [(-1e9 if j > i else scale * sum(
                    q[i][head * dh + c] * k[j][head * dh + c] for c in range(dh)))
                    for j in range(t)]
                prow = self._softmax(scores)
                raw_rows.append(prow)
                if attn_mask is not None:
                    keep = 1.0 - self.cfg.dropout
                    mrow = [0.0 if self._rng.random() < self.cfg.dropout else 1.0 / keep
                            for _ in range(t)]
                    mask_rows.append(mrow)
                    used_rows.append([prow[j] * mrow[j] for j in range(t)])
                else:
                    used_rows.append(prow)
            attn_raw.append(raw_rows)
            attn_used.append(used_rows)
            if attn_mask is not None:
                attn_mask.append(mask_rows)
        lc["attn"], lc["attn_raw"], lc["attn_mask"] = attn_used, attn_raw, attn_mask

        concat = []
        for i in range(t):
            row = [0.0] * d
            for head in range(h):
                for c in range(dh):
                    row[head * dh + c] = sum(
                        attn_used[head][i][j] * vv[j][head * dh + c] for j in range(t))
            concat.append(row)
        lc["concat"] = concat

        proj = self._matmul_add(concat, b.wo, b.bo)
        lc["proj"] = proj
        x2 = [[lc["xin"][i][c] + proj[i][c] for c in range(d)] for i in range(t)]
        lc["x2"] = x2

        ln2_out, ln2_hat, ln2_mean, ln2_inv = self._layer_norm(x2, b.ln2g, b.ln2b)
        lc["ln2_out"], lc["ln2_hat"], lc["ln2_inv"] = ln2_out, ln2_hat, ln2_inv
        f_pre = self._matmul_add(ln2_out, b.f1, b.bf1)
        lc["f_pre"] = f_pre
        f_gel, f_mask = [], None
        if training and self.cfg.dropout > 0:
            f_mask = []
            keep = 1.0 - self.cfg.dropout
            for i in range(t):
                mrow = [0.0 if self._rng.random() < self.cfg.dropout else 1.0 / keep
                        for _ in range(f)]
                f_mask.append(mrow)
                f_gel.append([gelu(f_pre[i][c]) * mrow[c] for c in range(f)])
        else:
            for i in range(t):
                f_gel.append([gelu(f_pre[i][c]) for c in range(f)])
        lc["f_gel"], lc["f_mask"] = f_gel, f_mask
        f_out = self._matmul_add(f_gel, b.f2, b.bf2)
        lc["f_out"] = f_out
        return [[x2[i][c] + f_out[i][c] for c in range(d)] for i in range(t)]

    # ------------------------------------------------------------------ backward
    def _new_grads(self) -> Dict[str, Any]:
        d, t = self.cfg.dim, self.cfg.context_len
        f = d * self.cfg.ffn_mult
        blocks = []
        for _ in self._blocks:
            blocks.append({
                "ln1g": [0.0] * d, "ln1b": [0.0] * d,
                "wq": [[0.0] * d for _ in range(d)], "bq": [0.0] * d,
                "wk": [[0.0] * d for _ in range(d)], "bk": [0.0] * d,
                "wv": [[0.0] * d for _ in range(d)], "bv": [0.0] * d,
                "wo": [[0.0] * d for _ in range(d)], "bo": [0.0] * d,
                "ln2g": [0.0] * d, "ln2b": [0.0] * d,
                "f1": [[0.0] * d for _ in range(f)], "bf1": [0.0] * f,
                "f2": [[0.0] * f for _ in range(d)], "bf2": [0.0] * d,
            })
        return {"wte": [[0.0] * d for _ in range(len(self._vocab))],
                "wpe": [[0.0] * d for _ in range(t)],
                "lnfg": [0.0] * d, "lnfb": [0.0] * d,
                "blocks": blocks}

    def _backward(self, cache: Dict[str, Any], targets: List[int],
                  d_logits: List[List[float]], g: Dict[str, Any]) -> None:
        d, t, h = self.cfg.dim, self.cfg.context_len, self.cfg.heads
        dh = d // h
        f = d * self.cfg.ffn_mult
        lnf_out, lnf_hat, lnf_inv = cache["lnf_out"], cache["lnf_hat"], cache["lnf_inv"]

        # Weight-tied head.
        d_lnf = [[0.0] * d for _ in range(t)]
        for p in range(t):
            drow = d_logits[p]
            for i in range(len(self._vocab)):
                dv = drow[i]
                if dv == 0.0:
                    continue
                emb = self._wte[i]
                drow_p = d_lnf[p]
                gwte = g["wte"][i]
                for c in range(d):
                    drow_p[c] += dv * emb[c]
                    gwte[c] += dv * lnf_out[p][c]

        d_up = self._layer_norm_backward(d_lnf, lnf_hat, self._lnfg, lnf_inv,
                                         g["lnfg"], g["lnfb"])

        for li in range(len(self._blocks) - 1, -1, -1):
            d_up = self._backward_block(self._blocks[li], cache["layers"][li],
                                        d_up, g["blocks"][li], g)

        for p in range(t):
            tok = cache["tokens"][p]
            gwte = g["wte"][tok]
            gwpe = g["wpe"][p]
            for c in range(d):
                gwte[c] += d_up[p][c]
                gwpe[c] += d_up[p][c]

    def _backward_block(self, b: _BlockParams, lc: Dict[str, Any],
                        d_out: List[List[float]], gb: Dict[str, Any],
                        g: Dict[str, Any]) -> List[List[float]]:
        d, t, h = self.cfg.dim, self.cfg.context_len, self.cfg.heads
        dh = d // h
        f = d * self.cfg.ffn_mult

        # Residual 2.
        d_x2 = [list(row) for row in d_out]
        d_f_out = [list(row) for row in d_out]

        # F2 backward.
        f_gel, f_out = lc["f_gel"], lc["f_out"]
        d_f_gel = [[sum(d_f_out[i][c] * b.f2[c][r] for c in range(d)) for r in range(f)]
                   for i in range(t)]
        for i in range(t):
            for c in range(d):
                gb["bf2"][c] += d_f_out[i][c]
        for i in range(t):
            dfp = d_f_gel[i]
            for r in range(f):
                for c in range(d):
                    gb["f2"][c][r] += d_f_out[i][c] * f_gel[i][r]

        # Dropout + GELU derivative.
        f_mask = lc["f_mask"]
        for i in range(t):
            for r in range(f):
                if f_mask is not None:
                    d_f_gel[i][r] *= f_mask[i][r]
                d_f_gel[i][r] *= gelu_grad(lc["f_pre"][i][r])

        # F1 backward.
        d_ln2 = [[sum(d_f_gel[i][r] * b.f1[r][c] for r in range(f)) for c in range(d)]
                 for i in range(t)]
        for i in range(t):
            for r in range(f):
                gb["bf1"][r] += d_f_gel[i][r]
        for i in range(t):
            for r in range(f):
                for c in range(d):
                    gb["f1"][r][c] += d_f_gel[i][r] * lc["ln2_out"][i][c]

        d_x2_from_ln = self._layer_norm_backward(d_ln2, lc["ln2_hat"], b.ln2g,
                                                 lc["ln2_inv"], gb["ln2g"], gb["ln2b"])
        for i in range(t):
            for c in range(d):
                d_x2[i][c] += d_x2_from_ln[i][c]

        # Residual 1: the same gradient feeds Wo and the block input.
        d_proj = d_x2

        d_concat = [[sum(d_proj[i][r] * b.wo[r][c] for r in range(d)) for c in range(d)]
                    for i in range(t)]
        for i in range(t):
            for c in range(d):
                gb["bo"][c] += d_proj[i][c]
        for i in range(t):
            for r in range(d):
                for c in range(d):
                    gb["wo"][r][c] += d_proj[i][r] * lc["concat"][i][c]

        # Attention backward per head.
        d_q = [[0.0] * d for _ in range(t)]
        d_k = [[0.0] * d for _ in range(t)]
        d_v = [[0.0] * d for _ in range(t)]
        attn_used, attn_raw = lc["attn"], lc["attn_raw"]
        attn_mask = lc["attn_mask"]
        q, k, v = lc["q"], lc["k"], lc["v"]
        for head in range(h):
            probs_used, probs_raw = attn_used[head], attn_raw[head]
            mask = attn_mask[head] if attn_mask is not None else None
            d_ah = [[sum(d_concat[i][head * dh + c] * v[j][head * dh + c]
                         for c in range(dh)) for j in range(t)] for i in range(t)]
            d_vh = [[sum(probs_used[i][j] * d_concat[i][head * dh + c]
                         for i in range(t)) for c in range(dh)] for j in range(t)]
            if mask is not None:
                for i in range(t):
                    for j in range(t):
                        d_ah[i][j] *= mask[i][j]
            for i in range(t):
                dot = sum(d_ah[i][j] * probs_raw[i][j] for j in range(t))
                for j in range(t):
                    d_ah[i][j] = probs_raw[i][j] * (d_ah[i][j] - dot)
            for i in range(t):
                for j in range(t):
                    s = d_ah[i][j] / math.sqrt(dh)
                    if s == 0.0:
                        continue
                    for c in range(dh):
                        d_q[i][head * dh + c] += s * k[j][head * dh + c]
                        d_k[j][head * dh + c] += s * q[i][head * dh + c]
            for j in range(t):
                for c in range(dh):
                    d_v[j][head * dh + c] += d_vh[j][c]

        # QKV + Wo projections backward.
        ln1_out = lc["ln1_out"]
        d_ln1 = [[0.0] * d for _ in range(t)]
        for i in range(t):
            for r in range(d):
                dq, dk, dv = d_q[i][r], d_k[i][r], d_v[i][r]
                if dq == 0.0 and dk == 0.0 and dv == 0.0:
                    continue
                for c in range(d):
                    s = 0.0
                    if dq != 0.0:
                        s += dq * b.wq[r][c]
                        gb["wq"][r][c] += dq * ln1_out[i][c]
                    if dk != 0.0:
                        s += dk * b.wk[r][c]
                        gb["wk"][r][c] += dk * ln1_out[i][c]
                    if dv != 0.0:
                        s += dv * b.wv[r][c]
                        gb["wv"][r][c] += dv * ln1_out[i][c]
                    d_ln1[i][c] += s
        for i in range(t):
            for c in range(d):
                gb["bq"][c] += d_q[i][c]
                gb["bk"][c] += d_k[i][c]
                gb["bv"][c] += d_v[i][c]

        d_xin = self._layer_norm_backward(d_ln1, lc["ln1_hat"], b.ln1g,
                                          lc["ln1_inv"], gb["ln1g"], gb["ln1b"])
        for i in range(t):
            for c in range(d):
                d_xin[i][c] += d_x2[i][c]
        return d_xin

    @staticmethod
    def _layer_norm_backward(dy: List[List[float]], hat: List[List[float]],
                             g: List[float], inv_std: List[float],
                             dg: List[float], db: List[float]) -> List[List[float]]:
        t, d = len(dy), len(dy[0])
        dx = []
        for i in range(t):
            dxhat = [dy[i][c] * g[c] for c in range(d)]
            dg_ = [0.0] * d
            mean_dxhat = 0.0
            mean_dot = 0.0
            for c in range(d):
                dg[c] += dy[i][c] * hat[i][c]
                db[c] += dy[i][c]
                mean_dxhat += dxhat[c]
                mean_dot += dxhat[c] * hat[i][c]
            mean_dxhat /= d
            mean_dot /= d
            dx.append([inv_std[i] * (dxhat[c] - mean_dxhat - hat[i][c] * mean_dot)
                       for c in range(d)])
        return dx

    def _apply_grads(self, g: Dict[str, Any], batch: int) -> None:
        inv = 1.0 / batch

        def scale_mat(m: List[List[float]]) -> None:
            for row in m:
                for c in range(len(row)):
                    row[c] *= inv

        def scale_vec(v: List[float]) -> None:
            for i in range(len(v)):
                v[i] *= inv

        scale_mat(g["wte"])
        scale_mat(g["wpe"])
        scale_vec(g["lnfg"])
        scale_vec(g["lnfb"])
        for gb in g["blocks"]:
            for key in ("wq", "wk", "wv", "wo", "f1", "f2"):
                scale_mat(gb[key])
            for key in ("ln1g", "ln1b", "bq", "bk", "bv", "bo", "ln2g", "ln2b", "bf1", "bf2"):
                scale_vec(gb[key])

        # Global gradient clipping.
        sq = 0.0

        def accum_mat(m: List[List[float]]) -> None:
            nonlocal sq
            for row in m:
                for x in row:
                    sq += x * x

        def accum_vec(v: List[float]) -> None:
            nonlocal sq
            for x in v:
                sq += x * x

        accum_mat(g["wte"])
        accum_mat(g["wpe"])
        accum_vec(g["lnfg"])
        accum_vec(g["lnfb"])
        for gb in g["blocks"]:
            for key in ("wq", "wk", "wv", "wo", "f1", "f2"):
                accum_mat(gb[key])
            for key in ("ln1g", "ln1b", "bq", "bk", "bv", "bo", "ln2g", "ln2b", "bf1", "bf2"):
                accum_vec(gb[key])
        norm = math.sqrt(sq)
        if norm > self.cfg.grad_clip > 0:
            s = self.cfg.grad_clip / norm

            def mul_mat(m: List[List[float]]) -> None:
                for row in m:
                    for c in range(len(row)):
                        row[c] *= s

            def mul_vec(v: List[float]) -> None:
                for i in range(len(v)):
                    v[i] *= s

            mul_mat(g["wte"])
            mul_mat(g["wpe"])
            mul_vec(g["lnfg"])
            mul_vec(g["lnfb"])
            for gb in g["blocks"]:
                for key in ("wq", "wk", "wv", "wo", "f1", "f2"):
                    mul_mat(gb[key])
                for key in ("ln1g", "ln1b", "bq", "bk", "bv", "bo", "ln2g", "ln2b", "bf1", "bf2"):
                    mul_vec(gb[key])

        lr = self.cfg.learning_rate
        self._adam.step("wte", self._wte, g["wte"], lr)
        self._adam.step("wpe", self._wpe, g["wpe"], lr)
        self._adam.step_vec("lnf.g", self._lnfg, g["lnfg"], lr)
        self._adam.step_vec("lnf.b", self._lnfb, g["lnfb"], lr)
        for li, (block, gb) in enumerate(zip(self._blocks, g["blocks"])):
            p = f"l{li}."
            self._adam.step_vec(p + "ln1.g", block.ln1g, gb["ln1g"], lr)
            self._adam.step_vec(p + "ln1.b", block.ln1b, gb["ln1b"], lr)
            self._adam.step(p + "wq", block.wq, gb["wq"], lr)
            self._adam.step_vec(p + "bq", block.bq, gb["bq"], lr)
            self._adam.step(p + "wk", block.wk, gb["wk"], lr)
            self._adam.step_vec(p + "bk", block.bk, gb["bk"], lr)
            self._adam.step(p + "wv", block.wv, gb["wv"], lr)
            self._adam.step_vec(p + "bv", block.bv, gb["bv"], lr)
            self._adam.step(p + "wo", block.wo, gb["wo"], lr)
            self._adam.step_vec(p + "bo", block.bo, gb["bo"], lr)
            self._adam.step_vec(p + "ln2.g", block.ln2g, gb["ln2g"], lr)
            self._adam.step_vec(p + "ln2.b", block.ln2b, gb["ln2b"], lr)
            self._adam.step(p + "f1", block.f1, gb["f1"], lr)
            self._adam.step_vec(p + "bf1", block.bf1, gb["bf1"], lr)
            self._adam.step(p + "f2", block.f2, gb["f2"], lr)
            self._adam.step_vec(p + "bf2", block.bf2, gb["bf2"], lr)

    # ------------------------------------------------------------------ inference
    def predict_next(self, context: str, count: int = 5) -> List[TokenPrediction]:
        """Next-token predictions after the context, best first (neural half)."""
        count = max(1, count)
        if not self.is_trained:
            return []
        tokens = self._pad(self._encode_context(context))
        cache = self._forward(tokens, training=False)
        probs = self._softmax(cache["logits"][-1])
        scored = [(self._vocab[i], probs[i]) for i in range(_SPECIAL_COUNT, len(self._vocab))
                  if probs[i] > 1e-9]
        scored.sort(key=lambda pair: -pair[1])
        top = scored[:count]
        total = sum(p for _, p in top)
        if total <= 0:
            return []
        return [TokenPrediction(tok, p / total) for tok, p in top]

    def score(self, text: str) -> float:
        """Mean per-token log-probability of the text (0 = no opinion)."""
        if not self.is_trained:
            return 0.0
        words = self._tokenize(text or "")
        if not words:
            return 0.0
        ids = [_BOS_ID] + [self._encode_word(w) for w in words] + [_EOS_ID]
        total, n = 0.0, 0
        for pos in range(1, len(ids)):
            window = self._pad(ids[max(0, pos - self.cfg.context_len):pos])
            cache = self._forward(window, training=False)
            probs = self._softmax(cache["logits"][-1])
            total += math.log(max(probs[ids[pos]], 1e-12))
            n += 1
        return total / n if n else 0.0

    def generate(self, prompt: str, max_tokens: int = 12,
                 temperature: float = 0.8) -> str:
        """Free-running continuation of the prompt (temperature sampling)."""
        if not self.is_trained:
            return ""
        ids = self._encode_context(prompt)
        produced: List[str] = []
        for _ in range(max(1, max_tokens)):
            window = self._pad(ids)
            cache = self._forward(window, training=False)
            logits = cache["logits"][-1]
            m = max(logits[_SPECIAL_COUNT:])
            weights = [math.exp((logits[i] - m) / max(0.05, temperature))
                       for i in range(_SPECIAL_COUNT, len(self._vocab))]
            total = sum(weights) or 1.0
            r = self._rng.random() * total
            acc = 0.0
            chosen = _SPECIAL_COUNT
            for i, wgt in enumerate(weights):
                acc += wgt
                if r <= acc:
                    chosen = _SPECIAL_COUNT + i
                    break
            if chosen == _EOS_ID:
                break
            produced.append(self._vocab[chosen])
            ids.append(chosen)
        return " ".join(produced)

    # ------------------------------------------------------------------ state
    def to_dict(self) -> dict:
        cfg = self.cfg
        return {
            "dim": cfg.dim, "heads": cfg.heads, "layers": cfg.layers,
            "context_len": cfg.context_len, "ffn_mult": cfg.ffn_mult,
            "dropout": cfg.dropout, "learning_rate": cfg.learning_rate,
            "seed": cfg.seed, "max_vocab": cfg.max_vocab,
            "vocab": list(self._vocab),
            "wte": self._wte, "wpe": self._wpe,
            "blocks": [{
                "ln1g": b.ln1g, "ln1b": b.ln1b,
                "wq": b.wq, "bq": b.bq, "wk": b.wk, "bk": b.bk,
                "wv": b.wv, "bv": b.bv, "wo": b.wo, "bo": b.bo,
                "ln2g": b.ln2g, "ln2b": b.ln2b,
                "f1": b.f1, "bf1": b.bf1, "f2": b.f2, "bf2": b.bf2,
            } for b in self._blocks],
            "lnfg": self._lnfg, "lnfb": self._lnfb,
            "buffer": list(self._buffer),
            "tokens_observed": self._tokens_observed,
            "epochs_trained": self._epochs_trained,
            "last_loss": self._last_loss,
        }

    def load_dict(self, data: dict) -> bool:
        """Restores a snapshot into this instance. Returns True on success."""
        if not data:
            return False
        try:
            cfg = self.cfg
            if (int(data.get("dim", cfg.dim)) != cfg.dim
                    or int(data.get("heads", cfg.heads)) != cfg.heads
                    or int(data.get("layers", cfg.layers)) != cfg.layers
                    or int(data.get("context_len", cfg.context_len)) != cfg.context_len):
                return False
            self._vocab = list(data["vocab"])
            self._index = {w: i for i, w in enumerate(self._vocab)}
            self._wte = [list(r) for r in data["wte"]]
            self._wpe = [list(r) for r in data["wpe"]]
            blocks = []
            for bd in data["blocks"]:
                b = _BlockParams(cfg.dim, cfg.dim * cfg.ffn_mult)
                b.ln1g = list(bd["ln1g"]); b.ln1b = list(bd["ln1b"])
                b.wq = [list(r) for r in bd["wq"]]; b.bq = list(bd["bq"])
                b.wk = [list(r) for r in bd["wk"]]; b.bk = list(bd["bk"])
                b.wv = [list(r) for r in bd["wv"]]; b.bv = list(bd["bv"])
                b.wo = [list(r) for r in bd["wo"]]; b.bo = list(bd["bo"])
                b.ln2g = list(bd["ln2g"]); b.ln2b = list(bd["ln2b"])
                b.f1 = [list(r) for r in bd["f1"]]; b.bf1 = list(bd["bf1"])
                b.f2 = [list(r) for r in bd["f2"]]; b.bf2 = list(bd["bf2"])
                blocks.append(b)
            self._blocks = blocks
            self._lnfg = list(data.get("lnfg", [1.0] * cfg.dim))
            self._lnfb = list(data.get("lnfb", [0.0] * cfg.dim))
            self._buffer = list(data.get("buffer", []))
            self._tokens_observed = int(data.get("tokens_observed", 0))
            self._epochs_trained = int(data.get("epochs_trained", 0))
            self._last_loss = float(data.get("last_loss", 0.0))
            self._adam.reset()
            return True
        except (KeyError, TypeError, ValueError):
            return False
