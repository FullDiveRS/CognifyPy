"""MindExporter — export a trained CognifyPy mind to standard ML formats.

One call, :func:`MindExporter.export_bundle`, writes:

* ``{name}.gguf``  — spec-compliant **GGUF v3**: llama.cpp-style metadata
  (``general.*``, personality traits, net dims, training stats) + F32 tensors
  (MicroNet ``w1/b1/w2/b2``, the Q-table and — once the neural language model
  is trained — the transformer weights) aligned to 32 bytes
* ``{name}-llamacpp.gguf`` — the trained transformer as a **runnable**
  ``gpt2``-architecture GGUF: llama.cpp, llama-cpp-python and Unsloth can load
  it, tokenize with the bundled word-level tokenizer and generate
* ``{name}.safetensors`` — the same tensors in the Hugging Face format
* ``tokenizer/tokenizer.json`` (+ ``vocab.json``, ``special_tokens_map.json``)
  — a HF WordPiece tokenizer built from the mind's own corpus
* ``train_config.json`` / ``finetune.sh`` / ``quantize.sh`` /
  ``training_data.txt`` / ``val_data.txt`` — the usual llama.cpp
  training / fine-tuning / quantization set
* ``config.json`` + ``README.md``

Scope note: a CognifyPy mind is a cognitive architecture (micro-net +
Q-learning + knowledge graph) plus a small internal GPT-style transformer.
The archive ``{name}.gguf`` uses the custom ``cognify`` architecture — real
and spec-compliant, but llama.cpp has no graph for it. The runnable
``{name}-llamacpp.gguf`` maps the transformer onto the stock ``gpt2``
architecture, so llama.cpp CAN run that file.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .gguf import GgufValueType, GgufWriter, TensorSpec, read_gguf
from .diagnostics import diagnose_gguf
from .safetensors import (SafetensorsExportResult, SafetensorsWriter, distribute,
                          index_filename, shard_filename)
from . import tokenizer as tokenizer_export
from . import llamacpp
from .options import ExportOptions
from ..core.options import ThinkingCeiling, ThinkingLevel

if TYPE_CHECKING:  # pragma: no cover
    from ..core.mind import Mind

NET_TENSOR_NAMES = ("cognify.net.w1", "cognify.net.b1",
                    "cognify.net.w2", "cognify.net.b2")


__all__ = ["ExportOptions", "MindExporter", "extract_tensors", "sanitize",
           "SafetensorsExportResult", "NET_TENSOR_NAMES",
           "extract_transformer_tensors", "LlamaCppGgufResult", "diagnose_gguf"]


def sanitize(name: str) -> str:
    """Filesystem/GGUF-safe lowercase model name (``My AI!`` -> ``my-ai``)."""
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", (name or "").strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "cognify-model"


def _flatten(matrix: List[List[float]]) -> List[float]:
    return [float(v) for row in matrix for v in row]


def extract_tensors(mind: "Mind") -> List[TensorSpec]:
    """Extracts the mind's tensors (empty list when learning is disabled).
    Once the neural language model is trained, its transformer tensors are
    included too — mirroring the C# exporter."""
    tensors: List[TensorSpec] = []
    learner = mind.learner
    if learner is not None:
        net = learner.net
        tensors.append(TensorSpec(NET_TENSOR_NAMES[0], _flatten(net.w1),
                                  net.n_hidden, net.n_in))
        tensors.append(TensorSpec(NET_TENSOR_NAMES[1], [float(v) for v in net.b1],
                                  net.n_hidden))
        tensors.append(TensorSpec(NET_TENSOR_NAMES[2], _flatten(net.w2),
                                  net.n_out, net.n_hidden))
        tensors.append(TensorSpec(NET_TENSOR_NAMES[3], [float(v) for v in net.b2],
                                  net.n_out))

        # Q-table: one row per state, one column per action.
        agent = learner.agent
        states = sorted({state for state, _action in agent._q})
        if states:
            from ..learning.qlearn import ACTIONS
            rows = [[float(agent.q_value(state, action)) for action in ACTIONS]
                    for state in states]
            tensors.append(TensorSpec("cognify.rl.q_table", _flatten(rows),
                                      len(states), len(ACTIONS)))

    tensors.extend(extract_transformer_tensors(mind))
    return tensors


def extract_transformer_tensors(mind: "Mind") -> List[TensorSpec]:
    """The neural language model's tensors (empty until it is trained):
    token/positional embeddings, per-block projections and LayerNorms, and the
    final LayerNorm — row-major ``[out][in]``, names ``cognify.transformer.*``."""
    neural = getattr(mind, "neural_language_model", None)
    if neural is None or not neural.is_trained:
        return []
    tensors: List[TensorSpec] = []
    for name, matrix, vector in neural.export_tensors():
        if matrix:
            tensors.append(TensorSpec(name, _flatten(matrix),
                                      len(matrix), len(matrix[0]) if matrix else 0))
        elif vector:
            tensors.append(TensorSpec(name, [float(v) for v in vector], len(vector)))
    return tensors


@dataclass
class LlamaCppGgufResult:
    """Result of a llama.cpp-runnable GGUF export."""
    path: str
    tensor_count: int
    vocab_size: int
    dim: int
    heads: int
    layers: int
    context_length: int
    feed_forward_length: int
    file_length: int


_GPT2_PREFIX = "cognify.transformer."
_TOKEN_NORMAL, _TOKEN_UNKNOWN, _TOKEN_CONTROL = 1, 2, 3
_SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


def _byte_map() -> List[str]:
    """GPT-2 byte-unicode table: bytes 33-126/161-172/174-255 map to
    themselves, the rest to U+0100+n (space 32 -> Ġ U+0120). Indexed by byte."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    char_for_byte = {b: chr(c) for b, c in zip(bs, cs)}
    return [char_for_byte[b] for b in range(256)]


def byte_encode(text: str, leading_space: bool = False) -> str:
    """GPT-2 byte-unicode encoding: UTF-8 bytes → printable characters
    (spaces become Ġ) — the convention llama.cpp and transformers both
    decode. ``leading_space`` prepends one synthetic Ġ for vocabulary words
    stored without spaces (skipped when the text already starts with one)."""
    table = _byte_map()
    if leading_space and not text.startswith(" "):
        text = " " + text
    raw = text.encode("utf-8")
    return "".join(table[b] for b in raw)


def build_bpe_merges(pieces: List[str]) -> List[str]:
    """BPE merge ranks that rebuild every vocabulary piece from single
    characters: for each piece the chain ('Ġ','w'), ('Ġw','o'), … merges
    adjacent symbols left to right, so BPE re-assembles the whole piece and
    the tokenizer looks up the exact word id. Pairs are deduplicated (first
    word wins) and ranks follow piece order."""
    ranks: List[str] = []
    seen = set()
    for piece in pieces:
        chars = list(piece)
        i = 0
        # No i++ after a merge: the merged symbol is immediately re-paired
        # with the next one, so repeated letters assemble correctly
        # ('Ġ h', 'Ġh e', 'Ġhe l', 'Ġhel l', 'Ġhell o' for 'Ġhello').
        while i + 1 < len(chars):
            entry = chars[i] + " " + chars[i + 1]
            if entry not in seen:
                seen.add(entry)
                ranks.append(entry)
            chars[i] = chars[i] + chars[i + 1]
            del chars[i + 1]
    return ranks


def export_gguf_llama_cpp(mind: "Mind", path: str,
                          options: Optional[ExportOptions] = None) -> LlamaCppGgufResult:
    """Exports the mind's TRAINED neural language model as a
    llama.cpp-runnable ``gpt2``-architecture GGUF (see the module docstring).
    Raises ``RuntimeError`` when the transformer has not been trained yet."""
    options = options or ExportOptions()
    neural = getattr(mind, "neural_language_model", None)
    if neural is None or not neural.is_trained:
        raise RuntimeError(
            "the neural language model is not trained yet — run train_deep_model() "
            "(or talk to the mind until it trains) before exporting for llama.cpp")

    cfg = neural.cfg
    vocab = neural.export_vocab()
    by_name = {t.name: t for t in extract_transformer_tensors(mind)}

    dim, layers, heads, context = cfg.dim, cfg.layers, cfg.heads, cfg.context_len
    wte = by_name[f"{_GPT2_PREFIX}wte"]
    wpe = by_name[f"{_GPT2_PREFIX}wpe"]
    vocab_size = wte.shape[0]
    ffn = by_name[f"{_GPT2_PREFIX}l0.ffn.w1"].shape[0]

    writer = GgufWriter()
    name = sanitize(options.model_name or mind.name)
    writer.add_string("general.architecture", "gpt2")
    writer.add_string("general.name", name)
    writer.add_string("general.description", options.description or
                      f"CognifyPy mind '{mind.name}' neural language model, "
                      "exported for llama.cpp")
    writer.add_uint32("general.file_type", 0)            # ALL_F32
    writer.add_uint32("general.alignment", 32)
    writer.add_uint32("general.quantization_version", 2)
    writer.add_string("general.license", "mit")

    writer.add_uint32("gpt2.context_length", context)
    writer.add_uint32("gpt2.embedding_length", dim)
    writer.add_uint32("gpt2.block_count", layers)
    writer.add_uint32("gpt2.feed_forward_length", ffn)
    writer.add_uint32("gpt2.attention.head_count", heads)
    writer.add_float32("gpt2.attention.layer_norm_epsilon", 1e-5)

    # tokenizer.ggml set: byte-unicode pieces, per-token types, specials and
    # merges that re-assemble every vocabulary word from its characters.
    pieces: List[str] = []
    types: List[int] = []
    for tid, word in enumerate(vocab):
        special = word in _SPECIAL_TOKENS
        pieces.append(word if special else byte_encode(word, leading_space=True))
        types.append(_TOKEN_NORMAL if not special else
                     (_TOKEN_UNKNOWN if tid == 1 else _TOKEN_CONTROL))
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre", "default")
    writer.add_value("tokenizer.ggml.tokens", GgufValueType.ARRAY, pieces)
    # llama.cpp requires token_type as an i32 array (GGUF_TYPE_INT32).
    writer.add_value("tokenizer.ggml.token_type", GgufValueType.ARRAY,
                     [(GgufValueType.INT32, t) for t in types])
    writer.add_value("tokenizer.ggml.merges", GgufValueType.ARRAY,
                     build_bpe_merges(pieces))
    writer.add_uint32("tokenizer.ggml.bos_token_id", 2)
    writer.add_uint32("tokenizer.ggml.eos_token_id", 3)
    writer.add_uint32("tokenizer.ggml.unknown_token_id", 1)
    writer.add_uint32("tokenizer.ggml.padding_token_id", 0)
    writer.add_bool("tokenizer.ggml.add_bos_token", False)
    writer.add_bool("tokenizer.ggml.add_eos_token", False)

    # Identity metadata — travels with the runnable weights.
    writer.add_bool("cognify.thinking.enabled", bool(mind.thinking_options.enabled))
    writer.add_string("cognify.thinking.level", ThinkingLevel.parse(mind.thinking_options.level).value)
    writer.add_string("cognify.thinking.ceiling", ThinkingCeiling.parse(mind.thinking_options.ceiling).value)
    writer.add_uint32("cognify.cycles", mind.cycle_count)

    writer.add_tensor(TensorSpec("token_embd.weight", wte.data, *wte.shape))
    writer.add_tensor(TensorSpec("position_embd.weight", wpe.data, *wpe.shape))
    for l in range(layers):
        p = f"{_GPT2_PREFIX}l{l}."
        writer.add_tensor(TensorSpec(f"blk.{l}.attn_norm.weight", by_name[f"{p}ln1.g"].data, dim))
        writer.add_tensor(TensorSpec(f"blk.{l}.attn_norm.bias", by_name[f"{p}ln1.b"].data, dim))
        wq, wk, wv = (by_name[f"{p}attn.w{k}"] for k in "qkv")
        fused = list(wq.data) + list(wk.data) + list(wv.data)
        writer.add_tensor(TensorSpec(f"blk.{l}.attn_qkv.weight", fused, dim * 3, dim))
        bq, bk, bv = (by_name[f"{p}attn.b{k}"] for k in "qkv")
        writer.add_tensor(TensorSpec(f"blk.{l}.attn_qkv.bias",
                                     list(bq.data) + list(bk.data) + list(bv.data), dim * 3))
        wo = by_name[f"{p}attn.wo"]
        writer.add_tensor(TensorSpec(f"blk.{l}.attn_output.weight", wo.data, *wo.shape))
        bo = by_name[f"{p}attn.bo"]
        writer.add_tensor(TensorSpec(f"blk.{l}.attn_output.bias", bo.data, *bo.shape))
        writer.add_tensor(TensorSpec(f"blk.{l}.ffn_norm.weight", by_name[f"{p}ln2.g"].data, dim))
        writer.add_tensor(TensorSpec(f"blk.{l}.ffn_norm.bias", by_name[f"{p}ln2.b"].data, dim))
        f1 = by_name[f"{p}ffn.w1"]
        writer.add_tensor(TensorSpec(f"blk.{l}.ffn_up.weight", f1.data, *f1.shape))
        writer.add_tensor(TensorSpec(f"blk.{l}.ffn_up.bias", by_name[f"{p}ffn.b1"].data, ffn))
        f2 = by_name[f"{p}ffn.w2"]
        writer.add_tensor(TensorSpec(f"blk.{l}.ffn_down.weight", f2.data, *f2.shape))
        writer.add_tensor(TensorSpec(f"blk.{l}.ffn_down.bias", by_name[f"{p}ffn.b2"].data, dim))
    writer.add_tensor(TensorSpec("output_norm.weight", by_name[f"{_GPT2_PREFIX}lnf.g"].data, dim))
    writer.add_tensor(TensorSpec("output_norm.bias", by_name[f"{_GPT2_PREFIX}lnf.b"].data, dim))

    full = Path(path).expanduser()
    directory = full.parent if str(full.parent) else Path(".")
    directory.mkdir(parents=True, exist_ok=True)
    writer.write(str(full))

    # ---- post-write self-verification ---------------------------------------
    # 1. Full GGUF round-trip: re-read the file and compare every tensor value
    #    against what we just wrote. A corrupt or misplaced payload fails HERE,
    #    not inside llama.cpp.
    written = writer.tensors
    gguf_check = read_gguf(str(full))
    for spec in written:
        back = gguf_check.read_tensor(spec.name)
        if back != spec.data:
            raise RuntimeError(
                f"export self-check failed: tensor {spec.name!r} does not read "
                "back identically — the written file is corrupt")

    # 2. llama.cpp conformance: hyperparameters, tensor names and shapes,
    #    tokenizer metadata — the exact checks the stock gpt2 loader does.
    report = diagnose_gguf(str(full))
    if not report.loadable_by_stock_llama_cpp or report.issues:
        raise RuntimeError(
            "export self-check failed: the produced GGUF would not load in "
            "stock llama.cpp — " + report.render())

    return LlamaCppGgufResult(
        path=str(full), tensor_count=writer.tensor_count, vocab_size=vocab_size,
        dim=dim, heads=heads, layers=layers, context_length=context,
        feed_forward_length=ffn, file_length=full.stat().st_size)


def build_gguf(mind: "Mind", options: ExportOptions,
               tensors: Optional[List[TensorSpec]] = None) -> GgufWriter:
    """Builds a GGUF writer preloaded with the mind's metadata and tensors."""
    tensors = extract_tensors(mind) if tensors is None else tensors
    writer = GgufWriter()

    name = sanitize(options.model_name or mind.name)
    writer.add_string("general.architecture", "cognify")
    writer.add_string("general.name", name)
    writer.add_string("general.description", options.description)
    writer.add_uint32("general.file_type", 0)          # 0 = ALL_F32
    writer.add_uint32("general.alignment", 32)
    writer.add_string("general.license", "mit")

    writer.add_uint32("cognify.cycles", mind.cycle_count)
    writer.add_bool("cognify.has_reasoning", bool(mind.flags.get("reasoning", True)))
    writer.add_bool("cognify.has_thinking", bool(mind.flags.get("thinking", True)))
    writer.add_bool("cognify.has_memory", bool(mind.flags.get("memory", True)))
    writer.add_bool("cognify.has_emotions", bool(mind.flags.get("emotions", True)))
    writer.add_bool("cognify.has_learning", mind.learner is not None)
    writer.add_uint32("cognify.capability_count", len(mind.capabilities))

    # Thinking budget metadata — travels with the weights so serving stacks
    # (llama.cpp / Unsloth) know this model's thinking switch, its operating
    # level and the hard cap it was built with.
    _t = mind.thinking_options
    writer.add_bool("cognify.thinking.enabled", bool(_t.enabled))
    writer.add_string("cognify.thinking.level", ThinkingLevel.parse(_t.level).value)
    writer.add_string("cognify.thinking.ceiling", ThinkingCeiling.parse(_t.ceiling).value)

    p = mind.personality
    writer.add_float32("cognify.personality.openness", p.openness)
    writer.add_float32("cognify.personality.conscientiousness", p.conscientiousness)
    writer.add_float32("cognify.personality.extraversion", p.extraversion)
    writer.add_float32("cognify.personality.agreeableness", p.agreeableness)
    writer.add_float32("cognify.personality.neuroticism", p.neuroticism)

    learner = mind.learner
    if learner is not None:
        writer.add_uint32("cognify.net.input_dim", learner.net.n_in)
        writer.add_uint32("cognify.net.hidden_dim", learner.net.n_hidden)
        writer.add_uint32("cognify.net.output_dim", learner.net.n_out)
        writer.add_uint32("cognify.training.samples_seen", learner.net.samples_seen)
        writer.add_uint32("cognify.training.epochs_trained", learner.net.epochs_trained)
        writer.add_uint32("cognify.rl.updates", learner.agent.updates)

    for tensor in tensors:
        writer.add_tensor(tensor)
    return writer


class MindExporter:
    """Exports minds to GGUF, safetensors, tokenizer.json and llama.cpp sets."""

    sanitize = staticmethod(sanitize)   # exposed for C#-style parity

    @staticmethod
    def export_gguf(mind: "Mind", path: str,
                    options: Optional[ExportOptions] = None) -> None:
        """Exports the mind as a GGUF v3 file. The written file is re-read and
        verified (spec structure, alignment, data in range) before this method
        returns — a corrupt export fails here, not inside gguf tooling later."""
        options = options or ExportOptions()
        full = Path(path).expanduser()
        if str(full.parent):
            full.parent.mkdir(parents=True, exist_ok=True)
        build_gguf(mind, options).write(str(full))
        report = diagnose_gguf(str(full))
        if not report.spec_valid:
            raise RuntimeError(
                "export self-check failed: the produced GGUF is not spec-valid — "
                + report.render())

    @staticmethod
    def export_gguf_llama_cpp(mind: "Mind", path: str,
                              options: Optional[ExportOptions] = None) -> LlamaCppGgufResult:
        """Exports the trained transformer as a llama.cpp-runnable
        ``gpt2``-architecture GGUF — the file to hand to llama.cpp, Unsloth or
        llama-cpp-python. Raises ``RuntimeError`` when untrained."""
        return export_gguf_llama_cpp(mind, path, options)

    @staticmethod
    def export_safetensors(mind: "Mind", path: str,
                           split_amount: Optional[int] = None,
                           options: Optional[ExportOptions] = None) -> SafetensorsExportResult:
        """Exports the mind's tensors as Hugging Face safetensors.

        ``split_amount`` selects the checkpoint sharding layout:

        * ``1`` (default) — one plain file at exactly ``path``, no index.
        * ``N >= 2`` — HF-style sharded checkpoint: shards named
          ``{base}-00001-of-{N}.safetensors`` … plus a HF weight-map index at
          ``{base}.safetensors.index.json`` (``metadata.total_size`` +
          ``weight_map``), the layout ``transformers`` / ``safetensors`` expect.
          Shards hold whole tensors and are balanced by payload size, so a
          split larger than the tensor count is clamped (a trained mind
          exports 5 tensors, so ``split_amount=62`` writes 5 shards —
          ``requested_shards``/``written_shards`` on the result report both).

        For backwards compatibility, an ``ExportOptions`` may still be passed
        positionally in the third slot.
        """
        if isinstance(split_amount, ExportOptions):        # legacy positional
            split_amount, options = 1, split_amount
        options = options or ExportOptions()
        split = 1 if split_amount is None else int(split_amount)
        if split <= 0:
            raise ValueError(
                f"split_amount must be >= 1 (got {split}); use 1 for a single unsplit file")

        tensors = extract_tensors(mind)
        if split > 1 and not tensors:
            raise RuntimeError(
                "cannot split a mind with no tensors into shards — build the mind "
                "with with_learning() so it has trained weights to export")

        full = Path(path).expanduser()
        directory = full.parent if str(full.parent) else Path(".")
        directory.mkdir(parents=True, exist_ok=True)

        base = full.name
        if base.lower().endswith(".safetensors"):
            base = base[: -len(".safetensors")]

        total_bytes = sum(t.byte_length for t in tensors)
        weight_map: Dict[str, str] = {}
        shard_files: List[str] = []
        index_path: Optional[str] = None

        if split <= 1:
            # Unsplit: write exactly the file that was asked for, no index.
            # "pt" is the Hugging Face convention transformers/Unsloth expect;
            # nonstandard values trigger warnings in strict loaders.
            writer = SafetensorsWriter()
            writer.add_metadata("format", "pt")
            writer.add_metadata("name", mind.name)
            for tensor in tensors:
                writer.add_tensor(tensor)
            writer.write(str(full))
            for tensor in tensors:
                weight_map[tensor.name] = full.name
            shard_files.append(str(full))
            written = 1
        else:
            written = min(split, len(tensors))
            for s, shard_tensors in enumerate(distribute(tensors, written)):
                shard_name = shard_filename(base, s, written)
                writer = SafetensorsWriter()
                writer.add_metadata("format", "pt")   # HF convention (see unsplit branch)
                writer.add_metadata("name", mind.name)
                for tensor in shard_tensors:
                    writer.add_tensor(tensor)
                shard_path = directory / shard_name
                writer.write(str(shard_path))
                shard_files.append(str(shard_path))
                for tensor in shard_tensors:
                    weight_map[tensor.name] = shard_name

            # HF-style index: {"metadata": {"total_size": N}, "weight_map": {...}}.
            index = {
                "metadata": {"total_size": total_bytes},
                "weight_map": weight_map,
            }
            index_path = str(directory / index_filename(base))
            with open(index_path, "w", encoding="utf-8") as fh:
                json.dump(index, fh, indent=2, sort_keys=True)

        return SafetensorsExportResult(
            directory=str(directory),
            shard_files=tuple(shard_files),
            index_path=index_path,
            tensor_count=len(tensors),
            requested_shards=split,
            written_shards=written,
            total_tensor_bytes=total_bytes,
            weight_map=dict(weight_map),
        )

    @staticmethod
    def export_tokenizer(mind: "Mind", directory: str,
                         options: Optional[ExportOptions] = None) -> int:
        options = options or ExportOptions()
        corpus = tokenizer_export.collect_corpus(mind)
        vocab = tokenizer_export.build_vocab(corpus, options.vocab_size)
        return tokenizer_export.write(directory, vocab, mind.name)

    @staticmethod
    def export_bundle(mind: "Mind", directory: str,
                      options: Optional[ExportOptions] = None) -> List[str]:
        """Writes the full export bundle; returns produced files (relative)."""
        options = options or ExportOptions()
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        name = sanitize(options.model_name or mind.name)
        produced: List[str] = []

        # 1. GGUF archive (custom cognify architecture — inspection/requantize).
        MindExporter.export_gguf(mind, str(out / f"{name}.gguf"), options)
        produced.append(f"{name}.gguf")

        # 1b. llama.cpp-runnable gpt2 GGUF (loads in llama.cpp / Unsloth).
        try:
            export_gguf_llama_cpp(mind, str(out / f"{name}-llamacpp.gguf"), options)
            produced.append(f"{name}-llamacpp.gguf")
        except RuntimeError:
            # Transformer not trained yet — the bundle ships without the
            # runnable file rather than failing the whole export.
            pass

        # 2. safetensors weights (optionally split HF-style across shards).
        st = MindExporter.export_safetensors(
            mind, str(out / f"{name}.safetensors"),
            max(1, int(options.safetensors_split)), options)
        produced += [Path(f).name for f in st.files]

        # 3. Tokenizer set.
        vocab_size = MindExporter.export_tokenizer(mind, str(out / "tokenizer"), options)
        produced += ["tokenizer/tokenizer.json", "tokenizer/vocab.json",
                     "tokenizer/special_tokens_map.json"]

        # 4. llama.cpp training / fine-tuning / quantization set.
        corpus = tokenizer_export.collect_corpus(mind)
        for path in llamacpp.write(str(out), corpus, name, options):
            produced.append(Path(path).name)

        # 5. config.json (HF-style architecture summary).
        tensors = extract_tensors(mind)
        arch = {
            "model_type": "cognify-mind",
            "architecture": "cognify",
            "name": mind.name,
            "flags": dict(mind.flags),
            "capabilities": [c.name for c in mind.capabilities],
            "vocab_size": vocab_size,
            "tensor_count": len(tensors),
            "cycles_trained": mind.cycle_count,
            "export_options": {
                "quantization": options.quantization_type,
                "context_length": options.context_length,
                "batch_size": options.batch_size,
                "epochs": options.epochs,
                "learning_rate": options.learning_rate,
                "safetensors_shards": st.written_shards,
            },
        }
        (out / "config.json").write_text(json.dumps(arch, indent=2), encoding="utf-8")
        produced.append("config.json")

        # 6. README explaining the bundle.
        if st.is_sharded:
            st_row = (f"| `{name}-00001-of-{st.written_shards:05d}.safetensors` "
                      f"(+ `{name}.safetensors.index.json`) | Hugging Face SHARDED "
                      f"safetensors ({st.written_shards} shards, whole tensors, "
                      f"size-balanced). Load via the index weight_map with `transformers` |")
        else:
            st_row = (f"| `{name}.safetensors` | Hugging Face safetensors with the same "
                      f"tensors. Load: `safetensors.numpy.load_file(\"{name}.safetensors\")` |")

        readme = f"""# {mind.name} — exported model bundle

Produced by ``cognify.export.MindExporter`` from a live cognitive mind.

| File | Purpose |
|------|---------|
| `{name}.gguf` | GGUF v3 archive: mind metadata + MicroNet weights + Q-table + transformer weights (F32, 32-byte aligned). Inspect: `gguf-dump {name}.gguf` |
| `{name}-llamacpp.gguf` | llama.cpp-RUNNABLE gpt2 GGUF of the trained neural language model — load in llama.cpp, Unsloth, llama-cpp-python |
{st_row}
| `tokenizer/tokenizer.json` | HF WordPiece tokenizer built from the mind's corpus ({vocab_size} entries) |
| `train_config.json` | llama.cpp finetune-style hyperparameter config |
| `finetune.sh` | Ready-to-run `llama-finetune` (LoRA) command |
| `quantize.sh` | `llama-quantize` pipeline -> Q4_K_M / Q5_K_M / Q8_0 |
| `training_data.txt` / `val_data.txt` | Corpus harvested from the mind (~90/10 split) |
| `config.json` | Architecture summary |

## Usage notes

- `{name}-llamacpp.gguf` is a standard gpt2-architecture GGUF: llama.cpp (any
  recent build), llama-cpp-python and Unsloth can load it, tokenize with the
  bundled word-level tokenizer and generate. The FFN uses the tanh GELU
  approximation while llama.cpp evaluates exact GELU (a sub-0.003 delta per
  activation); out-of-vocabulary words fall back to `<unk>`.
- `{name}.gguf` is the archive file (custom `cognify` architecture). It is
  spec-compliant and readable by the gguf tooling ecosystem (gguf-py,
  llama-quantize), but llama.cpp has no graph for the `cognify` architecture —
  treat it as a training/initialization/archive artifact (e.g. LoRA seeds via
  `finetune.sh`).
- Quantization: run `bash quantize.sh` after building llama.cpp.
  Q4_K_M is the usual quality/size trade-off; Q8_0 is near-lossless. These
  stock quant types load in any llama.cpp build that supports K-quants (2023+).
- Every GGUF this bundle shipped was re-read and verified after writing (spec
  structure, alignment, payload round-trip, and for `{name}-llamacpp.gguf` the
  full gpt2 loader conformance: hyperparameters, tensor names and shapes,
  tokenizer metadata).
- If a GGUF ever fails to load with "tensor layout does not match this build's
  quantization types", the file contains quant types newer than the installed
  build (e.g. dynamic quants from a publisher fork such as Unsloth, or MXFP4):
  pick another quant, or use the llama.cpp build the model card names.
  Re-quantizing an F32 source with `llama-quantize` always yields universally
  loadable weights. Inspect any file first with
  `python -m cognify.export.diagnostics <file.gguf>`
  (CognifyPy) or `GgufDiagnostics.Inspect(path)` (Cognify).
"""
        (out / "README.md").write_text(readme, encoding="utf-8")
        produced.append("README.md")

        return produced
