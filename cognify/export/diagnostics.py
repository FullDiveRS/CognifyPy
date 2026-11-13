"""GGUF diagnostics — will this file load in a stock llama.cpp build?

Answers, BEFORE a runner fails, the questions that matter:

* is the file structurally valid GGUF (spec level)?
* which tensor quantization types does it actually contain?
* can a STOCK llama.cpp build load it — and if not, why not, and what next?

This is the honest diagnosis for the classic load failure "the installed
llama.cpp cannot read this GGUF's weights (its tensor layout does not match
this build's quantization types)" — usually a file written by a NEWER
llama.cpp, or by a publisher's own fork (e.g. dynamic quants), that the local
build simply does not know how to decode.

Usage:

    from cognify.export.diagnostics import diagnose_gguf
    report = diagnose_gguf("yume-llamacpp.gguf")
    print(report.render())

    $ python -m cognify.export.diagnostics model.gguf
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .gguf import GGUF_MAGIC, GgufValueType

# ---------------------------------------------------------------------------
# GGML tensor type names. Ids are stable across llama.cpp; the highest ids grow
# over time, so unknown ids are reported as such (that is exactly the "newer
# build than yours" case). Table verified against ggml.h (ggml-org/llama.cpp).
# ---------------------------------------------------------------------------
_GGML_TYPE_NAMES: Dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_2 (removed)",
    5: "Q4_3 (removed)", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
    21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
    27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
    31: "Q4_0_4_4 (removed)", 32: "Q4_0_4_8 (removed)",
    33: "Q4_0_8_8 (removed)", 34: "TQ1_0", 35: "TQ2_0",
    36: "IQ4_NL_4_4 (removed)", 37: "IQ4_NL_4_8 (removed)",
    38: "IQ4_NL_8_8 (removed)", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0",
    42: "Q2_0", 43: "F8_E4M3",
}

# Types every llama.cpp build decodes (no quant history attached).
_UNIVERSAL_TYPES = {0, 1}          # F32, F16

_SCALAR_SIZES = {                   # spec value sizes for tolerant parsing
    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
}


def ggml_type_name(type_id: int) -> str:
    """Human name for a GGML type id (unknown ids say so — that is the
    "written by a newer llama.cpp" case)."""
    return _GGML_TYPE_NAMES.get(
        type_id, f"type-{type_id} (unknown here — likely a newer llama.cpp quant)")


@dataclass
class GgufDiagnosticsReport:
    """One llama.cpp-compatibility inspection of a GGUF file."""
    path: str = ""
    version: int = 0
    architecture: str = ""
    alignment: int = 32
    tensor_count: int = 0
    metadata_count: int = 0
    tensor_types: List[Tuple[int, str, int]] = field(default_factory=list)
    spec_valid: bool = False
    loadable_by_stock_llama_cpp: bool = False
    issues: List[str] = field(default_factory=list)
    advice: List[str] = field(default_factory=list)
    verdict: str = ""

    def render(self) -> str:
        """Renders the report as readable text (for logs and CLIs)."""
        lines = [f"GGUF {self.path}: version {self.version}, "
                 f"architecture '{self.architecture}', "
                 f"{self.tensor_count} tensor(s), "
                 f"{self.metadata_count} metadata key(s), "
                 f"alignment {self.alignment}"]
        if self.tensor_types:
            lines[0] += " — tensors: " + ", ".join(
                f"{count}× {name}" for _, name, count in self.tensor_types)
        for issue in self.issues:
            lines.append("  [issue]  " + issue)
        for hint in self.advice:
            lines.append("  [advice] " + hint)
        lines.append("  verdict: " + self.verdict)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# tolerant binary parsing (diagnostics never hard-fails — everything lands in
# the report's issues list)
# ---------------------------------------------------------------------------
def _read_u32(buf: bytes, pos: int) -> Tuple[int, int]:
    return struct.unpack_from("<I", buf, pos)[0], pos + 4


def _read_u64(buf: bytes, pos: int) -> Tuple[int, int]:
    return struct.unpack_from("<Q", buf, pos)[0], pos + 8


def _read_string(buf: bytes, pos: int) -> Tuple[str, int]:
    n, pos = _read_u64(buf, pos)
    raw = buf[pos:pos + n]
    if len(raw) != n:
        raise ValueError("truncated file (string)")
    return raw.decode("utf-8", errors="replace"), pos + n


def _read_value(buf: bytes, pos: int, vtype: int):
    if vtype == GgufValueType.STRING:
        return _read_string(buf, pos)
    if vtype == GgufValueType.ARRAY:
        etype, pos = _read_u32(buf, pos)
        count, pos = _read_u64(buf, pos)
        items = []
        for _ in range(count):
            item, pos = _read_value(buf, pos, etype)
            items.append(item)
        return items, pos
    size = _SCALAR_SIZES.get(vtype)
    if size is None:
        raise ValueError(f"unsupported scalar type {vtype}")
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
           6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}[vtype]
    return struct.unpack_from(fmt, buf, pos)[0], pos + size


def _tensor_bytes(ne: List[int], ggml_type: int) -> int:
    """Rough payload size of one tensor (exact for F32/F16/ints; for any quant
    type an upper bound: the densest real quant is Q8_0/Q8_1 at 1.125 bytes per
    element, so 1.25 B/elem + headroom never false-positives on legit quant
    files while still catching gross truncation)."""
    elements = 1
    for d in ne:
        elements *= d
    exact = {0: elements * 4, 1: elements * 2, 30: elements * 2,
             24: elements, 25: elements * 2, 26: elements * 4,
             27: elements * 8, 28: elements * 8}
    if ggml_type in exact:
        return exact[ggml_type]
    return -(-elements * 5 // 4) + 64      # ceil(elements * 1.25) + 64


def _lint_gpt2(tensors: Dict[str, Tuple[List[int], int]], metadata: Dict[str, object],
               issues: List[str], advice: List[str]) -> bool:
    """Conformance lint for the stock gpt2 loader: required hyperparameters,
    tensor names and shapes, tokenizer metadata. Returns True when clean."""
    loadable = True
    for key in ("gpt2.context_length", "gpt2.embedding_length", "gpt2.block_count",
                "gpt2.feed_forward_length", "gpt2.attention.head_count",
                "gpt2.attention.layer_norm_epsilon"):
        if key not in metadata:
            issues.append(
                f"gpt2 model is missing required hyperparameter '{key}'")
            loadable = False
    for key in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges"):
        if key not in metadata:
            issues.append(f"gpt2 model is missing tokenizer metadata '{key}'")
            loadable = False
    if "token_embd.weight" not in tensors:
        issues.append("gpt2 model has no 'token_embd.weight'")
        loadable = False

    def scalar(key: str) -> int:
        value = metadata.get(key)
        return value if isinstance(value, int) else 0

    emb, ctx = scalar("gpt2.embedding_length"), scalar("gpt2.context_length")
    blocks, ffn = scalar("gpt2.block_count"), scalar("gpt2.feed_forward_length")

    for name, (ne, _t) in tensors.items():
        problem = None
        if name == "token_embd.weight" and emb > 0 and (len(ne) != 2 or ne[0] != emb):
            problem = f"token_embd.weight ne={ne} but embedding_length={emb}"
        elif name == "position_embd.weight" and ctx > 0 and (len(ne) != 2 or ne[0] != emb):
            problem = f"position_embd.weight ne={ne} but embedding_length={emb}"
        elif name == "output_norm.weight" and (len(ne) != 1 or ne[0] != emb):
            problem = f"output_norm.weight ne={ne} but embedding_length={emb}"
        elif name.startswith("blk."):
            parts = name.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                block = int(parts[1])
                if not 0 <= block < blocks:
                    problem = (f"tensor '{name}' refers to block {block} "
                               f"but block_count={blocks}")
                elif emb > 0 and len(ne) == 2:
                    group = parts[-2]
                    if group == "attn_qkv" and ne[0] != emb:
                        problem = f"{name} ne={ne} but embedding_length={emb}"
                    elif group == "ffn_up" and ffn > 0 and ne[0] != emb:
                        problem = f"{name} ne={ne} but embedding_length={emb}"
                    elif group == "ffn_down" and ffn > 0 and ne[0] != ffn:
                        problem = f"{name} ne={ne} but feed_forward_length={ffn}"
        if problem is not None:
            issues.append(problem)
            advice.append("Tensor shapes must agree with the gpt2 hyperparameters, "
                          "or llama.cpp rejects the model at load time. "
                          "Re-export with cognify.export.export_gguf_llama_cpp.")
            loadable = False
    return loadable


def diagnose_gguf(path: str) -> GgufDiagnosticsReport:
    """Inspects a GGUF file and returns the full report. Never raises for spec
    problems — everything lands in ``report.issues``."""
    issues: List[str] = []
    advice: List[str] = []

    version = 0
    arch = ""
    alignment = 32
    type_counts: Dict[int, int] = {}
    tensor_count = 0
    metadata_count = 0
    spec_valid = True
    tensors: Dict[str, Tuple[List[int], int]] = {}
    offsets: Dict[str, int] = {}
    metadata: Dict[str, object] = {}

    try:
        with open(path, "rb") as fh:
            buf = fh.read()
        pos = 0
        magic, pos = _read_u32(buf, pos)
        if magic != GGUF_MAGIC:
            issues.append("not a GGUF file (bad magic)")
            spec_valid = False
        else:
            version, pos = _read_u32(buf, pos)
            if not 2 <= version <= 3:
                issues.append(f"GGUF version {version} — stock builds read v2 and v3")
                advice.append("Re-export the file with GGUF v2/v3 "
                              "(Cognify and CognifyPy both write v3).")
                spec_valid = False
            n_tensors, pos = _read_u64(buf, pos)
            n_kv, pos = _read_u64(buf, pos)
            metadata_count = n_kv
            for _ in range(n_kv):
                key, pos = _read_string(buf, pos)
                vtype, pos = _read_u32(buf, pos)
                value, pos = _read_value(buf, pos, vtype)
                metadata[key] = value
            arch = str(metadata.get("general.architecture", ""))
            align = metadata.get("general.alignment")
            if isinstance(align, int) and align > 0:
                alignment = align

            for _ in range(n_tensors):
                name, pos = _read_string(buf, pos)
                n_dims, pos = _read_u32(buf, pos)
                if not 1 <= n_dims <= 4:
                    issues.append(f"tensor '{name}': {n_dims} dimensions (spec allows 1..4)")
                    spec_valid = False
                    break
                ne = []
                for _ in range(n_dims):
                    d, pos = _read_u64(buf, pos)
                    ne.append(d)
                ggml_type, pos = _read_u32(buf, pos)
                offset, pos = _read_u64(buf, pos)
                tensors[name] = (ne, ggml_type)
                offsets[name] = offset
                type_counts[ggml_type] = type_counts.get(ggml_type, 0) + 1
                if offset % alignment:
                    issues.append(
                        f"tensor '{name}': offset {offset} not aligned to {alignment}")
                    spec_valid = False
                tensor_count += 1

            data_start = (pos + alignment - 1) // alignment * alignment
            # The furthest tensor's data end (offset + payload size, relative
            # to the data section) must fit inside the file.
            last_end = max((offsets[name] + _tensor_bytes(ne, t)
                            for name, (ne, t) in tensors.items()), default=0)
            if data_start + last_end > len(buf):
                issues.append("tensor data runs past the end of the file "
                              f"({data_start + last_end} > {len(buf)})")
                advice.append("The file is truncated or was written with absolute "
                              "instead of data-relative offsets — export it again.")
                spec_valid = False
    except Exception as exc:                                   # noqa: BLE001
        issues.append(f"could not parse the file: {exc}")
        spec_valid = False

    loadable = spec_valid
    type_list = [(tid, ggml_type_name(tid), count)
                 for tid, count in sorted(type_counts.items())]

    for tid, name, count in type_list:
        if tid in _UNIVERSAL_TYPES:
            continue
        loadable = False
        issues.append(f"{count} tensor(s) use {name} (ggml type {tid}) — "
                      "decodable only by llama.cpp builds that know this quant")
        advice.append(
            "This file was likely written by a newer llama.cpp, or a publisher's "
            "own fork. Pick another quant of the model, or use the llama.cpp build "
            "the model card names. If the model card says it runs on stock "
            "llama.cpp, the file may be damaged: download it again. Re-quantizing "
            "an F32 source with llama-quantize (Q8_0, Q5_K_M or Q4_K_M) also "
            "produces universally loadable weights.")

    if spec_valid and not arch:
        issues.append("no general.architecture metadata — llama.cpp cannot pick a graph")
        loadable = False
    elif spec_valid and arch not in ("gpt2", "cognify"):
        issues.append(f"architecture '{arch}' — stock llama.cpp has no graph for it")
        advice.append("Use a standard architecture (gpt2) for runnable exports; "
                      "CognifyPy's export_gguf_llama_cpp does this out of the box.")
        loadable = False
    elif spec_valid and arch == "cognify":
        advice.append("This is CognifyPy's archive GGUF (custom 'cognify' "
                      "architecture with the micro-net and RL Q-table). It is "
                      "spec-valid and readable by gguf tooling, but llama.cpp has "
                      "no graph for it — hand llama.cpp the '*-llamacpp.gguf' "
                      "(gpt2) file from the bundle instead.")
        loadable = False   # not RUNNABLE, even though the file is fine

    if spec_valid and arch == "gpt2":
        loadable = _lint_gpt2(tensors, metadata, issues, advice) and loadable

    if not spec_valid:
        verdict = "structurally invalid GGUF — repair or re-export"
    elif loadable:
        verdict = "spec-valid and loadable by a stock llama.cpp build"
    elif arch == "cognify":
        verdict = ("spec-valid Cognify archive GGUF "
                   "(not runnable by llama.cpp — by design)")
    else:
        verdict = "not loadable by a stock llama.cpp build (see issues and advice)"

    return GgufDiagnosticsReport(
        path=str(path), version=version, architecture=arch, alignment=alignment,
        tensor_count=tensor_count, metadata_count=metadata_count,
        tensor_types=type_list, spec_valid=spec_valid,
        loadable_by_stock_llama_cpp=loadable, issues=issues, advice=advice,
        verdict=verdict)


if __name__ == "__main__":                                     # pragma: no cover
    import sys
    if len(sys.argv) != 2:
        print("usage: python -m cognify.export.diagnostics <model.gguf>")
        raise SystemExit(2)
    print(diagnose_gguf(sys.argv[1]).render())
