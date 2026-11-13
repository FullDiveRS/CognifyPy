"""Safetensors writer — Hugging Face format, pure stdlib.

Layout: 8-byte little-endian header length N, then N bytes of JSON header
(tensor descriptors + optional ``__metadata__``), then the raw tensor byte
region. The header is padded with spaces so N is a multiple of 8. Data
offsets are relative to the start of the byte region, contiguous, ascending.

Also implements Hugging Face **checkpoint sharding**: splitting a model into
``{base}-00001-of-{N}.safetensors``-style shards plus a
``{base}.safetensors.index.json`` weight map — exactly how
``transformers.save_pretrained`` names sharded checkpoints.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .gguf import TensorSpec


class SafetensorsWriter:
    """Accumulates F32 tensors + metadata, then writes one safetensors file."""

    def __init__(self) -> None:
        self._tensors: List[TensorSpec] = []
        self._metadata: Dict[str, str] = {}

    def add_tensor(self, tensor: TensorSpec) -> "SafetensorsWriter":
        self._tensors.append(tensor)
        return self

    def add_metadata(self, key: str, value: str) -> "SafetensorsWriter":
        self._metadata[key] = value
        return self

    @property
    def tensor_count(self) -> int:
        return len(self._tensors)

    def write(self, path: str) -> None:
        header: Dict[str, object] = {}
        if self._metadata:
            header["__metadata__"] = dict(self._metadata)

        offset = 0
        for tensor in self._tensors:
            end = offset + tensor.byte_length
            header[tensor.name] = {
                "dtype": "F32",
                "shape": list(tensor.shape),
                "data_offsets": [offset, end],
            }
            offset = end

        header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
        padding = (8 - len(header_bytes) % 8) % 8
        header_bytes += b" " * padding

        with open(path, "wb") as buf:
            buf.write(struct.pack("<Q", len(header_bytes)))
            buf.write(header_bytes)
            for tensor in self._tensors:
                buf.write(struct.pack(f"<{len(tensor.data)}f", *tensor.data))


# ---------------------------------------------------------------------------
# Sharded exports ({base}-00001-of-{N}.safetensors + index.json)
# ---------------------------------------------------------------------------

def shard_filename(base_name: str, zero_based_index: int, total_shards: int) -> str:
    """``shard_filename("yume", 0, 6) -> "yume-00001-of-00006.safetensors"``."""
    width = max(5, len(str(abs(total_shards))))
    return (f"{base_name}-{zero_based_index + 1:0{width}d}"
            f"-of-{total_shards:0{width}d}.safetensors")


def index_filename(base_name: str) -> str:
    """``index_filename("yume") -> "yume.safetensors.index.json"``."""
    return f"{base_name}.safetensors.index.json"


def distribute(tensors: Sequence[TensorSpec], shards: int) -> List[List[TensorSpec]]:
    """Distributes tensors across *shards* balanced by payload bytes.

    Largest-tensor-first greedy: each tensor goes to the currently lightest
    shard (ties -> lowest shard index). Tensors are never split in half, so
    callers must cap ``shards`` at the tensor count. Within each shard the
    tensors keep their original registration order.
    """
    if shards <= 0:
        raise ValueError("at least one shard is required")
    if len(tensors) < shards:
        raise ValueError(f"cannot fill {shards} shards from {len(tensors)} tensors")

    buckets: List[List[int]] = [[] for _ in range(shards)]
    loads = [0] * shards
    order = sorted(range(len(tensors)), key=lambda i: (-tensors[i].byte_length, i))
    for i in order:
        target = min(range(shards), key=lambda s: (loads[s], s))
        buckets[target].append(i)
        loads[target] += tensors[i].byte_length
    return [[tensors[i] for i in sorted(bucket)] for bucket in buckets]


@dataclass(frozen=True)
class SafetensorsExportResult:
    """Files written by a safetensors export (sharded or not)."""
    directory: str
    shard_files: Tuple[str, ...]
    index_path: Optional[str]
    tensor_count: int
    requested_shards: int
    written_shards: int
    total_tensor_bytes: int
    weight_map: Dict[str, str]

    @property
    def files(self) -> Tuple[str, ...]:
        """Every file written (shards, then the index if present)."""
        if self.index_path is None:
            return self.shard_files
        return self.shard_files + (self.index_path,)

    @property
    def is_sharded(self) -> bool:
        return self.index_path is not None

    @property
    def primary_file(self) -> str:
        return self.shard_files[0]
