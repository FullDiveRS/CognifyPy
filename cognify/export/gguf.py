"""GGUF v3 writer — spec-compliant, little-endian, pure stdlib.

Follows the GGUF format defined by llama.cpp (gguf-py specs):

* header: magic ``GGUF`` (u32), version (u32) = 3, tensor_count (u64), kv_count (u64)
* key-value metadata: string key + value-type (u32) + value
* tensor infos: name, n_dims (u32), ne (u64 each — **GGML order, ne[0] is the
  fastest-varying dimension**, i.e. the logical shape written reversed), ggml
  type (u32), offset (u64)
* tensor offsets are **relative to the start of the tensor data section** (the
  first tensor sits at offset 0) — absolute file positions make every
  spec-conformant reader (llama.cpp, gguf-py) read the wrong bytes
* tensor data region aligned to ``general.alignment`` (default 32 bytes)

The files can be inspected with ``gguf-dump`` (gguf-py), verified with
:func:`read_gguf`, and requantized with ``llama-quantize``.
"""
from __future__ import annotations

import struct
from typing import BinaryIO, Dict, List, Sequence, Tuple, Union

GGUF_MAGIC = 0x46554747          # "GGUF"
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32
GGML_F32 = 0                     # ggml tensor type id for 32-bit IEEE float


class GgufValueType:
    """GGUF value types (spec v3)."""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


_SCALAR_FMT: Dict[int, str] = {
    GgufValueType.UINT8: "<B", GgufValueType.INT8: "<b",
    GgufValueType.UINT16: "<H", GgufValueType.INT16: "<h",
    GgufValueType.UINT32: "<I", GgufValueType.INT32: "<i",
    GgufValueType.FLOAT32: "<f", GgufValueType.BOOL: "<B",
    GgufValueType.UINT64: "<Q", GgufValueType.INT64: "<q",
    GgufValueType.FLOAT64: "<d",
}


class TensorSpec:
    """A named F32 tensor built from a row-major payload.

    ``shape`` is the LOGICAL shape, outer dimension first (shape[0] = rows,
    shape[1] = columns for a matrix) — the writer converts to GGML's ne order,
    where ne[0] must be the fastest-varying (innermost) dimension.
    """

    def __init__(self, name: str, data: Sequence[float], *shape: int) -> None:
        if not name:
            raise ValueError("tensor name must not be empty")
        n = 1
        for dim in shape:
            n *= dim
        if len(data) != n:
            raise ValueError(
                f"tensor {name!r}: {len(data)} values but shape {shape} implies {n}")
        self.name = name
        # Normalize to F32 precision NOW (the file format is F32): a Python
        # float is 64-bit, and a payload written as "<f" rounds to 32 bits.
        # Storing the rounded values makes round-trip verification exact
        # instead of precision-dependent — mirroring C#'s float[] semantics.
        packed = struct.pack(f"<{n}f", *(float(v) for v in data)) if n else b""
        self.data = list(struct.unpack(f"<{n}f", packed)) if n else []
        self.shape = tuple(int(d) for d in shape)

    @property
    def byte_length(self) -> int:
        return len(self.data) * 4


def _write_string(buf: BinaryIO, s: str) -> None:
    raw = s.encode("utf-8")
    buf.write(struct.pack("<Q", len(raw)))
    buf.write(raw)


def _write_scalar(buf: BinaryIO, vtype: int, value: Union[int, float, bool, str]) -> None:
    if vtype == GgufValueType.STRING:
        _write_string(buf, str(value))
    elif vtype == GgufValueType.BOOL:
        buf.write(struct.pack("<B", 1 if value else 0))
    else:
        buf.write(struct.pack(_SCALAR_FMT[vtype], value))


def _infer_elem_type(value: object) -> int:
    """Element type for raw arrays (bool first — it subclasses int)."""
    if isinstance(value, bool):
        return GgufValueType.BOOL
    if isinstance(value, int):
        return GgufValueType.UINT32
    if isinstance(value, float):
        return GgufValueType.FLOAT32
    return GgufValueType.STRING


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class GgufWriter:
    """Accumulates metadata and F32 tensors, then writes one GGUF v3 file."""

    def __init__(self, alignment: int = DEFAULT_ALIGNMENT) -> None:
        self.alignment = int(alignment)
        self._kv: List[Tuple[str, int, object]] = []
        self._kv_index: Dict[str, int] = {}
        self._tensors: List[TensorSpec] = []

    # ------------------------------------------------------------------ metadata
    def add_value(self, key: str, vtype: int, value: object) -> "GgufWriter":
        if key in self._kv_index:
            self._kv[self._kv_index[key]] = (key, vtype, value)
        else:
            self._kv_index[key] = len(self._kv)
            self._kv.append((key, vtype, value))
        return self

    def add_string(self, key: str, value: str) -> "GgufWriter":
        return self.add_value(key, GgufValueType.STRING, value or "")

    def add_uint32(self, key: str, value: int) -> "GgufWriter":
        return self.add_value(key, GgufValueType.UINT32, int(value))

    def add_uint64(self, key: str, value: int) -> "GgufWriter":
        return self.add_value(key, GgufValueType.UINT64, int(value))

    def add_float32(self, key: str, value: float) -> "GgufWriter":
        return self.add_value(key, GgufValueType.FLOAT32, float(value))

    def add_bool(self, key: str, value: bool) -> "GgufWriter":
        return self.add_value(key, GgufValueType.BOOL, bool(value))

    def add_string_array(self, key: str, values: Sequence[str]) -> "GgufWriter":
        return self.add_value(key, GgufValueType.ARRAY,
                              [(GgufValueType.STRING, str(v)) for v in values])

    # ------------------------------------------------------------------ tensors
    def add_tensor(self, tensor: TensorSpec) -> "GgufWriter":
        self._tensors.append(tensor)
        return self

    @property
    def tensor_count(self) -> int:
        return len(self._tensors)

    @property
    def tensors(self) -> List[TensorSpec]:
        """Registered tensors, in write order (exposed for post-write
        verification round-trips)."""
        return list(self._tensors)

    @property
    def metadata_count(self) -> int:
        return len(self._kv)

    # ------------------------------------------------------------------ write
    def write(self, path: str) -> None:
        with open(path, "wb") as buf:
            buf.write(struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION,
                                  len(self._tensors), len(self._kv)))

            # ---- key-value metadata
            for key, vtype, value in self._kv:
                _write_string(buf, key)
                buf.write(struct.pack("<I", vtype))
                if vtype == GgufValueType.ARRAY:
                    items = list(value)     # raw values or (elem_type, val) pairs
                    if items and isinstance(items[0], tuple) and len(items[0]) == 2 \
                            and isinstance(items[0][0], int):
                        elem_type = items[0][0]
                        payload = [v for _, v in items]
                    else:
                        elem_type = _infer_elem_type(items[0]) if items \
                            else GgufValueType.STRING
                        payload = items
                    buf.write(struct.pack("<I", elem_type))
                    buf.write(struct.pack("<Q", len(payload)))
                    for item in payload:
                        _write_scalar(buf, elem_type, item)
                else:
                    _write_scalar(buf, vtype, value)

            # ---- tensor infos with pre-computed aligned offsets.
            # The data region starts AFTER the whole info block, so its size
            # must be computed upfront (name len + dims + ne + type + off).
            # Offsets are RELATIVE to the data section (spec: "the offset of
            # the tensor data within the tensor data section") — the first
            # tensor is at offset 0. Shapes are written in GGML ne order:
            # ne[0] is the fastest-varying (innermost) dimension, so the
            # logical shape (outer first) is written REVERSED.
            info_size = 0
            for tensor in self._tensors:
                info_size += 8 + len(tensor.name.encode("utf-8"))   # name
                info_size += 4                                      # n_dims
                info_size += 8 * len(tensor.shape)                  # ne
                info_size += 4                                      # ggml type
                info_size += 8                                      # offset
            data_start = _align(buf.tell() + info_size, self.alignment)

            offset = 0
            for tensor in self._tensors:
                _write_string(buf, tensor.name)
                buf.write(struct.pack("<I", len(tensor.shape)))
                for dim in reversed(tensor.shape):
                    buf.write(struct.pack("<Q", dim))               # ne[0] = innermost
                buf.write(struct.pack("<I", GGML_F32))
                buf.write(struct.pack("<Q", offset))
                offset = _align(offset + tensor.byte_length, self.alignment)

            # ---- pad to alignment, then tensor data
            self._pad(buf)
            for tensor in self._tensors:
                self._pad(buf)
                buf.write(struct.pack(f"<{len(tensor.data)}f", *tensor.data))

    def _pad(self, buf: BinaryIO) -> None:
        remainder = buf.tell() % self.alignment
        if remainder:
            buf.write(b"\x00" * (self.alignment - remainder))


class InvalidGgufError(Exception):
    """Raised when a GGUF file violates the spec (bad magic, misaligned or
    out-of-range tensor data, truncated file, ...)."""


class GgufTensorInfo:
    """One tensor read back from a GGUF file."""

    def __init__(self, name: str, ggml_type: int, shape: Sequence[int],
                 offset: int, byte_length: int) -> None:
        self.name = name
        self.ggml_type = ggml_type
        self.shape = tuple(shape)          # logical shape, outer first
        self.offset = offset               # relative to the data section
        self.byte_length = byte_length

    @property
    def element_count(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


class GgufFile:
    """A strict GGUF v3 reader used to VERIFY what the writer produced — the
    same checks llama.cpp performs on load: magic + version, metadata value
    types, tensor offsets relative to the aligned data section, every tensor
    fully inside the file, and F32 payloads that decode to the exact values
    written. ``info.shape`` is the logical shape (ne reversed back)."""

    def __init__(self, path: str) -> None:
        with open(path, "rb") as fh:
            self._bytes = fh.read()
        pos = 0

        magic, = struct.unpack_from("<I", self._bytes, pos)
        pos += 4
        if magic != GGUF_MAGIC:
            raise InvalidGgufError(f"bad magic 0x{magic:X} (expected GGUF)")
        self.version, = struct.unpack_from("<I", self._bytes, pos)
        pos += 4
        if not 2 <= self.version <= 3:
            raise InvalidGgufError(f"unsupported GGUF version {self.version}")
        tensor_count, kv_count = struct.unpack_from("<QQ", self._bytes, pos)
        pos += 16

        self.metadata: Dict[str, object] = {}
        for _ in range(kv_count):
            key = self._read_string(pos)[0]
            pos = self._read_string(pos)[1]
            vtype, = struct.unpack_from("<I", self._bytes, pos)
            pos += 4
            value, pos = self._read_value(vtype, pos)
            self.metadata[key] = value

        self.alignment = DEFAULT_ALIGNMENT
        align = self.metadata.get("general.alignment")
        if isinstance(align, int):
            self.alignment = align
        if self.alignment <= 0:
            raise InvalidGgufError("general.alignment is zero")

        self.tensors: List[GgufTensorInfo] = []
        max_end = 0
        for _ in range(tensor_count):
            name, pos = self._read_string(pos)
            n_dims, = struct.unpack_from("<I", self._bytes, pos)
            pos += 4
            if not 1 <= n_dims <= 4:
                raise InvalidGgufError(f"tensor {name!r}: {n_dims} dimensions")
            ne = []
            for _ in range(n_dims):
                d, = struct.unpack_from("<Q", self._bytes, pos)
                pos += 8
                if d <= 0:
                    raise InvalidGgufError(f"tensor {name!r}: non-positive dimension")
                ne.append(d)
            ggml_type, rel = struct.unpack_from("<IQ", self._bytes, pos)
            pos += 12
            if ggml_type != GGML_F32:
                raise InvalidGgufError(f"tensor {name!r}: only F32 supported (type {ggml_type})")
            if rel % self.alignment:
                raise InvalidGgufError(f"tensor {name!r}: offset {rel} not aligned to {self.alignment}")
            elements = 1
            for d in ne:
                elements *= d
            byte_length = elements * 4
            self.tensors.append(GgufTensorInfo(name, ggml_type, tuple(reversed(ne)), rel, byte_length))
            max_end = max(max_end, rel + byte_length)

        self.data_start = self._align(pos)
        if self.data_start + max_end > len(self._bytes):
            raise InvalidGgufError(
                "tensor data runs past the end of the file — offsets must be "
                "relative to the data section")

    # ------------------------------------------------------------------ read
    def tensor(self, name: str) -> GgufTensorInfo:
        for info in self.tensors:
            if info.name == name:
                return info
        raise InvalidGgufError(f"no tensor named {name!r}")

    def read_tensor(self, name: str) -> List[float]:
        info = self.tensor(name)
        start = self.data_start + info.offset
        return list(struct.unpack_from(f"<{info.element_count}f", self._bytes, start))

    # ------------------------------------------------------------- internals
    def _read_string(self, pos: int) -> Tuple[str, int]:
        (length,) = struct.unpack_from("<Q", self._bytes, pos)
        pos += 8
        raw = self._bytes[pos:pos + length]
        if len(raw) != length:
            raise InvalidGgufError("truncated file (string)")
        return raw.decode("utf-8"), pos + length

    def _read_value(self, vtype: int, pos: int) -> Tuple[object, int]:
        if vtype == GgufValueType.STRING:
            return self._read_string(pos)
        if vtype == GgufValueType.ARRAY:
            elem_type, = struct.unpack_from("<I", self._bytes, pos)
            pos += 4
            count, = struct.unpack_from("<Q", self._bytes, pos)
            pos += 8
            items = []
            for _ in range(count):
                item, pos = self._read_value(elem_type, pos)
                items.append(item)
            return items, pos
        fmt = _SCALAR_FMT.get(vtype)
        if fmt is None:
            raise InvalidGgufError(f"unsupported scalar type {vtype}")
        (value,) = struct.unpack_from(fmt, self._bytes, pos)
        return value, pos + struct.calcsize(fmt)

    def _align(self, value: int) -> int:
        return (value + self.alignment - 1) // self.alignment * self.alignment


def read_gguf(path: str) -> GgufFile:
    """Reads and validates a GGUF file; raises :class:`InvalidGgufError` on
    any spec violation. See :class:`GgufFile` for what is checked."""
    return GgufFile(path)
