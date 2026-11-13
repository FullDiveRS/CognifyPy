"""Tests for sharded safetensors exports ({base}-00001-of-{N}.safetensors + index).

Mirrors the C# ExportTests sharding suite one-to-one.
"""
import json
import struct
from pathlib import Path

import pytest

from cognify import MindBuilder
from cognify.export import ExportOptions, MindExporter
from cognify.export.safetensors import distribute, index_filename, shard_filename


def make_trained_mind():
    mind = (MindBuilder("Yume")
            .with_big_five(0.9, 0.6, 0.75, 0.85, 0.3)
            .with_seed(7)
            .build())
    mind.process("My name is Alex.")
    mind.process("Birds are animals. Animals are living things.")
    mind.process("The sky is blue.")
    mind.feedback(0.8)
    mind.process("Thanks, that helped a lot!")
    mind.feedback(-0.2)
    mind.process("Hmm, not quite what I meant.")
    return mind


def parse_safetensors(path):
    """Tensor name -> (shape, raw payload bytes), read at declared offsets."""
    buf = Path(path).read_bytes()
    (header_len,) = struct.unpack_from("<Q", buf, 0)
    header = json.loads(buf[8:8 + header_len])
    data_start = 8 + header_len
    out = {}
    for name, desc in header.items():
        if name == "__metadata__":
            continue
        start, end = desc["data_offsets"]
        payload = buf[data_start + start:data_start + end]
        out[name] = (tuple(desc["shape"]), payload)
    return out


class TestNamingHelpers:
    def test_shard_filename_uses_five_digit_padding(self):
        assert shard_filename("yume", 0, 62) == "yume-00001-of-00062.safetensors"
        assert shard_filename("yume", 61, 62) == "yume-00062-of-00062.safetensors"
        assert shard_filename("yume", 0, 6) == "yume-00001-of-00006.safetensors"

    def test_padding_widens_past_99999(self):
        name = shard_filename("big", 0, 123456)
        assert name == "big-000001-of-123456.safetensors"

    def test_index_filename(self):
        assert index_filename("yume") == "yume.safetensors.index.json"

    def test_distribute_is_balanced_and_lossless(self):
        from cognify.export.gguf import TensorSpec
        tensors = [TensorSpec("big", [0.0] * 100, 100),
                   TensorSpec("mid", [0.0] * 40, 40),
                   TensorSpec("small", [0.0] * 10, 10)]
        shards = distribute(tensors, 2)
        flattened = [t.name for shard in shards for t in shard]
        assert sorted(flattened) == ["big", "mid", "small"]   # nothing lost or duplicated
        loads = [sum(t.byte_length for t in shard) for shard in shards]
        assert max(loads) - min(loads) <= max(t.byte_length for t in tensors)


class TestShardedExport:
    def test_split_writes_hf_shards_and_index(self, tmp_path):
        mind = make_trained_mind()
        result = MindExporter.export_safetensors(mind, str(tmp_path / "yume.safetensors"), 3)

        assert result.is_sharded
        assert result.written_shards == 3
        assert len(result.shard_files) == 3
        assert result.tensor_count == 5          # w1, b1, w2, b2, q_table
        assert result.requested_shards == 3

        expected = ["yume-00001-of-00003.safetensors",
                    "yume-00002-of-00003.safetensors",
                    "yume-00003-of-00003.safetensors"]
        for i, name in enumerate(expected):
            assert Path(result.shard_files[i]).name == name
            assert (tmp_path / name).is_file()

        # Index follows the HF convention and carries total_size + weight_map.
        assert result.index_path is not None
        assert Path(result.index_path).name == "yume.safetensors.index.json"
        index = json.loads((tmp_path / "yume.safetensors.index.json").read_text())
        assert index["metadata"]["total_size"] == result.total_tensor_bytes
        assert len(index["weight_map"]) == 5
        assert set(index["weight_map"].values()) <= set(expected)

        # Result and file agree on the weight map.
        assert result.weight_map == index["weight_map"]

    def test_shard_payloads_reassemble_to_single_file(self, tmp_path):
        mind = make_trained_mind()
        single = tmp_path / "single.safetensors"
        MindExporter.export_safetensors(mind, str(single))
        sharded = MindExporter.export_safetensors(mind, str(tmp_path / "shard.safetensors"), 3)

        reference = parse_safetensors(single)
        assert len(reference) == 5

        reassembled = {}
        for shard in sharded.shard_files:
            for name, entry in parse_safetensors(shard).items():
                assert name not in reassembled, f"tensor {name} appears in two shards"
                reassembled[name] = entry

        assert set(reassembled) == set(reference)
        for name, (shape, payload) in reference.items():
            assert reassembled[name][0] == shape
            assert reassembled[name][1] == payload            # byte-for-byte identical

    def test_split_larger_than_tensor_count_clamps(self, tmp_path):
        mind = make_trained_mind()
        # The feature-request example: split into 62. Shards hold whole
        # tensors, so the split clamps to the tensor count (5).
        result = MindExporter.export_safetensors(mind, str(tmp_path / "yume.safetensors"), 62)

        assert result.requested_shards == 62
        assert result.written_shards == result.tensor_count == 5
        assert (tmp_path / "yume-00001-of-00005.safetensors").is_file()
        assert (tmp_path / "yume-00005-of-00005.safetensors").is_file()
        assert not (tmp_path / "yume-00006-of-00005.safetensors").exists()

        # No empty shards.
        for shard in result.shard_files:
            assert len(parse_safetensors(shard)) > 0

    def test_split_one_writes_exact_file_without_index(self, tmp_path):
        mind = make_trained_mind()
        path = tmp_path / "plain.safetensors"
        result = MindExporter.export_safetensors(mind, str(path), 1)

        assert not result.is_sharded
        assert result.index_path is None
        assert result.files == (str(path),)
        assert result.primary_file == str(path)
        assert path.is_file()
        assert not (tmp_path / "plain.safetensors.index.json").exists()

    def test_default_call_still_writes_single_file(self, tmp_path):
        mind = make_trained_mind()
        path = tmp_path / "default.safetensors"
        result = MindExporter.export_safetensors(mind, str(path))
        assert path.is_file()
        assert not result.is_sharded

    def test_legacy_positional_options_still_accepted(self, tmp_path):
        mind = make_trained_mind()
        path = tmp_path / "legacy.safetensors"
        result = MindExporter.export_safetensors(mind, str(path), ExportOptions())
        assert path.is_file()
        assert not result.is_sharded

    def test_invalid_split_raises(self, tmp_path):
        mind = make_trained_mind()
        path = str(tmp_path / "yume.safetensors")
        with pytest.raises(ValueError):
            MindExporter.export_safetensors(mind, path, 0)
        with pytest.raises(ValueError):
            MindExporter.export_safetensors(mind, path, -5)

    def test_split_mind_without_tensors_raises(self, tmp_path):
        mind = MindBuilder("Bare").only("thinking", "memory").build()
        mind.process("hello")
        path = tmp_path / "bare.safetensors"
        with pytest.raises(RuntimeError):
            MindExporter.export_safetensors(mind, str(path), 3)

        # Unsplit export of a tensor-less mind still works (metadata-only file).
        result = MindExporter.export_safetensors(mind, str(path), 1)
        assert path.is_file()
        assert result.tensor_count == 0


class TestShardedBundle:
    def test_bundle_with_split_writes_shards(self, tmp_path):
        mind = make_trained_mind()
        out = tmp_path / "bundle"
        files = MindExporter.export_bundle(mind, str(out), ExportOptions(safetensors_split=3))

        assert "yume-00001-of-00003.safetensors" in files
        assert "yume.safetensors.index.json" in files
        assert "yume.safetensors" not in files
        assert (out / "yume-00002-of-00003.safetensors").is_file()
        assert (out / "yume.safetensors.index.json").is_file()

        config = json.loads((out / "config.json").read_text())
        assert config["export_options"]["safetensors_shards"] == 3

        # README mentions the sharded layout.
        assert "yume-00001-of-00003.safetensors" in (out / "README.md").read_text()

    def test_bundle_default_keeps_single_safetensors(self, tmp_path):
        mind = make_trained_mind()
        out = tmp_path / "bundle"
        files = MindExporter.export_bundle(mind, str(out))
        assert "yume.safetensors" in files
        assert not (out / "yume.safetensors.index.json").exists()
