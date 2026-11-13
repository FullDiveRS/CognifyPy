"""Tests for cognify.export: GGUF v3, safetensors, tokenizer.json, bundle."""
import json
import struct
from pathlib import Path

import pytest

from cognify import MindBuilder
from cognify.export import ExportOptions, MindExporter
from cognify.export.diagnostics import diagnose_gguf
from cognify.export.exporter import (byte_encode, build_bpe_merges,
                                     export_gguf_llama_cpp)
from cognify.export.gguf import InvalidGgufError, read_gguf


def make_trained_mind():
    mind = (MindBuilder("Yume")
            .with_big_five(0.9, 0.6, 0.75, 0.85, 0.3)
            .with_seed(7)
            .build())
    mind.process("My name is Alex.")
    mind.process("Birds are animals. Animals are living things.")
    mind.process("The sky is blue.")
    mind.feedback(0.8)
    mind.process("Thanks, that helped a lot!")   # consumes the reward -> Q update
    mind.feedback(-0.2)
    mind.process("Hmm, not quite what I meant.") # consumes the negative reward
    return mind


def read_gguf_header(path):
    buf = Path(path).read_bytes()
    magic, version, tensor_count, kv_count = struct.unpack_from("<IIQQ", buf, 0)
    return buf, magic, version, tensor_count, kv_count


def gguf_read_string(buf, off):
    (n,) = struct.unpack_from("<Q", buf, off)
    off += 8
    return buf[off:off + n].decode("utf-8"), off + n


class TestGguf:
    def test_magic_and_version(self, tmp_path):
        mind = make_trained_mind()
        path = tmp_path / "yume.gguf"
        MindExporter.export_gguf(mind, str(path))

        buf, magic, version, tensor_count, kv_count = read_gguf_header(path)
        assert magic == 0x46554747                    # "GGUF"
        assert buf[:4] == b"GGUF"
        assert version == 3
        assert tensor_count == 5                      # w1, b1, w2, b2, q_table
        assert kv_count >= 10

    def test_full_walk_kvs_and_tensor_offsets_aligned(self, tmp_path):
        mind = make_trained_mind()
        path = tmp_path / "yume.gguf"
        MindExporter.export_gguf(mind, str(path))

        gguf = read_gguf(str(path))
        assert gguf.metadata["general.architecture"] == "cognify"
        assert gguf.metadata["general.name"] == "yume"
        assert gguf.metadata["general.alignment"] == 32
        assert gguf.metadata["cognify.personality.openness"] == pytest.approx(0.9, abs=1e-6)

        # tensor infos: aligned, RELATIVE to the data section (spec), F32,
        # non-overlapping and fully inside the file
        seen, next_offset = [], 0
        for info in gguf.tensors:
            assert info.ggml_type == 0                        # GGML F32
            assert info.offset % 32 == 0                      # aligned
            assert info.offset >= next_offset                 # non-overlapping
            assert info.shape == tuple(info.shape)            # logical, outer first
            next_offset = info.offset + info.byte_length
            seen.append(info.name)
        assert seen == ["cognify.net.w1", "cognify.net.b1",
                        "cognify.net.w2", "cognify.net.b2", "cognify.rl.q_table"]
        assert gguf.data_start + next_offset <= path.stat().st_size

    def test_tensor_payload_matches_live_network(self, tmp_path):
        mind = make_trained_mind()
        net = mind.learner.net
        path = tmp_path / "yume.gguf"
        MindExporter.export_gguf(mind, str(path))

        gguf = read_gguf(str(path))
        info = gguf.tensor("cognify.net.w1")
        assert info.shape == (net.n_hidden, net.n_in)         # logical shape
        flat = gguf.read_tensor("cognify.net.w1")
        expected = [v for row in net.w1 for v in row]
        for got, want in zip(flat[:16], expected[:16]):
            assert got == pytest.approx(want, abs=1e-6)

    def test_works_without_learning(self, tmp_path):
        mind = MindBuilder("NoLearn").only("thinking", "memory").build()
        mind.process("hello")
        path = tmp_path / "nolearn.gguf"
        MindExporter.export_gguf(mind, str(path))
        gguf = read_gguf(str(path))
        assert len(gguf.tensors) == 0


class TestSafetensors:
    def test_header_and_contiguity(self, tmp_path):
        mind = make_trained_mind()
        path = tmp_path / "yume.safetensors"
        MindExporter.export_safetensors(mind, str(path))

        buf = path.read_bytes()
        (header_len,) = struct.unpack_from("<Q", buf, 0)
        assert header_len % 8 == 0
        header = json.loads(buf[8:8 + header_len])
        assert "__metadata__" in header
        # "pt" is the Hugging Face convention transformers/Unsloth expect.
        assert header["__metadata__"]["format"] == "pt"

        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        assert len(tensors) == 5
        spans = sorted((v["data_offsets"][0], v["data_offsets"][1])
                       for v in tensors.values())
        assert spans[0][0] == 0
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            assert e1 == s2
        assert len(buf) == 8 + header_len + spans[-1][1]

        w1 = header["cognify.net.w1"]
        assert w1["dtype"] == "F32"
        assert w1["shape"] == [mind.learner.net.n_hidden, mind.learner.net.n_in]

    def test_payload_round_trip(self, tmp_path):
        mind = make_trained_mind()
        net = mind.learner.net
        path = tmp_path / "yume.safetensors"
        MindExporter.export_safetensors(mind, str(path))

        buf = path.read_bytes()
        (header_len,) = struct.unpack_from("<Q", buf, 0)
        header = json.loads(buf[8:8 + header_len])
        s, e = header["cognify.net.b2"]["data_offsets"]
        data_start = 8 + header_len
        vals = struct.unpack_from(f"<{(e - s) // 4}f", buf, data_start + s)
        assert len(vals) == net.n_out
        for got, want in zip(vals, net.b2):
            assert got == pytest.approx(want, abs=1e-6)


class TestTokenizer:
    def test_hf_schema_and_corpus_words(self, tmp_path):
        mind = make_trained_mind()
        tok_dir = tmp_path / "tokenizer"
        size = MindExporter.export_tokenizer(mind, str(tok_dir))

        assert size > 5
        tok = json.loads((tok_dir / "tokenizer.json").read_text(encoding="utf-8"))
        assert tok["version"] == "1.0"
        assert tok["normalizer"]["type"] == "Lowercase"
        assert tok["pre_tokenizer"]["type"] == "Whitespace"
        model = tok["model"]
        assert model["type"] == "WordPiece"
        assert model["unk_token"] == "[UNK]"

        vocab = model["vocab"]
        assert vocab["[PAD]"] == 0
        assert vocab["[UNK]"] == 1
        for word in ("birds", "animals", "living"):
            assert word in vocab, f"corpus word {word!r} missing"

        ids = sorted(vocab.values())
        assert ids == list(range(len(ids)))               # dense ids

        # companion files
        plain = json.loads((tok_dir / "vocab.json").read_text(encoding="utf-8"))
        assert plain == vocab
        specials = json.loads((tok_dir / "special_tokens_map.json").read_text(encoding="utf-8"))
        assert len(specials) == 5


class TestBundle:
    def test_complete_bundle_with_llamacpp_configs(self, tmp_path):
        mind = make_trained_mind()
        out = tmp_path / "bundle"
        files = MindExporter.export_bundle(
            mind, str(out), ExportOptions(model_name="Yume", epochs=2))

        for expected in ("yume.gguf", "yume.safetensors", "config.json", "README.md",
                         "train_config.json", "finetune.sh", "quantize.sh",
                         "training_data.txt", "val_data.txt",
                         "tokenizer/tokenizer.json", "tokenizer/vocab.json",
                         "tokenizer/special_tokens_map.json"):
            assert (out / expected).exists(), f"missing {expected}"
        assert "yume.gguf" in files

        finetune = (out / "finetune.sh").read_text(encoding="utf-8")
        assert "llama-finetune" in finetune
        assert "yume-llamacpp.gguf" in finetune   # the runnable export, not the archive
        assert "--lora-out" in finetune
        # Regression: doubled backslashes are an escaped backslash for bash —
        # NOT a line continuation — so the trainer call must use singles.
        assert not any(line.endswith("\\\\") for line in finetune.splitlines())
        assert "\\\n" in finetune
        assert "llama-cli --model $MODEL --lora $LORA_OUT" in finetune
        import subprocess
        for script in ("finetune.sh", "quantize.sh"):
            result = subprocess.run(["bash", "-n", str(out / script)],
                                    capture_output=True, text=True)
            assert result.returncode == 0, result.stderr

        quantize = (out / "quantize.sh").read_text(encoding="utf-8")
        assert "llama-quantize" in quantize
        assert "yume-llamacpp.gguf" in quantize          # runnable source, not the archive
        assert "yume.Q4_K_M.gguf" in quantize
        assert "Q8_0" in quantize

        cfg = json.loads((out / "train_config.json").read_text(encoding="utf-8"))
        assert cfg["model"] == "yume-llamacpp.gguf"
        assert cfg["train_data"] == "training_data.txt"
        assert cfg["epochs"] == 2
        assert cfg["quantization"]["target"] == "yume.Q4_K_M.gguf"

        training = (out / "training_data.txt").read_text(encoding="utf-8").lower()
        assert "birds" in training
        assert "animals" in training

        arch = json.loads((out / "config.json").read_text(encoding="utf-8"))
        assert arch["model_type"] == "cognify-mind"
        assert arch["tensor_count"] == 5

    def test_bundle_without_learning_module(self, tmp_path):
        mind = MindBuilder("Bare").only("thinking", "memory").build()
        mind.process("just words here")
        out = tmp_path / "bundle"
        files = MindExporter.export_bundle(mind, str(out))
        assert (out / "bare.gguf").exists()
        assert (out / "bare.safetensors").exists()
        assert len(files) >= 10


class TestSanitize:
    def test_sanitize(self):
        assert MindExporter.sanitize("Yume!") == "yume"
        assert MindExporter.sanitize("My AI  V2") == "my-ai-v2"
        assert MindExporter.sanitize("   ") == "cognify-model"


class TestLlamaCppGguf:
    """The llama.cpp-runnable gpt2 export (tensor names, KVs, tokenizer)."""

    @staticmethod
    def make_mind_with_trained_transformer():
        mind = make_trained_mind()
        report = mind.train_deep_model(epochs=1)
        assert report.epochs >= 1
        return mind

    def test_requires_trained_model(self, tmp_path):
        mind = MindBuilder("Fresh").only("thinking", "memory").build()
        mind.process("hello there")
        with pytest.raises(RuntimeError):
            export_gguf_llama_cpp(mind, str(tmp_path / "x.gguf"))

    def test_gpt2_architecture_kvs_and_tensor_set(self, tmp_path):
        mind = self.make_mind_with_trained_transformer()
        neural = mind.neural_language_model
        cfg = neural.cfg
        path = tmp_path / "yume-llamacpp.gguf"
        result = export_gguf_llama_cpp(mind, str(path))

        assert result.tensor_count == 2 + cfg.layers * 12 + 2
        gguf = read_gguf(str(path))
        assert gguf.metadata["general.architecture"] == "gpt2"
        assert gguf.metadata["gpt2.context_length"] == cfg.context_len
        assert gguf.metadata["gpt2.embedding_length"] == cfg.dim
        assert gguf.metadata["gpt2.block_count"] == cfg.layers
        assert gguf.metadata["gpt2.attention.head_count"] == cfg.heads
        assert gguf.metadata["gpt2.attention.layer_norm_epsilon"] == pytest.approx(1e-5, abs=1e-8)
        assert gguf.metadata["tokenizer.ggml.model"] == "gpt2"

        # exactly the llama.cpp gpt2 tensor set (output head tied -> absent)
        expected = ["token_embd.weight", "position_embd.weight"]
        for l in range(cfg.layers):
            expected += [f"blk.{l}.attn_norm.weight", f"blk.{l}.attn_norm.bias",
                         f"blk.{l}.attn_qkv.weight", f"blk.{l}.attn_qkv.bias",
                         f"blk.{l}.attn_output.weight", f"blk.{l}.attn_output.bias",
                         f"blk.{l}.ffn_norm.weight", f"blk.{l}.ffn_norm.bias",
                         f"blk.{l}.ffn_up.weight", f"blk.{l}.ffn_up.bias",
                         f"blk.{l}.ffn_down.weight", f"blk.{l}.ffn_down.bias"]
        expected += ["output_norm.weight", "output_norm.bias"]
        assert [t.name for t in gguf.tensors] == expected

        # shapes: logical, outer first — [V, D] embeddings, [3D, D] fused qkv
        vocab_size = neural.vocabulary_size
        assert gguf.tensor("token_embd.weight").shape == (vocab_size, cfg.dim)
        assert gguf.tensor("position_embd.weight").shape == (cfg.context_len, cfg.dim)
        assert gguf.tensor("blk.0.attn_qkv.weight").shape == (cfg.dim * 3, cfg.dim)
        assert gguf.tensor("blk.0.attn_qkv.bias").shape == (cfg.dim * 3,)
        assert "output.weight" not in {t.name for t in gguf.tensors}

        # tokenizer: tokens count == embedding rows, specials kept verbatim,
        # word pieces byte-unicode with a leading Ġ
        tokens = gguf.metadata["tokenizer.ggml.tokens"]
        assert len(tokens) == vocab_size
        assert tokens[:4] == ["<pad>", "<unk>", "<bos>", "<eos>"]
        assert tokens[4].startswith("Ġ")
        assert gguf.metadata["tokenizer.ggml.bos_token_id"] == 2
        assert gguf.metadata["tokenizer.ggml.eos_token_id"] == 3
        assert gguf.metadata["tokenizer.ggml.unknown_token_id"] == 1
        assert gguf.metadata["tokenizer.ggml.padding_token_id"] == 0

        # merges rebuild each vocabulary piece from single characters
        merges = gguf.metadata["tokenizer.ggml.merges"]
        assert merges, "merges must not be empty (llama.cpp requires them)"
        first_piece = tokens[4]
        chars = list(first_piece)
        while len(chars) > 1:
            assert chars[0] + " " + chars[1] in merges
            chars[0] += chars[1]
            del chars[1]
        assert "".join(chars) == first_piece

        # data intact: embeddings readable and finite
        emb = gguf.read_tensor("token_embd.weight")
        assert len(emb) == vocab_size * cfg.dim
        assert all(v == v for v in emb[:64])   # no NaNs

    def test_fused_qkv_matches_separate_weights(self, tmp_path):
        mind = self.make_mind_with_trained_transformer()
        path = tmp_path / "yume-llamacpp.gguf"
        export_gguf_llama_cpp(mind, str(path))
        gguf = read_gguf(str(path))
        fused = gguf.read_tensor("blk.0.attn_qkv.weight")
        cfg = mind.neural_language_model.cfg
        by_name = {name: (matrix, vector)
                   for name, matrix, vector in mind.neural_language_model.export_tensors()}
        wq, _ = by_name["cognify.transformer.l0.attn.wq"]
        wk, _ = by_name["cognify.transformer.l0.attn.wk"]
        wv, _ = by_name["cognify.transformer.l0.attn.wv"]
        flat_q = [v for row in wq for v in row]
        flat_k = [v for row in wk for v in row]
        flat_v = [v for row in wv for v in row]
        assert len(flat_q) == cfg.dim * cfg.dim
        assert fused[:len(flat_q)] == pytest.approx(flat_q, abs=1e-6)
        assert fused[len(flat_q):len(flat_q) + len(flat_k)] == pytest.approx(flat_k, abs=1e-6)
        assert fused[len(flat_q) + len(flat_k):] == pytest.approx(flat_v, abs=1e-6)

    def test_byte_encode_and_merges_units(self):
        assert byte_encode("world", True) == "Ġworld"
        assert byte_encode(" world", True) == "Ġworld"   # no double Ġ
        assert byte_encode("a", False) == "a"
        assert byte_encode("naïve", True) == "ĠnaÃ¯ve"   # UTF-8 bytes mapped
        merges = build_bpe_merges(["Ġhello", "Ġhi"])
        assert merges == ["Ġ h", "Ġh e", "Ġhe l", "Ġhel l", "Ġhell o", "Ġh i"]

    def test_bundle_includes_llamacpp_gguf(self, tmp_path):
        mind = self.make_mind_with_trained_transformer()
        out = tmp_path / "bundle"
        files = MindExporter.export_bundle(mind, str(out))
        assert "yume-llamacpp.gguf" in files
        gguf = read_gguf(str(out / "yume-llamacpp.gguf"))
        assert gguf.metadata["general.architecture"] == "gpt2"


class TestGgufDiagnostics:
    """The anti-"Failed to load model" tooling: which quant types a GGUF
    contains, whether a stock llama.cpp build can load it, and what to do
    when it cannot."""

    @staticmethod
    def write_foreign_quant_gguf(path, type_id):
        """Minimal structurally-valid GGUF v3 (arch=gpt2) with ONE tensor of
        the given ggml type id — payload filler, never decoded."""
        import struct as _s
        buf = _s.pack("<IIQQ", 0x46554747, 3, 1, 1)          # magic/version/tensors/kvs
        key = "general.architecture".encode()
        buf += _s.pack("<Q", len(key)) + key + _s.pack("<I", 8)
        val = b"gpt2"
        buf += _s.pack("<Q", len(val)) + val
        name = b"blk.0.attn_qkv.weight"
        buf += _s.pack("<Q", len(name)) + name
        buf += _s.pack("<I", 1) + _s.pack("<Q", 32)          # n_dims, ne=[32]
        buf += _s.pack("<I", type_id) + _s.pack("<Q", 0)     # ggml type, offset
        pad = (-len(buf)) % 32
        buf += b"\x00" * pad + b"\xab" * 104                 # aligned payload filler
        Path(path).write_bytes(buf)

    def test_vouches_for_the_runnable_gpt2_export(self, tmp_path):
        mind = TestLlamaCppGguf.make_mind_with_trained_transformer()
        path = tmp_path / "ok.gguf"
        export_gguf_llama_cpp(mind, str(path))               # self-check must pass
        report = diagnose_gguf(str(path))
        assert report.spec_valid
        assert report.loadable_by_stock_llama_cpp
        assert report.architecture == "gpt2"
        assert report.issues == []
        assert report.advice == []
        assert all(name == "F32" for _, name, _ in report.tensor_types)

    def test_explains_the_cognify_archive(self, tmp_path):
        mind = TestLlamaCppGguf.make_mind_with_trained_transformer()
        path = tmp_path / "archive.gguf"
        MindExporter.export_gguf(mind, str(path))
        report = diagnose_gguf(str(path))
        assert report.spec_valid
        assert report.architecture == "cognify"
        assert not report.loadable_by_stock_llama_cpp        # by design
        assert "archive" in report.verdict
        assert any("llamacpp.gguf" in hint for hint in report.advice)

    def test_flags_foreign_quant_types_with_honest_advice(self, tmp_path):
        # MXFP4 (type 39): the quant class publisher forks ship and older
        # llama.cpp builds cannot decode — the user's exact load failure.
        path = tmp_path / "mxfp4.gguf"
        self.write_foreign_quant_gguf(path, 39)
        report = diagnose_gguf(str(path))
        assert report.spec_valid
        assert not report.loadable_by_stock_llama_cpp
        assert any(tid == 39 and name == "MXFP4" for tid, name, _ in report.tensor_types)
        assert any("MXFP4" in issue for issue in report.issues)
        assert any("newer llama.cpp" in a and "Pick another quant" in a
                   for a in report.advice)

    def test_names_unknown_quant_ids_as_unknown(self, tmp_path):
        path = tmp_path / "future.gguf"
        self.write_foreign_quant_gguf(path, 200)
        report = diagnose_gguf(str(path))
        assert any(tid == 200 and "unknown" in name for tid, name, _ in report.tensor_types)
        assert not report.loadable_by_stock_llama_cpp

    def test_rejects_structurally_broken_files(self, tmp_path):
        mind = TestLlamaCppGguf.make_mind_with_trained_transformer()
        path = tmp_path / "truncated.gguf"
        export_gguf_llama_cpp(mind, str(path))
        size = path.stat().st_size
        with open(path, "r+b") as fh:
            fh.truncate(size // 2)
        report = diagnose_gguf(str(path))
        assert not report.spec_valid
        assert not report.loadable_by_stock_llama_cpp
        assert any("past the end" in issue for issue in report.issues)
