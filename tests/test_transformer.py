"""Tests for the transformer language model, the hybrid blend and the
mind-level wiring (persistence included)."""
import math
from dataclasses import replace

import pytest

from cognify.core.mind import Mind
from cognify.learning.hybrid_next_token import fluency, predict_next
from cognify.learning.next_token import NextTokenModel
from cognify.learning.transformer import (
    TransformerConfig,
    TransformerLM,
    TransformerTrainOptions,
)
from cognify.persistence.store import SqliteMindStore

# Deliberately repetitive: the model MUST learn these transitions.
CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "the cat sat on the mat",
    "the dog sat on the rug",
    "the cat ran to the mat",
    "the dog ran to the rug",
]


def tiny_config(seed: int = 7) -> TransformerConfig:
    return TransformerConfig(dim=8, heads=2, layers=1, context_len=6,
                             ffn_mult=2, dropout=0.0, learning_rate=0.02,
                             seed=seed, batch_size=8, max_windows_per_epoch=64)


@pytest.fixture(scope="module")
def trained() -> TransformerLM:
    lm = TransformerLM(tiny_config())
    for line in CORPUS:
        lm.observe(line)
    lm.train(40)
    return lm


class TestUntrained:
    def test_no_opinions(self):
        lm = TransformerLM(tiny_config())
        assert not lm.is_trained
        assert lm.predict_next("the cat") == []
        assert lm.score("the cat sat") == 0.0
        assert lm.generate("the") == ""

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            TransformerLM(TransformerConfig(dim=10, heads=3))


class TestTraining:
    def test_loss_reduces(self, trained):
        assert trained.last_loss < 2.5
        assert trained.is_trained

    def test_predictions_learn_transitions(self, trained):
        # After a bare "the" both continuations exist in this corpus
        # (sentence-start "the cat/dog", mid-sentence "on the mat/rug").
        after_the = trained.predict_next("the", 2)
        assert {p.token for p in after_the} <= {"cat", "dog", "mat", "rug"}

        # Unambiguous mid-sentence context: after "sat on the" only mat/rug.
        after_seat = trained.predict_next("sat on the", 2)
        assert {p.token for p in after_seat} <= {"mat", "rug"}

        # Unambiguous subject context: after "the cat" only sat/ran.
        after_cat = trained.predict_next("the cat", 2)
        assert {p.token for p in after_cat} <= {"sat", "ran"}

        assert 0.99 <= sum(p.probability for p in after_seat) <= 1.01

    def test_no_special_tokens_in_predictions(self, trained):
        for line in CORPUS:
            for p in trained.predict_next(line, 5):
                assert "<" not in p.token

    def test_score_beats_gibberish(self, trained):
        real = trained.score("the cat sat on the mat")
        gibberish = trained.score("zqx wvv kkk jjj qqq")
        assert real > gibberish
        assert real < 0

    def test_generate_uses_known_words(self, trained):
        generated = trained.generate("the cat", max_tokens=6)
        for word in generated.split(" "):
            if word:
                assert "<" not in word

    def test_vocabulary_grows(self):
        lm = TransformerLM(tiny_config())
        before = lm.vocabulary_size  # 4 specials
        lm.observe("the cat sat")
        assert lm.vocabulary_size > before
        assert lm.tokens_observed == 3


class TestPersistence:
    def test_roundtrip_preserves_predictions(self, trained):
        before = trained.predict_next("the cat", 5)
        restored = TransformerLM(tiny_config())
        assert restored.load_dict(trained.to_dict())
        after = restored.predict_next("the cat", 5)
        assert [p.token for p in before] == [p.token for p in after]
        for a, b in zip(before, after):
            assert abs(a.probability - b.probability) < 0.01
        assert restored.is_trained
        assert restored.buffered_sentences == trained.buffered_sentences

    def test_rejects_wrong_architecture(self, trained):
        other = TransformerLM(TransformerConfig(dim=16, heads=2, layers=1,
                                                context_len=6, ffn_mult=2,
                                                dropout=0.0, seed=7))
        assert not other.load_dict(trained.to_dict())
        assert not other.load_dict({})
        assert not other.load_dict({"dim": "banana"})


class TestHybrid:
    def test_untrained_neural_falls_back_to_ngram(self):
        ngram = NextTokenModel()
        for line in CORPUS:
            ngram.observe(line)
        neural = TransformerLM(tiny_config())  # never trained

        hybrid = predict_next(ngram, neural, "the cat", 5)
        plain = ngram.predict_next("the cat", 5)
        assert [p.token for p in hybrid] == [p.token for p in plain]
        assert [p.probability for p in hybrid] == [p.probability for p in plain]
        assert fluency(ngram, neural, "the cat sat") == ngram.score("the cat sat")

    def test_trained_neural_blends(self, trained):
        ngram = NextTokenModel()
        for line in CORPUS:
            ngram.observe(line)
        hybrid = predict_next(ngram, trained, "the cat", 5)
        assert hybrid
        assert 0.99 <= sum(p.probability for p in hybrid) <= 1.01
        f = fluency(ngram, trained, "the cat sat on the mat")
        assert f < 0


class TestMindWiring:
    def _tiny_mind(self) -> Mind:
        from cognify.core.builder import MindBuilder
        mind = MindBuilder("DeepTest").build()
        mind.neural_language_model = TransformerLM(tiny_config())
        return mind

    def test_train_deep_model_powers_prediction(self):
        mind = self._tiny_mind()
        assert not mind.neural_language_model.is_trained
        for line in CORPUS:
            mind.train(line)
        report = mind.train_deep_model(30)
        assert report.epochs == 30
        assert mind.neural_language_model.is_trained
        predictions = mind.predict_next("the cat", 5)
        assert predictions
        assert any(p.token in ("sat", "ran") for p in predictions)
        assert mind.score_fluency("the cat sat on the mat") < 0

    def test_before_deep_training_uses_ngram(self):
        mind = self._tiny_mind()
        for line in CORPUS:
            mind.train(line)
        from_mind = mind.predict_next("the cat", 5)
        from_ngram = mind.language_model.predict_next("the cat", 5)
        assert [p.token for p in from_mind] == [p.token for p in from_ngram]

    def test_snapshot_keeps_transformer(self, tmp_path):
        mind = self._tiny_mind()
        for line in CORPUS:
            mind.train(line)
        mind.train_deep_model(30)
        before = mind.neural_language_model.predict_next("the cat", 3)

        store = SqliteMindStore(str(tmp_path / "minds.db"))
        store.save(mind, "deep-test")
        restored = store.load("deep-test")

        assert restored.neural_language_model.is_trained
        after = restored.neural_language_model.predict_next("the cat", 3)
        assert [p.token for p in before] == [p.token for p in after]
        store.close()


def _observed() -> TransformerLM:
    lm = TransformerLM(tiny_config())
    for line in CORPUS:
        lm.observe(line)
    return lm


def test_train_loop_reports_per_epoch_loss_and_progress():
    lm = _observed()
    seen = []
    report = lm.train(TransformerTrainOptions(
        epochs=6, progress=lambda e, total, loss: seen.append((e, loss))))

    assert len(report.per_epoch_losses) == 6
    assert report.learning_rates == [tiny_config().learning_rate] * 6  # no decay
    assert len(seen) == 6 and seen[-1][0] == 6
    assert abs(seen[-1][1] - report.final_loss) < 1e-9
    assert report.initial_loss > report.final_loss          # training reduces loss
    assert not report.stopped_early
    assert all(loss == loss and abs(loss) < math.inf for loss in report.per_epoch_losses)


def test_train_loop_decays_learning_rate():
    lm = _observed()
    report = lm.train(TransformerTrainOptions(epochs=4, learning_rate=0.01, decay=0.5))
    assert report.learning_rates == pytest.approx([0.01, 0.005, 0.0025, 0.00125])


def test_train_loop_stops_early_when_loss_stalls():
    lm = _observed()
    seen = []
    report = lm.train(TransformerTrainOptions(
        epochs=20, min_delta=10.0, patience=2,
        progress=lambda e, total, loss: seen.append(e)))

    assert report.stopped_early
    assert report.epochs == 3                                # 1 progress epoch + 2 stagnant
    assert len(report.per_epoch_losses) == 3
    assert len(seen) == 3


class TestValidationSplit:
    """train_file(validation_split=...) — 80/20-style hold-out scored with
    the SAME cross-entropy training minimizes, forward-only."""

    def _mind(self) -> Mind:
        from cognify.core.builder import MindBuilder
        mind = MindBuilder("SplitTest").with_memory().with_thinking().build()
        mind.neural_language_model = TransformerLM(tiny_config())
        return mind

    def _corpus_file(self, tmp_path):
        path = tmp_path / "corpus.txt"
        path.write_text("\n".join(CORPUS) + "\n", encoding="utf-8")
        return str(path)

    def test_split_trains_80_and_validates_20(self, tmp_path):
        mind = self._mind()
        path = self._corpus_file(tmp_path)
        mind.train_file(path)                     # plain pass — still everything
        mind.train_deep_model(5)

        report = mind.train_file(path, 0.2, 42)
        assert report.validation_split == 0.2
        assert report.validation_lines == 1       # 6 lines * 0.2 = 1 (floor)
        assert report.lines == 5                  # only the 80% trained
        assert not math.isnan(report.validation_loss)
        assert report.validation_words > 0
        assert "Validation: 1 line(s) held out" in str(report)

    def test_split_is_deterministic_for_a_seed(self, tmp_path):
        mind = self._mind()
        path = self._corpus_file(tmp_path)
        mind.train_file(path)                     # fills the language buffer
        mind.train_deep_model(5)
        # Same seed → same held-out line → same token count and score.
        first = mind.train_file(path, 0.5, 7)
        second = mind.train_file(path, 0.5, 7)
        assert first.validation_words == second.validation_words
        assert first.validation_loss == second.validation_loss
        # A different seed may hold out a different line (different tokens).
        other = mind.train_file(path, 0.5, 9)
        assert other.validation_words >= 0

    def test_no_split_trains_everything(self, tmp_path):
        mind = self._mind()
        path = self._corpus_file(tmp_path)
        report = mind.train_file(path)
        assert report.lines == len(CORPUS)
        assert report.validation_split == 0.0
        assert report.validation_lines == 0
        assert math.isnan(report.validation_loss)
        assert "Validation" not in str(report)

    def test_untrained_model_scores_nan(self, tmp_path):
        mind = self._mind()
        path = self._corpus_file(tmp_path)
        report = mind.train_file(path, 0.2)       # no deep training yet
        assert report.validation_lines == 1
        assert math.isnan(report.validation_loss)
        assert "no score" in str(report)

    def test_evaluate_never_touches_the_weights(self, tmp_path):
        lm = TransformerLM(tiny_config())
        for line in CORPUS:
            lm.observe(line)
        lm.train(5)
        snapshot = lm.to_dict()
        loss, tokens = lm.evaluate(CORPUS)
        assert not math.isnan(loss) and tokens > 0
        # evaluate() leaves the buffer untouched: re-evaluating scores
        # exactly the same number.
        again, again_tokens = lm.evaluate(CORPUS)
        assert again == loss and again_tokens == tokens


class TestGenerationGuard:
    """generate() ends sentences on EOS (like llama.cpp) and never repeats
    a word three times in a row — the degenerate loop tiny models fall
    into."""

    def test_no_triple_repeats(self, trained):
        for prompt in ("the cat", "the", "sat on the", "the dog"):
            words = trained.generate(prompt, max_tokens=30).split()
            for i in range(len(words) - 2):
                assert not (words[i] == words[i + 1] == words[i + 2]), words

    def test_can_end_before_max_tokens(self, trained):
        # With EOS in the sampling distribution a trained model can finish
        # a short sentence early — the reply is often shorter than the cap.
        short_replies = 0
        for _ in range(6):
            words = trained.generate("the cat", max_tokens=24).split()
            if len(words) < 24:
                short_replies += 1
        assert short_replies >= 1


class TestLoopTransformer:
    """Loop Transformer (cfg.loop): one block stack applied N times per
    pass — weight-tied recurrent depth (Universal Transformer / ALBERT
    style). Same parameters, deeper compute; the GGUF export unrolls the
    loop into layers × loop physical blocks."""

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            TransformerLM(replace(tiny_config(), loop=0))
        with pytest.raises(ValueError):
            TransformerLM(replace(tiny_config(), loop=9))
        assert TransformerLM().cfg.loop == 1   # classic default

    def test_looping_adds_no_parameters_and_trains(self):
        plain = TransformerLM(tiny_config())
        looped = TransformerLM(replace(tiny_config(), loop=2))
        for line in CORPUS:
            plain.observe(line)
            looped.observe(line)
        p1, p2 = plain.train(20), looped.train(20)
        assert p2.parameters == p1.parameters   # weight tying: same count
        assert p2.final_loss < p2.initial_loss  # and looping learns
        for word in looped.generate("the cat", 6).split():
            assert "<" not in word

    def test_roundtrip_keeps_loop(self):
        looped = TransformerLM(replace(tiny_config(), loop=3))
        for line in CORPUS:
            looped.observe(line)
        looped.train(10)
        restored = TransformerLM(replace(tiny_config(), loop=3))
        assert restored.load_dict(looped.to_dict())
        assert restored.cfg.loop == 3
        assert restored.is_trained
        assert restored.generate("the cat", 6) != ""
        # Architecture mismatch: a one-pass model refuses a looped snapshot.
        assert not TransformerLM(tiny_config()).load_dict(looped.to_dict())

    def test_builder_hook(self):
        from cognify.core.builder import MindBuilder
        mind = MindBuilder("Looped").with_transformer(
            lambda c: replace(c, layers=1, loop=2)).build()
        assert mind.neural_language_model.cfg.loop == 2
        assert mind.neural_language_model.cfg.layers == 1
