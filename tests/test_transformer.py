"""Tests for the transformer language model, the hybrid blend and the
mind-level wiring (persistence included)."""
import pytest

from cognify.core.mind import Mind
from cognify.learning.hybrid_next_token import fluency, predict_next
from cognify.learning.next_token import NextTokenModel
from cognify.learning.transformer import TransformerConfig, TransformerLM
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
