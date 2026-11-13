"""Tests for the n-gram next-token prediction model."""
import pytest

from cognify.core.builder import MindBuilder
from cognify.learning.next_token import NextTokenModel


def test_observe_then_predict_tops_the_taught_continuation():
    model = NextTokenModel()
    for _ in range(10):
        model.observe("birds are animals. birds can fly. birds have wings.")

    predictions = model.predict_next("birds are", 5)
    assert predictions
    assert predictions[0].token == "animals"
    assert predictions[0].probability > 0.5


def test_predict_falls_back_to_bigram():
    model = NextTokenModel()
    for _ in range(10):
        model.observe("the cat sat on the mat")

    # "sat on" was observed, but never "zebra sat on" — bigram backoff.
    predictions = model.predict_next("zebra sat", 5)
    assert any(p.token == "on" for p in predictions)


def test_predict_always_returns_something():
    model = NextTokenModel()
    model.observe("hello world")
    predictions = model.predict_next("completely unseen xylophone context", 3)
    assert predictions  # unigram fallback


def test_score_prefers_text_similar_to_observed():
    model = NextTokenModel()
    for _ in range(10):
        model.observe("birds are animals. birds can fly.")
    seen = model.score("birds are animals")
    unseen = model.score("quantum kaleidoscopes negotiate ferociously")
    assert seen > unseen


def test_untrained_model_has_no_opinion():
    model = NextTokenModel()
    assert model.score("anything at all") == 0.0
    assert not model.is_trained
    assert model.predict_next("anything", 3) == []


def test_serialize_roundtrip():
    model = NextTokenModel()
    for _ in range(6):
        model.observe("birds are animals. birds can fly.")
    clone = NextTokenModel()
    clone.load_dict(model.to_dict())
    assert clone.total_tokens == model.total_tokens
    assert clone.vocabulary == model.vocabulary
    assert clone.predict_next("birds are", 3)[0].token == \
           model.predict_next("birds are", 3)[0].token


def test_merge_adds_counts():
    a, b = NextTokenModel(), NextTokenModel()
    a.observe("birds are animals")
    b.observe("birds are animals")
    a.merge(b)
    assert a.total_tokens == 6
    assert a.predict_next("birds are", 1)[0].token == "animals"


def test_mind_observes_conversation_and_training():
    mind = MindBuilder("Talker").with_preset("curious_child").build()
    assert not mind.language_model.is_trained
    mind.train("birds are animals. birds can fly. birds have wings.")
    mind.ask("what are birds?")
    assert mind.language_model.is_trained
    assert mind.language_model.total_tokens > 0
    assert mind.predict_next("birds are", 3)[0].token == "animals"


def test_language_model_persists_through_snapshot():
    from cognify.persistence.store import SqliteMindStore
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "snap.db")
    store = SqliteMindStore(path)
    mind = MindBuilder("Sleepy").with_preset("curious_child").build()
    mind.train("birds are animals. birds can fly.")
    store.save(mind, "sleepy")
    clone = store.load("sleepy")
    assert clone.language_model.is_trained
    assert clone.predict_next("birds are", 3)[0].token == "animals"
    store.close()
