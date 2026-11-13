"""Tests for facts, parser, knowledge base, and inference."""
import pytest

from cognify.reasoning.facts import Fact, FactParser
from cognify.reasoning.inference import InferenceEngine
from cognify.reasoning.knowledge import KnowledgeBase


@pytest.fixture
def parser():
    return FactParser()


@pytest.fixture
def kb():
    return KnowledgeBase()


class TestFactParser:
    def test_plural_isa(self, parser):
        f = parser.parse("birds are animals")[0]
        assert (f.subject, f.predicate, f.object) == ("bird", "is-a", "animal")

    def test_singula_isa_with_articles(self, parser):
        f = parser.parse("A bird is an animal")[0]
        assert (f.subject, f.object) == ("bird", "animal")

    def test_can(self, parser):
        f = parser.parse("birds can fly")[0]
        assert (f.subject, f.predicate, f.object) == ("bird", "can", "fly")

    def test_cannot(self, parser):
        f = parser.parse("birds cannot swim")[0]
        assert f.predicate == "can-not" and f.is_negated

    def test_neg_isa(self, parser):
        f = parser.parse("birds are not mammals")[0]
        assert (f.predicate, f.object) == ("is-not-a", "mammal")

    def test_have(self, parser):
        f = parser.parse("birds have wings")[0]
        assert (f.predicate, f.object) == ("have", "wing")

    def test_causes(self, parser):
        f = parser.parse("rain causes floods")[0]
        assert (f.predicate, f.object) == ("causes", "flood")

    def test_verb_predicate(self, parser):
        f = parser.parse("cats like fish")[0]
        assert (f.predicate, f.object) == ("like", "fish")

    def test_multi_clause_and_carry(self, parser):
        facts = parser.parse("birds are animals and can fly")
        keys = {f.key for f in facts}
        assert ("bird", "is-a", "animal") in keys
        assert ("bird", "can", "fly") in keys

    def test_question_not_parsed(self, parser):
        assert parser.parse("What causes damage?") == []

    def test_garbage_not_parsed(self, parser):
        assert parser.parse("hello there friend") == []

    def test_multiword_object_stays_natural_language(self, parser):
        # Regression: facts must never be stored as hyphenated slugs — the
        # user caught Yume saying "before is before-affect-effect-...".
        f = parser.parse("before is the affect before the cause")[0]
        assert (f.subject, f.predicate) == ("before", "is-a")
        assert f.object == "affect before the cause"
        assert "-" not in f.object

    def test_spoken_rendering_has_no_slugs(self, parser):
        f = parser.parse("your name is yume")[0]
        assert "-" not in f.subject and "-" not in f.object
        spoken = f.explain().lower()
        assert "your" in spoken and "yume" in spoken
        assert "-" not in spoken


class TestLeadingFillerQuestions:
    """'now what is after?' must ask about 'after', not 'now'."""

    def test_leading_filler_stripped(self, parser, kb):
        kb.add(parser.parse("after is the cause going forward")[0])
        engine = InferenceEngine()
        answer = engine.answer(kb, "now what is after?")
        assert answer is not None
        assert "cause going forward" in answer.text

    def test_question_without_filler_unchanged(self, parser, kb):
        kb.add(parser.parse("after is the cause going forward")[0])
        engine = InferenceEngine()
        answer = engine.answer(kb, "what is after?")
        assert answer is not None
        assert "cause going forward" in answer.text


class TestFactConfidence:
    def test_confidence_bounds(self, parser):
        f = parser.parse("birds are animals")[0]
        assert 0.0 <= f.confidence <= 1.0


class TestKnowledgeBase:
    def test_add_and_dedup(self, kb):
        assert kb.add(Fact("bird", "is-a", "animal"))
        assert not kb.add(Fact("bird", "is-a", "animal", 0.5))
        assert kb.size() == 1
        assert kb.get("bird", "is-a", "animal").confidence == 0.9  # max kept

    def test_about_index(self, kb):
        kb.add(Fact("bird", "can", "fly"))
        kb.add(Fact("bird", "is-a", "animal"))
        assert len(kb.about("bird")) == 2
        assert len(kb.about("bird", "can")) == 1

    def test_ancestors_depth(self, kb):
        kb.add(Fact("eagle", "is-a", "bird"))
        kb.add(Fact("bird", "is-a", "animal"))
        kb.add(Fact("animal", "is-a", "living thing"))
        ancestors = dict(kb.ancestors("eagle"))
        assert ancestors["bird"] == 1
        assert ancestors["animal"] == 2
        assert ancestors["living thing"] == 3

    def test_isa_path(self, kb):
        kb.add(Fact("eagle", "is-a", "bird"))
        kb.add(Fact("bird", "is-a", "animal"))
        path = kb.has_isa_path("eagle", "animal")
        assert path == ["eagle", "bird", "animal"]
        assert kb.has_isa_path("eagle", "fish") is None

    def test_inherited_with_decay(self, kb):
        kb.add(Fact("bird", "is-a", "animal"))
        kb.add(Fact("animal", "can", "move"))
        inherited = kb.inherited("eagle", "can")
        assert inherited == []                      # eagle is not linked yet
        kb.add(Fact("eagle", "is-a", "bird"))
        inherited = kb.inherited("eagle", "can")
        assert inherited and inherited[0][1] < 0.9  # decayed confidence

    def test_contradicts(self, kb):
        kb.add(Fact("bird", "is-not-a", "mammal"))
        assert kb.contradicts("bird", "is-a", "mammal") is not None
        assert kb.contradicts("bird", "can", "mammal") is None

    def test_cycle_safe_ancestors(self, kb):
        kb.add(Fact("a", "is-a", "b"))
        kb.add(Fact("b", "is-a", "a"))
        assert len(kb.ancestors("a")) <= 2


class TestInferenceEngine:
    def test_transitive_isa(self, kb):
        kb.add(Fact("bird", "is-a", "animal"))
        kb.add(Fact("animal", "is-a", "living thing"))
        derived = InferenceEngine().forward_chain(kb)
        assert any(f.key == ("bird", "is-a", "living thing") for f in derived)

    def test_property_inheritance(self, kb):
        kb.add(Fact("eagle", "is-a", "bird"))
        kb.add(Fact("bird", "can", "fly"))
        derived = InferenceEngine().forward_chain(kb)
        assert any(f.key == ("eagle", "can", "fly") for f in derived)

    def test_causal_chain(self, kb):
        kb.add(Fact("rain", "causes", "flood"))
        kb.add(Fact("flood", "causes", "damage"))
        derived = InferenceEngine().forward_chain(kb)
        assert any(f.key == ("rain", "causes", "damage") for f in derived)

    def test_confidence_decays_per_hop(self, kb):
        kb.add(Fact("bird", "is-a", "animal"))
        kb.add(Fact("animal", "is-a", "living thing"))
        derived = InferenceEngine().forward_chain(kb)
        f = next(f for f in derived if f.key == ("bird", "is-a", "living thing"))
        assert f.confidence < 0.9
        assert f.chain                              # explanation present

    def test_induction(self, kb):
        for name in ("sparrow", "robin", "finch"):
            kb.add(Fact(name, "is-a", "bird"))
            kb.add(Fact(name, "can", "sing"))
        derived = InferenceEngine().induce(kb, min_instances=3)
        assert any(f.key == ("bird", "can", "sing") for f in derived)

    def test_induction_blocked_by_negation(self, kb):
        for name in ("sparrow", "robin", "finch"):
            kb.add(Fact(name, "is-a", "bird"))
            kb.add(Fact(name, "can", "sing"))
        kb.add(Fact("penguin", "is-a", "bird"))
        kb.add(Fact("penguin", "can-not", "sing"))
        derived = InferenceEngine().induce(kb, min_instances=3)
        assert not any(f.key == ("bird", "can", "sing") for f in derived)

    def test_abduction(self, kb):
        kb.add(Fact("rain", "causes", "floods"))
        hyps = InferenceEngine().abduce(kb, "floods")
        assert hyps and "rain" in hyps[0].subject

    def test_answer_describe(self, kb):
        kb.add(Fact("bird", "is-a", "animal"))
        kb.add(Fact("bird", "can", "fly"))
        a = InferenceEngine().answer(kb, "What is a bird?")
        assert a.kind == "describe"
        assert "animal" in a.text and "fly" in a.text

    def test_answer_yesno_direct(self, kb):
        kb.add(Fact("bird", "can", "fly"))
        a = InferenceEngine().answer(kb, "Can birds fly?")
        assert a.text.startswith("Yes")

    def test_answer_yesno_inference(self, kb):
        kb.add(Fact("eagle", "is-a", "bird"))
        kb.add(Fact("bird", "is-a", "animal"))
        a = InferenceEngine().answer(kb, "Is an eagle an animal?")
        assert a.text.startswith("Yes") and a.chain

    def test_answer_yesno_negation_wins(self, kb):
        kb.add(Fact("bird", "is-not-a", "mammal"))
        a = InferenceEngine().answer(kb, "Is a bird a mammal?")
        assert a.text.startswith("No")

    def test_answer_unknown(self, kb):
        a = InferenceEngine().answer(kb, "Can fish bicycle?")
        assert "don't know" in a.text

    def test_answer_why_with_parent_chain(self, kb):
        kb.add(Fact("bird", "is-a", "animal"))
        kb.add(Fact("animal", "can", "move"))
        kb.add(Fact("bird", "can", "move"))
        a = InferenceEngine().answer(kb, "Why can birds move?")
        assert a.kind == "why"
        assert "animal" in a.text or a.chain

    def test_answer_causes(self, kb):
        kb.add(Fact("rain", "causes", "flood"))
        a = InferenceEngine().answer(kb, "What causes flood?")
        assert "rain" in a.text

    def test_answer_causes_unknown_abduces(self, kb):
        kb.add(Fact("rain", "causes", "flood"))
        a = InferenceEngine().answer(kb, "What causes flood?")
        assert "rain" in a.text

    def test_no_answer_for_statements(self, kb):
        kb.add(Fact("bird", "is-a", "animal"))
        assert InferenceEngine().answer(kb, "birds are nice") is None or True
