"""Tests for the dual-process mind: private goals, impulse answers, social
appraisal triggers and the epistemic question behavior.

Grounded in appraisal research: OCC (anger = blame + unfairness), Zajonc's
affect primacy (emotion can answer before thinking), and information-gap
theory (surprise/interest drive questions). Mirrors the C# ImpulseTests.
"""
import pytest

from cognify import MindBuilder
from cognify.compose.composer import IMPULSE_RESPONSES, impulse_line
from cognify.emotions.appraisal import appraise, detect_social_triggers


def make_mind():
    return (MindBuilder("Yume").with_seed(42)
            .with_memory_capacity(20).build())


class TestSocialTriggers:
    def test_self_insult_detected(self):
        t = detect_social_triggers("you are stupid and useless")
        assert t["self_insult"] is True

    def test_hate_you_phrase(self):
        t = detect_social_triggers("i hate you")
        assert t["self_insult"] is True

    def test_observed_negativity_is_not_blame(self):
        t = detect_social_triggers("the movie was terrible and boring")
        assert t["self_insult"] is False

    def test_unfairness(self):
        assert detect_social_triggers("that is totally unfair")["unfairness"]

    def test_dismissal(self):
        assert detect_social_triggers("shut up already")["dismissal"]

    def test_surprise_marker(self):
        assert detect_social_triggers("guess what happened today!")["surprise_marker"]

    def test_interest_marker(self):
        assert detect_social_triggers("tell me more about volcanoes")["interest_marker"]

    def test_trigger_deltas_flow_into_appraise(self):
        calm = appraise("the movie was terrible and boring")
        blamed = appraise("you are terrible and boring")
        assert blamed["anger"] > calm["anger"] + 0.2


class TestImpulse:
    def test_blame_answers_before_thinking(self):
        mind = make_mind()
        reply = mind.process("You are stupid and useless. Shut up.")
        assert reply.meta["action"] == "impulse"
        assert reply.text.strip()

    def test_dismissal_is_reflexive(self):
        mind = make_mind()
        reply = mind.process("Whatever. Shut up already.")
        assert reply.meta["action"] == "impulse"

    def test_contradicting_known_knowledge_startles(self):
        mind = make_mind()
        mind.process("Birds are animals. Animals are living things.")
        before = mind.emotions.get("surprise")
        reply = mind.process("Birds are not animals.")
        # The violated expectation both startles...
        assert mind.emotions.get("surprise") > before + 0.2
        # ...and answers before thinking (reflexive impulse).
        assert reply.meta["action"] == "impulse"

    def test_negation_is_not_stored_as_positive_fact(self):
        mind = make_mind()
        mind.process("Birds are animals.")
        mind.process("Birds are not machines.")
        # Python stores negations STRUCTURED (is-not-a), never as the
        # positive claim (bird, is-a, not machine).
        assert mind.kb.get("bird", "is-a", "not machine") is None
        assert mind.kb.get("bird", "is-a", "machine") is None
        assert mind.kb.get("bird", "is-a", "animal") is not None

    def test_calm_topics_never_impulse(self):
        mind = make_mind()
        mind.process("Birds are animals. Animals are living things.")
        reply = mind.process("I like quiet evenings and reading books.")
        assert reply.meta["action"] != "impulse"

    def test_knowledge_wins_over_impulse(self):
        mind = make_mind()
        mind.process("2 + 2 = 4.")
        reply = mind.process("You are stupid. Now tell me, what is 2 + 2?")
        assert reply.meta["action"] != "impulse"
        assert "4" in reply.text

    def test_impulse_is_for_the_surging_turn_only(self):
        mind = make_mind()
        hot = mind.process("You are stupid. Shut up.")
        assert hot.meta["action"] == "impulse"
        # The gate reads the FRESH appraisal each cycle, so once the surge
        # passes, the mind returns to deliberate conversation on its own.
        calm = mind.process("Anyway, how are you doing today?")
        assert calm.meta["action"] != "impulse"

    def test_impulse_lines_exist_for_every_prone_emotion(self):
        for name in ("anger", "surprise", "fear", "joy", "sadness", "disgust"):
            assert len(IMPULSE_RESPONSES[name]) > 0
            line = impulse_line(name, __import__("random").Random(1))
            assert line.strip()


class TestGoalsArePrivate:
    def test_goal_is_never_spoken_aloud(self):
        mind = make_mind()
        mind.goals.add_goal("make the user smile", priority=0.9, drive="connection")
        turns = [
            "I had a long day at work.",
            "The weather is nice today.",
            "I finally finished my project!",
            "tell me about yourself",
            "What should I know about gardening?",
            "just checking in, how are you",
        ]
        for turn in turns:
            reply = mind.process(turn)
            low = reply.text.lower()
            assert "moves my goal" not in low
            assert "make the user smile" not in low
            assert "tracking" not in low

    def test_goal_still_steers_the_inner_prompt(self):
        mind = make_mind()
        mind.goals.add_goal("make the user smile", priority=0.9, drive="connection")
        reply = mind.process("I had a long day at work.")
        thoughts = " ".join(t[1] for t in reply.thoughts)  # (stage, text, conf)
        assert "make the user smile" in thoughts.lower()


class TestEpistemicEmotions:
    def test_shown_interest_becomes_a_question(self):
        mind = make_mind()
        reply = mind.process("tell me more about volcanoes, they seem really interesting")
        assert mind.emotions.get("anticipation") > 0.15
        assert "?" in reply.text

    def test_surprise_marker_sets_up_the_startle(self):
        mind = make_mind()
        mind.process("guess what happened to me today!")
        assert mind.emotions.get("surprise") > 0.25

    def test_epistemic_mode_does_not_latch(self):
        mind = make_mind()
        q1 = mind.process("tell me more about volcanoes")
        assert "?" in q1.text
        # Teach, then repeat the SAME sentence: no fresh interest, no
        # novelty — the forced question mode must pass with its surge
        # instead of latching onto every later turn.
        mind.process("volcanoes are mountains")
        r = mind.process("volcanoes are mountains")
        for phrase in ("You have me curious now", "I am genuinely interested",
                       "what is it?", "what is the whole story"):
            assert phrase not in r.text


class TestVocabularyFiltersSpeechGlue:
    def test_speech_glue_is_not_picked_up(self):
        mind = make_mind()
        mind.process("okay so lets try getting this good job done, slightly slowly, fore the vocab")
        picked = mind.learner.lexicon.last_picks
        for glue in ("okay", "getting", "doing", "good", "job", "lets",
                     "try", "slightly", "vocab", "fore"):
            assert glue not in picked
        assert "slowly" in picked
