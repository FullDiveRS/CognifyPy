"""Tests for emotions, appraisal, and personality."""
import pytest

from cognify.emotions.appraisal import appraise
from cognify.emotions.emotion_state import (
    EMOTION_NAMES, INTERJECTIONS, NEGATIVE_EMOTIONS, POSITIVE_EMOTIONS,
    EmotionalState,
)
from cognify.emotions.personality import Personality


class TestEmotionalState:
    def test_initially_calm(self):
        state = EmotionalState()
        assert state.dominant() is None
        assert state.valence == 0.0

    def test_apply_and_dominant(self):
        state = EmotionalState()
        state.apply({"joy": 0.8, "surprise": 0.3})
        assert state.dominant() == "joy"
        assert state.valence > 0.5

    def test_negative_valence(self):
        state = EmotionalState()
        state.apply({"fear": 0.9})
        assert state.valence < 0

    def test_clamped(self):
        state = EmotionalState()
        state.apply({"joy": 5.0})
        assert state.get("joy") == 1.0

    def test_decay(self):
        state = EmotionalState()
        state.apply({"joy": 1.0})
        state.decay(0.5)
        assert state.get("joy") == pytest.approx(0.5)

    def test_sensitivity_scales(self):
        a = EmotionalState(sensitivity=1.0)
        b = EmotionalState(sensitivity=2.0)
        a.apply({"joy": 0.4})
        b.apply({"joy": 0.4})
        assert b.get("joy") == pytest.approx(0.8)

    def test_snapshot_roundtrip(self):
        state = EmotionalState()
        state.apply({"anger": 0.4})
        clone = EmotionalState()
        clone.load(state.snapshot())
        assert clone.values == state.values

    def test_arousal_range(self):
        state = EmotionalState()
        state.apply({e: 1.0 for e in EMOTION_NAMES})
        assert state.arousal == pytest.approx(1.0)

    def test_darwin_emotions_are_tracked(self):
        # Plutchik's 8 primaries + Darwin's 29 finer feelings.
        assert len(EMOTION_NAMES) == 37
        state = EmotionalState()
        assert len(state.values) == 37
        for name in ("love", "devotion", "tenderness", "suffering", "weeping",
                     "high spirits", "low spirits", "anxiety", "grief",
                     "dejection", "despair", "hatred", "disdain", "contempt",
                     "guilt", "pride", "helplessness", "patience",
                     "affirmation", "negation", "self attention", "shyness",
                     "modesty", "blushing", "reflection", "meditation",
                     "ill temper", "sulkiness", "determination"):
            assert name in state.values

    def test_every_emotion_has_an_interjection(self):
        # The composer speaks through INTERJECTIONS — none may be missing.
        assert set(EMOTION_NAMES) == set(INTERJECTIONS)

    def test_valence_sets_cover_all_names(self):
        # Every emotion is positive, negative, or one of the valence-neutral
        # feelings (surprise, reflection-style states).
        neutral = {"surprise", "anticipation", "reflection", "meditation",
                   "blushing", "shyness", "self attention"}
        classified = POSITIVE_EMOTIONS | NEGATIVE_EMOTIONS | neutral
        assert classified >= set(EMOTION_NAMES)
        assert not (POSITIVE_EMOTIONS & NEGATIVE_EMOTIONS)


class TestAppraisal:
    def test_pleasant_words_raise_joy(self):
        d = appraise("I love this wonderful thing")
        assert d["joy"] > 0.3
        assert d["fear"] == 0.0

    def test_fear_words(self):
        d = appraise("this is scary and dangerous")
        assert d["fear"] > 0.4

    def test_anger_words(self):
        d = appraise("you betrayed me, that is unfair")
        assert d["anger"] > 0.3

    def test_novelty_drives_surprise(self):
        calm = appraise("the cat sat", novelty=0.0)
        novel = appraise("the cat sat", novelty=0.9)
        assert novel["surprise"] > calm["surprise"]

    def test_goal_relevance_drives_anticipation(self):
        d = appraise("photography sounds nice", goal_relevance=0.9)
        assert d["anticipation"] > 0.2

    def test_exclamation_amplifies(self):
        plain = appraise("this is scary")
        loud = appraise("this is scary!")
        assert loud["fear"] > plain["fear"]

    def test_neuroticism_amplifies_negative(self):
        calm = Personality(neuroticism=0.1)
        jumpy = Personality(neuroticism=0.9)
        d_calm = appraise("this is terrible", personality=calm)
        d_jumpy = appraise("this is terrible", personality=jumpy)
        assert d_jumpy["sadness"] > d_calm["sadness"]

    def test_agreeableness_dampens_anger(self):
        harsh = Personality(agreeableness=0.1)
        kind = Personality(agreeableness=0.95)
        d_harsh = appraise("you betrayed me", personality=harsh)
        d_kind = appraise("you betrayed me", personality=kind)
        assert d_kind["anger"] < d_harsh["anger"]

    def test_empty_text_safe(self):
        d = appraise("")
        assert all(v == 0.0 for v in d.values())


class TestDarwinEmotions:
    """The finer-grained feelings from Darwin's taxonomy."""

    def test_despair_words(self):
        d = appraise("everything feels hopeless, I am in despair")
        assert d["despair"] > 0.3

    def test_determination_words(self):
        d = appraise("I am determined and I will persist")
        assert d["determination"] > 0.2
        assert d["determination"] > d["anticipation"]

    def test_weeping_words(self):
        d = appraise("I am weeping and sobbing")
        assert d["weeping"] > 0.3

    def test_blushing_words(self):
        d = appraise("I blush and feel flushed")
        assert d["blushing"] > 0.3

    def test_hatred_words_are_not_anger(self):
        # Third-person hatred: "I hate you" is appraised as BLAME aimed at
        # the mind (self-insult -> anger), the correct human response since
        # the social-appraisal triggers landed. This test is about the
        # Darwin "hatred" vocabulary specifically.
        d = appraise("watching them filled me with hatred and loathing")
        assert d["hatred"] > d["anger"]
        assert d["hatred"] > 0.3

    def test_hate_you_is_blame_not_observed_hatred(self):
        # "I hate you" directed at the mind provokes ANGER (blame per OCC),
        # stronger than the hatred vocabulary alone.
        d = appraise("I hate you")
        assert d["anger"] > 0.4

    def test_grief_words(self):
        d = appraise("I am grieving, full of sorrow")
        assert d["grief"] > 0.3

    def test_anxiety_words(self):
        d = appraise("I feel uneasy and full of dread")
        assert d["anxiety"] > 0.3

    def test_affirmation_and_negation(self):
        yes = appraise("yes exactly absolutely")
        assert yes["affirmation"] > 0.4
        no = appraise("no, that is wrong")
        assert no["negation"] > 0.2

    def test_love_is_its_own_emotion(self):
        d = appraise("I love this")
        assert d["love"] > 0.3

    def test_high_spirits_positive_low_spirits_negative(self):
        up = appraise("I feel elated and thrilled")
        down = appraise("I feel gloomy and miserable")
        assert up["high spirits"] > 0.3
        assert down["low spirits"] > 0.3

    def test_neuroticism_amplifies_despair(self):
        calm = Personality(neuroticism=0.1)
        jumpy = Personality(neuroticism=0.9)
        d_calm = appraise("I am in despair", personality=calm)
        d_jumpy = appraise("I am in despair", personality=jumpy)
        assert d_jumpy["despair"] > d_calm["despair"]


class TestPersonality:
    def test_defaults_mid(self):
        p = Personality()
        assert p.openness == 0.5 and p.neuroticism == 0.5

    def test_clamped(self):
        p = Personality(2.0, -1.0, 0.5, 0.5, 0.5)
        assert p.openness == 1.0 and p.conscientiousness == 0.0

    def test_from_preset(self):
        p = Personality.from_preset("curious_child")
        assert p.openness == pytest.approx(0.9)

    def test_describe(self):
        text = Personality.from_preset("focused_analyst").describe()
        assert "openness" in text and "high" in text

    def test_dict_roundtrip(self):
        p = Personality(0.1, 0.2, 0.3, 0.4, 0.5)
        assert Personality.from_dict(p.to_dict()).to_dict() == p.to_dict()

    def test_style_hints(self):
        intro = Personality(extraversion=0.1, openness=0.1)
        extro = Personality(extraversion=0.9, openness=0.9)
        assert extro.verbosity > intro.verbosity
