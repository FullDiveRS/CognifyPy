"""Tests for vocabulary growth: the mind picks up NEW words from the user
while talking, weaves them back into conversation, and coins new sentence
shapes as it goes — mirrors the C# VocabularyTests."""
import pytest

from cognify import Mind, MindBuilder
from cognify.core.gender import FEMALE
from cognify.learning.lexicon import Lexicon, STOP_WORDS, FILLER_WORDS


# ------------------------------------------------------------------ lexicon
def test_lexicon_picks_up_new_words_once_each():
    lexicon = Lexicon()
    first = lexicon.observe("I adopted a wonderful axolotl yesterday", cycle=1)
    assert "axolotl" in first
    assert "adopted" in first
    assert "wonderful" in first

    # Same words again: nothing new.
    second = lexicon.observe("my axolotl is wonderful", cycle=2)
    assert "axolotl" not in second
    assert "wonderful" not in second


def test_lexicon_skips_stopwords_and_short_tokens():
    lexicon = Lexicon()
    picked = lexicon.observe("the and was but is it a an of to my", cycle=1)
    assert picked == []
    assert lexicon.size == 0


def test_lexicon_skips_time_fillers():
    lexicon = Lexicon()
    picked = lexicon.observe("currently I am really very happy today", cycle=1)
    for filler in ("currently", "really", "very", "today"):
        assert filler not in picked


def test_lexicon_fresh_word_is_spoken_once_only():
    lexicon = Lexicon()
    lexicon.observe("quantum pancakes taste surreal", cycle=1)
    fresh = lexicon.fresh_word()
    assert fresh is not None

    lexicon.mark_spoken(fresh)
    # After being woven into a reply, the same word is never offered again.
    assert lexicon.fresh_word() != fresh
    assert lexicon.was_spoken(fresh)


def test_lexicon_roundtrip_state():
    lexicon = Lexicon()
    lexicon.observe("serendipity is my favorite word today", cycle=3)
    lexicon.mark_spoken("serendipity")

    restored = Lexicon()
    restored.load_dict(lexicon.to_dict())

    assert restored.size == lexicon.size
    assert restored.knows("serendipity")
    assert restored.was_spoken("serendipity")
    # Spoken words do not come back as fresh.
    assert restored.fresh_word() != "serendipity"


# ------------------------------------------------------------------ mind
def test_mind_weaves_new_words_into_conversation():
    mind = MindBuilder("Yume").with_gender("girl", FEMALE).build()
    wove = False
    for turn in range(12):
        reply = mind.process("let me tell you about my pet axolotl")
        low = reply.text.lower()
        if ("picked up the word" in low or "brand-new word" in low
                or "vocabulary just grew" in low):
            wove = True
            break
    assert wove, "the mind should weave a freshly learned word into a reply"
    assert mind.learner.lexicon.knows("axolotl")
    assert mind.learner.lexicon.was_spoken("axolotl")


def test_mind_does_not_weave_the_same_word_twice():
    mind = MindBuilder("Yume").with_gender("girl", FEMALE).build()
    woven = 0
    for turn in range(12):
        reply = mind.process("the axolotl says hello again")
        low = reply.text.lower()
        if ("picked up the word" in low and '"axolotl"' in low):
            woven += 1
    assert woven <= 1, "each new word should be woven into at most one reply"


def test_mind_grows_sayings_while_talking():
    mind = MindBuilder("Yume").with_gender("girl", FEMALE).build()
    before = mind.phrases.count
    for _ in range(14):
        mind.process("today we talk about photosynthesis")
    assert mind.phrases.count > before, "self-grown sayings should join the pool"
    assert any("photosynthesis" in s for s in mind.phrases.all())


def test_mind_lexicon_persists_across_snapshot():
    mind = MindBuilder("Yume").with_gender("girl", FEMALE).build()
    mind.process("my grandmother calls this petrichor")
    clone = Mind.from_dict(mind.to_dict())
    assert clone.learner.lexicon.knows("petrichor"), \
        "learned words must survive a save/restore"


def test_mind_uses_they_them_only_for_unknown_people():
    mind = MindBuilder("Yume").with_gender("girl", FEMALE).build()
    used = False
    for _ in range(8):
        reply = mind.process("someone new joined my class today and I want to tell you about them")
        low = reply.text.lower()
        if "they" in low and "them" in low:
            used = True
            break
    assert used, "a person of unknown gender should be described with they/them"
