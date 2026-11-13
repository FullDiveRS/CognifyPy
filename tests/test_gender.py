"""Tests for the gender identity system (pronouns, self-description,
persistence) — mirrors the C# GenderTests.

The gender system offers exactly two identity presets — she/her and he/him —
plus a genderless mode. they/them is reserved for describing OTHER people
whose gender the mind does not know; it can never be a mind's own pronouns.
"""
import pytest

from cognify import (FEMALE, MALE, UNKNOWN_PERSON, MindBuilder, PronounSet,
                     UNSPECIFIED)
from cognify.core.gender import (GenderOptions, is_reserved_label,
                                 parse_pronouns)
from cognify.core.mind import Mind


# ------------------------------------------------------------------ presets
def test_pronoun_presets_carry_correct_words():
    assert (FEMALE.subject, FEMALE.object, FEMALE.possessive,
            FEMALE.possessive_pronoun, FEMALE.reflexive) == \
        ("she", "her", "her", "hers", "herself")
    assert (MALE.subject, MALE.object, MALE.possessive, MALE.reflexive) == \
        ("he", "him", "his", "himself")


def test_unknown_person_is_they_them_and_reserved():
    assert (UNKNOWN_PERSON.subject, UNKNOWN_PERSON.object,
            UNKNOWN_PERSON.possessive, UNKNOWN_PERSON.reflexive) == \
        ("they", "them", "their", "themself")
    assert UNKNOWN_PERSON.label == "they/them"


def test_unspecified_has_no_pronoun_words_at_all():
    # A genderless mind has NO pronoun words: it only ever says "I".
    # they/them must not hide in the unspecified set.
    assert UNSPECIFIED.subject == ""
    assert UNSPECIFIED.object == ""
    assert UNSPECIFIED.possessive == ""
    assert UNSPECIFIED.label == "unspecified"


def test_parse_pronouns_accepts_only_she_her_and_he_him():
    assert parse_pronouns("she/her") is FEMALE
    assert parse_pronouns("female") is FEMALE
    assert parse_pronouns("male") is MALE
    assert parse_pronouns("he/him") is MALE
    # they/them and non-binary labels are NOT settings for the mind's own
    # pronouns — parsing rejects them.
    assert parse_pronouns("they/them") is None
    assert parse_pronouns("non-binary") is None
    assert parse_pronouns("nonbinary") is None
    assert parse_pronouns("xe/xem") is None


def test_reserved_labels_are_detected():
    assert is_reserved_label("they/them")
    assert is_reserved_label("non-binary person")
    assert is_reserved_label("nonbinary")
    assert not is_reserved_label("girl")
    assert not is_reserved_label("she/her")


# ------------------------------------------------------------------ guards
def test_unknown_person_cannot_become_mind_pronouns():
    mind = MindBuilder("Yume").build()
    with pytest.raises(ValueError, match="reserved"):
        mind.set_pronouns(UNKNOWN_PERSON)
    # the mind is unchanged after the rejected attempt
    assert mind.gender.is_unset


def test_reserved_identity_is_rejected():
    mind = MindBuilder("Yume").build()
    with pytest.raises(ValueError, match="reserved"):
        mind.set_gender("they/them")
    with pytest.raises(ValueError, match="reserved"):
        mind.set_gender("non-binary person")
    assert mind.gender.is_unset


def test_custom_pronoun_set_without_label_raises():
    with pytest.raises(ValueError):
        PronounSet("", "xe", "xem", "xir", "xirs", "xemself")


# ------------------------------------------------------------------ builder
def test_with_gender_sets_identity_and_pronouns():
    mind = MindBuilder("Yume").with_gender("girl", FEMALE).build()
    assert mind.gender_identity == "girl"
    assert mind.pronouns == FEMALE
    assert not mind.gender.is_unset


def test_with_gender_preset_label_infers_pronouns():
    mind = MindBuilder("Sol").with_gender("he/him").build()
    assert mind.gender_identity == ""
    assert mind.pronouns == MALE


def test_default_mind_has_no_gender():
    mind = MindBuilder("Plain").build()
    assert mind.gender_identity == ""
    assert mind.pronouns == UNSPECIFIED
    assert mind.gender.is_unset
    assert mind.gender_clause() == ""


def test_with_pronouns_only():
    mind = MindBuilder("Sol").with_pronouns(MALE).build()
    assert mind.gender_identity == ""
    assert mind.pronouns == MALE
    assert mind.gender_clause() == "I go by he/him pronouns."


# ------------------------------------------------------------------ clause
def test_gender_clause_combines_identity_and_pronouns():
    mind = MindBuilder("Yume").with_gender("girl", FEMALE).build()
    assert mind.gender_clause() == "I am a girl, and I go by she/her pronouns."


def test_gender_clause_uses_an_before_vowel():
    mind = MindBuilder("Sol").with_gender("android", MALE).build()
    assert mind.gender_clause() == "I am an android, and I go by he/him pronouns."


def test_gender_clause_keeps_caller_article():
    mind = MindBuilder("Sol").with_gender("a storyteller", MALE).build()
    assert mind.gender_clause().startswith("I am a storyteller")


# ------------------------------------------------------------------ self
def test_describe_self_includes_gender_when_set():
    mind = MindBuilder("Yume").with_gender("girl", FEMALE).build()
    self_text = mind.describe_self()
    assert self_text.startswith("I am Yume")
    assert "I am a girl, and I go by she/her pronouns." in self_text
    # The numeric mood readout is developer-facing introspection: hidden in
    # chat by default, shown only when feelings(True) is configured.
    assert "Right now" not in self_text

    mind.expression.feelings(True)
    assert "Right now" in mind.describe_self()


def test_describe_self_without_gender_unmentioned():
    mind = MindBuilder("Plain").build()
    mind.process("hello")
    self_text = mind.describe_self()
    assert self_text.startswith("I am Plain")
    assert "pronouns" not in self_text


def test_describe_self_never_uses_they_them_for_the_mind():
    # No configuration path may leave the mind describing itself with
    # they/them — those pronouns are for unknown OTHER people only.
    mind = MindBuilder("Yume").build()
    mind.process("hello there")
    low = mind.describe_self().lower()
    assert "they" not in low
    assert "them" not in low


def test_identity_answer_includes_gender_when_asked():
    mind = MindBuilder("Yume").with_gender("girl", FEMALE).build()
    answer = mind.process("Yume, who are you?")
    assert "I am a girl" in answer.text
    assert "she/her" in answer.text


def test_identity_answer_without_gender_never_mentions_gender():
    mind = MindBuilder("Plain").build()
    answer = mind.process("who are you?")
    assert answer.text.startswith("I am Plain")
    assert "pronouns" not in answer.text


# ------------------------------------------------------------------ runtime
def test_set_gender_applies_immediately():
    mind = MindBuilder("Sage").build()
    assert mind.gender.is_unset
    mind.set_gender("boy", MALE)
    assert mind.gender_identity == "boy"
    assert "I am a boy" in mind.describe_self()
    mind.set_pronouns(FEMALE)
    assert mind.pronouns == FEMALE
    assert "she/her" in mind.gender_clause()


def test_set_gender_with_preset_label_only_sets_pronouns():
    mind = MindBuilder("Sage").with_gender("girl", FEMALE).build()
    mind.set_gender("she/her")
    assert mind.gender_identity == "girl"      # identity kept
    assert mind.pronouns == FEMALE


# ------------------------------------------------------------------ snapshot
def test_gender_survives_snapshot_roundtrip():
    mind = MindBuilder("Yume").with_gender("girl", FEMALE).build()
    mind.process("hello")
    data = mind.to_dict()
    assert data["gender"]["identity"] == "girl"

    clone = Mind.from_dict(data)
    assert clone.gender_identity == "girl"
    assert clone.pronouns.label == FEMALE.label
    assert clone.pronouns.reflexive == FEMALE.reflexive
    assert "I am a girl" in clone.describe_self()


def test_gender_set_after_build_persists_to_snapshot():
    mind = MindBuilder("Sage").build()
    mind.set_gender("boy", MALE)
    clone = Mind.from_dict(mind.to_dict())
    assert clone.gender_identity == "boy"
    assert clone.pronouns.label == MALE.label


def test_old_snapshot_without_gender_restores_unset():
    mind = MindBuilder("Legacy").build()
    data = mind.to_dict()
    data.pop("gender", None)  # simulate a pre-gender snapshot
    clone = Mind.from_dict(data)
    assert clone.gender.is_unset
    assert clone.gender_clause() == ""


def test_legacy_snapshot_with_nonbinary_gender_is_sanitized():
    # A snapshot from a version that still offered they/them for the mind
    # itself restores clean: no they/them pronouns, no non-binary identity.
    mind = MindBuilder("Legacy").build()
    data = mind.to_dict()
    data["gender"] = {
        "identity": "non-binary person",
        "pronouns": {"label": "they/them", "subject": "they", "object": "them",
                     "possessive": "their", "possessive_pronoun": "theirs",
                     "reflexive": "themself"},
    }
    clone = Mind.from_dict(data)
    assert clone.pronouns.label != "they/them"
    assert "non-binary" not in clone.gender_identity.lower()


def test_gender_options_roundtrip_dict():
    opts = GenderOptions("girl", FEMALE)
    clone = GenderOptions.from_dict(opts.to_dict())
    assert clone.identity == "girl"
    assert clone.pronouns == FEMALE
    assert GenderOptions.from_dict(None).is_unset
    assert GenderOptions.from_dict({"identity": "boy"}).identity == "boy"
