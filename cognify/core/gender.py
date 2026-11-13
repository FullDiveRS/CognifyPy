"""Gender identity for a mind: how it describes itself and which pronouns
others use for it.

Exactly two identity presets exist — FEMALE (she/her) and MALE (he/him) —
plus a genderless mode (UNSPECIFIED). A mind with no gender configured
behaves exactly as before: first-person "I" is genderless, and gender is
never volunteered in conversation.

"they/them" is deliberately NOT an option for a mind's own pronouns: it is
reserved for describing OTHER people whose gender the mind does not know
(UNKNOWN_PERSON).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PronounSet:
    """The words other people use to talk ABOUT the mind ("she is curious",
    "tell her", "that is hers"). Configuration code should pick one of the
    presets (FEMALE / MALE) or leave the mind genderless. All fields are
    lowercase."""

    label: str                      # display label, e.g. "she/her"
    subject: str = ""               # "she is curious"
    object: str = ""                # "talk to her"
    possessive: str = ""            # "her name" (possessive determiner)
    possessive_pronoun: str = ""    # "that book is hers" ("" -> determiner)
    reflexive: str = ""             # "she taught herself" ("" -> subject+self)

    def __post_init__(self) -> None:
        if not (self.label or "").strip():
            raise ValueError("a pronoun set needs a label")
        has_words = bool(self.subject.strip())
        object_ = (self.object or "").strip() or self.subject.strip()
        possessive = (self.possessive or "").strip() or self.subject.strip()
        poss_pron = (self.possessive_pronoun or "").strip() or possessive
        refl = (self.reflexive or "").strip() or (f"{self.subject.strip()}self" if has_words else "")
        object.__setattr__(self, "label", self.label.strip().lower())
        object.__setattr__(self, "subject", self.subject.strip().lower())
        object.__setattr__(self, "object", object_.lower())
        object.__setattr__(self, "possessive", possessive.lower())
        object.__setattr__(self, "possessive_pronoun", poss_pron.lower())
        object.__setattr__(self, "reflexive", refl.lower())

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.label


#: She / her / her / hers / herself.
FEMALE = PronounSet("she/her", "she", "her", "her", "hers", "herself")

#: He / him / his / his / himself.
MALE = PronounSet("he/him", "he", "him", "his", "his", "himself")

#: They / them / their / theirs / themself — RESERVED for describing a third
#: person whose gender the mind does not know ("someone is at the door — they
#: are waiting"). Never valid as a mind's own pronoun set.
UNKNOWN_PERSON = PronounSet("they/them", "they", "them", "their", "theirs", "themself")

#: Pronouns left unspecified — the mind has NO pronoun words at all and
#: speaks first-person "I" only.
UNSPECIFIED = PronounSet("unspecified")

#: Pronoun sets a mind may adopt for itself (she/her, he/him, or none).
_ALLOWED_SELF_SETS = (FEMALE, MALE, UNSPECIFIED)


def parse_pronouns(label: str):
    """Parses "she/her" style labels to a preset when one matches. Only the
    two identity presets parse — "they/them" and non-binary labels return
    ``None`` because they are not settings for a mind's own pronouns."""
    key = (label or "").strip().lower()
    if key in ("she/her", "her/hers", "female", "f"):
        return FEMALE
    if key in ("he/him", "him/his", "male", "m"):
        return MALE
    return None


def is_reserved_label(label: str) -> bool:
    """True for pronoun/identity labels a mind may not adopt for itself
    (they/them and non-binary variants, including phrases embedding them)."""
    key = (label or "").strip().lower()
    if not key:
        return False
    return (key == "nb" or key == "enby"
            or "non-binary" in key or "nonbinary" in key
            or "they/them" in key or "them/their" in key
            or "they them" in key)


def validate_self_pronouns(p: PronounSet) -> None:
    """Raises ValueError when *p* may not be used as a mind's OWN pronoun set:
    they/them is reserved for describing unknown people, and only the
    she/her / he/him presets (or none at all) are offered."""
    if p == UNKNOWN_PERSON or p.label == "they/them":
        raise ValueError(
            "they/them is reserved for referring to people whose gender the mind "
            "does not know — it cannot be a mind's own pronoun set. "
            "Use FEMALE, MALE or leave the mind genderless (UNSPECIFIED).")
    if p not in _ALLOWED_SELF_SETS and p.label not in ("she/her", "he/him", "unspecified"):
        raise ValueError(
            "Only the she/her and he/him presets can be set as a mind's own pronouns.")


def starts_with_article(word: str) -> bool:
    return word.lower().startswith(("a ", "an ", "the "))


def article_for(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


class GenderOptions:
    """Identity word + pronoun set for a mind. Serializes to/from a plain
    dict for snapshots."""

    __slots__ = ("identity", "pronouns")

    def __init__(self, identity: str = "", pronouns: PronounSet = UNSPECIFIED) -> None:
        self.identity = (identity or "").strip()
        if is_reserved_label(self.identity):
            raise ValueError(
                f"'{self.identity}' is not a pronoun set a mind can adopt for itself. "
                "they/them is reserved for describing other people whose gender the "
                "mind does not know. Use she/her or he/him, or leave the mind genderless.")
        self.pronouns = pronouns or UNSPECIFIED
        validate_self_pronouns(self.pronouns)

    def _validate_pronouns(self) -> None:
        validate_self_pronouns(self.pronouns)

    @property
    def is_unset(self) -> bool:
        return not self.identity and self.pronouns == UNSPECIFIED

    def set(self, identity: str, pronouns: PronounSet | None = None) -> "GenderOptions":
        """Sets identity and pronouns in one call. When the identity word
        itself names a pronoun set ("she/her", "female"...) it is treated as
        a pronoun choice only — the label is not stored as an identity word."""
        if pronouns is None:
            parsed = parse_pronouns(identity)
            if parsed is not None:
                self.pronouns = parsed
                return self
        word = (identity or "").strip()
        if is_reserved_label(word):
            # validate BEFORE touching state — a rejected attempt must not
            # leave a half-applied identity behind
            raise ValueError(
                f"'{word}' is not a pronoun set a mind can adopt for itself. "
                "they/them is reserved for describing other people whose gender the "
                "mind does not know. Use she/her or he/him, or leave the mind genderless.")
        self.identity = word
        if pronouns is not None:
            self.pronouns = pronouns
            validate_self_pronouns(self.pronouns)
        return self

    def gender_clause(self) -> str:
        """The clause the mind uses when describing itself — "I am a girl,
        and I go by she/her pronouns." — or "" when unset."""
        identity_part = ""
        if self.identity:
            word = self.identity
            identity_part = (f"I am {word}" if starts_with_article(word)
                             else f"I am {article_for(word)} {word}")
        pronoun_part = ("" if self.pronouns == UNSPECIFIED
                        else f"I go by {self.pronouns.subject}/{self.pronouns.object} pronouns")
        if identity_part and pronoun_part:
            return f"{identity_part}, and {pronoun_part}."
        if identity_part:
            return identity_part + "."
        if pronoun_part:
            return pronoun_part + "."
        return ""

    def to_dict(self) -> dict:
        return {
            "identity": self.identity,
            "pronouns": {
                "label": self.pronouns.label,
                "subject": self.pronouns.subject,
                "object": self.pronouns.object,
                "possessive": self.pronouns.possessive,
                "possessive_pronoun": self.pronouns.possessive_pronoun,
                "reflexive": self.pronouns.reflexive,
            },
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "GenderOptions":
        if not data:
            return cls()
        p = data.get("pronouns") or {}
        label = str(p.get("label", "") or "")
        identity = str(data.get("identity", "") or "").strip()
        # Snapshots from older versions may carry pronoun sets the gender
        # system no longer offers — sanitize to what it can keep.
        if is_reserved_label(identity):
            identity = ""
        if not label:
            return cls(identity=identity)
        pronouns = PronounSet(label,
                              str(p.get("subject") or ""),
                              str(p.get("object") or ""),
                              str(p.get("possessive") or ""),
                              str(p.get("possessive_pronoun") or ""),
                              str(p.get("reflexive") or ""))
        if pronouns.label not in ("she/her", "he/him", "unspecified"):
            pronouns = UNSPECIFIED
        options = cls.__new__(cls)
        options.identity = identity
        options.pronouns = pronouns
        return options
