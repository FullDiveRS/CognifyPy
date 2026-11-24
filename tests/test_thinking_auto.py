"""Tests for the standalone AUTOMATIC thinking resolver
(cognify.thinking.thinking_auto — mirror of the C# ``ThinkingAuto``)."""
import pytest

from cognify.core.options import ThinkingCeiling, ThinkingLevel
from cognify.thinking.thinking_auto import choose


def test_bare_hello_thinks_shallow():
    level = choose("hi", has_direct_answer=False, recalled_count=0,
                   emotion_intensity=0.1, epistemic_gap=False,
                   ceiling=ThinkingCeiling.ONLY5)
    assert level == ThinkingLevel.LOW


def test_complex_why_how_question_thinks_deep():
    level = choose(
        "Why do flowers turn toward the sun, and how does the whole process "
        "actually work inside the plant over a full day, from dawn to dusk?",
        has_direct_answer=False, recalled_count=0, emotion_intensity=0.2,
        epistemic_gap=True, ceiling=ThinkingCeiling.ONLY5)
    assert level in (ThinkingLevel.VERY_HIGH, ThinkingLevel.MAX), \
        f"a heavy why/how probe should think very hard, got {level}"


def test_statement_with_known_answer_thinks_light_questions_think_deep():
    # A bare statement whose answer is already known and whose memories are
    # rich stays light — thinking exists for the hard parts.
    statement = choose("tulips are my favorite flowers", has_direct_answer=True,
                       recalled_count=3, emotion_intensity=0.2,
                       epistemic_gap=False, ceiling=ThinkingCeiling.ONLY5)
    assert statement == ThinkingLevel.LOW

    # The SAME known ground, asked as a question, still deserves real
    # deliberation — questions are what thinking is for (+2).
    question = choose("what is a tulip?", has_direct_answer=True,
                      recalled_count=3, emotion_intensity=0.2,
                      epistemic_gap=False, ceiling=ThinkingCeiling.ONLY5)
    assert question in (ThinkingLevel.HIGH, ThinkingLevel.VERY_HIGH)
    assert list(ThinkingLevel).index(question) > list(ThinkingLevel).index(statement)


@pytest.mark.parametrize("ceiling", [ThinkingCeiling.ONLY1, ThinkingCeiling.ONLY2,
                                     ThinkingCeiling.ONLY3, ThinkingCeiling.ONLY4])
def test_never_exceeds_the_ceiling(ceiling):
    level = choose(
        "Why does the whole mechanism of memory formation actually work "
        "the way it does over time?",
        has_direct_answer=False, recalled_count=0, emotion_intensity=0.9,
        epistemic_gap=True, ceiling=ceiling)
    assert level == ceiling.clamp(ThinkingLevel.MAX)


def test_is_deterministic():
    a = choose("why is the sky blue and how does scattering work?",
               False, 0, 0.3, True, ThinkingCeiling.ONLY5)
    b = choose("why is the sky blue and how does scattering work?",
               False, 0, 0.3, True, ThinkingCeiling.ONLY5)
    assert a == b


def test_long_input_carries_more_weight():
    short = choose("tulips are nice", False, 2, 0.1, False,
                   ThinkingCeiling.ONLY5)
    long = choose("tulips are nice and I have been thinking about how "
                  "the whole garden changes through every season",
                  False, 2, 0.1, False, ThinkingCeiling.ONLY5)
    assert list(ThinkingLevel).index(long) > list(ThinkingLevel).index(short)
