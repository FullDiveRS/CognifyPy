"""Comprehension regressions from the apple-and-number-system transcript.

Mirrors the C# ArithmeticComprehensionTests: natural quantities ("no
apples", "an apple"), the "how may" typo, the taught counting ladder
(reasoning along "0, 1, 2, ..., 9" instead of the calculator), honest
learned counts (conditionals / existentials / "can be" are not facts),
template-level repeat avoidance and topic salience.
"""
from cognify import MindBuilder, Stimulus
from cognify.reasoning.facts import FactParser

NUMBER_SYSTEM_LESSON = (
    "there is a 10 based number system, starts at 0, 0 can be nothing or can be a holder for something. "
    "the system is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. "
    "if i have no apples and i buy an apple. how many do i currently have?"
)


def make_mind(**kw):
    return (MindBuilder("Yume")
            .with_seed(7)
            .with_reasoning()
            .with_thinking()
            .with_memory()
            .with_tools()
            .build())


class TestCountingLadder:
    def test_taught_number_line_lets_her_count_on_without_calculator(self):
        mind = make_mind()
        mind.process(Stimulus.text("the system is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9"))

        resp = mind.process(Stimulus.text("what is 3 + 4?"))

        assert not any("calculator" in t for t in resp.tool_results)
        assert "7" in resp.text
        assert "counting on" in resp.text.lower()

    def test_beyond_the_taught_line_she_uses_the_calculator_openly(self):
        mind = make_mind()
        mind.process(Stimulus.text("the system is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9"))

        resp = mind.process(Stimulus.text("what is 8 + 5?"))  # 13 lies beyond the line

        assert any("calculator" in t for t in resp.tool_results)
        assert "13" in resp.text

    def test_taught_line_also_counts_backward(self):
        mind = make_mind()
        mind.process(Stimulus.text("the system is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9"))

        resp = mind.process(Stimulus.text(
            "if i have 3 apples and i eat an apple, how many apples do i have?"))

        assert not any("calculator" in t for t in resp.tool_results)
        assert "2 apples" in resp.text


class TestNaturalQuantities:
    def test_no_apples_and_an_apple_are_understood(self):
        mind = make_mind()

        resp = mind.process(Stimulus.text(NUMBER_SYSTEM_LESSON))

        assert "1 apple" in resp.text          # answered, not just "learned"
        assert not any("calculator" in t for t in resp.tool_results)

    def test_quoted_word_problem_is_answered_not_echoed(self):
        mind = make_mind()
        mind.process(Stimulus.text(NUMBER_SYSTEM_LESSON))

        resp = mind.process(Stimulus.text(
            'what is the answer to "if i have no apples and i buy an apple. '
            'how many do i currently have?"'))

        assert "1 apple" in resp.text
        assert "remember our earlier exchange" not in resp.text.lower()

    def test_how_may_typo_is_still_a_question(self):
        mind = make_mind()
        mind.process(Stimulus.text(NUMBER_SYSTEM_LESSON))

        resp = mind.process(Stimulus.text(
            "if i have 0 apples, and i buy an apple. how may apples do i have now?"))

        assert "1 apple" in resp.text
        assert "learned" not in resp.text.lower()

    def test_stated_quantity_alone_is_answerable(self):
        mind = make_mind()

        resp = mind.process(Stimulus.text("i have 3 apples, how many apples do i have?"))

        assert "3 apples" in resp.text


class TestHonestLearnedCounts:
    def test_conditionals_existentials_and_can_be_are_not_facts(self):
        parser = FactParser()
        assert parser.parse("if i have no apples and i buy an apple") == []
        assert parser.parse("there is a 10 based number system, starts at 0") == []
        assert parser.parse("0 can be nothing or can be a holder for something") == []

        # Real knowledge still parses.
        assert len(parser.parse("birds can fly")) == 1
        assert len(parser.parse("the system is base 10")) == 1

    def test_lesson_learns_one_honest_fact_not_garbage(self):
        mind = make_mind()
        before = mind.kb.size()

        mind.process(Stimulus.text(NUMBER_SYSTEM_LESSON))

        assert mind.kb.size() == before + 1    # only "the system is 0-1-...-9"


class TestTemplateRepeats:
    def test_teaching_turns_never_reuse_the_same_sentence_template(self):
        mind = make_mind()
        replies = []
        for line in ("2 + 2 is 4", "3 + 3 is 6", "10 + 10 is 20", "5 + 5 is 10"):
            replies.append(mind.process(Stimulus.text(line)).text)

        for i in range(len(replies)):
            for j in range(i + 1, len(replies)):
                assert replies[i] != replies[j]


class TestTopicSalience:
    def test_time_fillers_never_become_the_topic(self):
        mind = make_mind()
        mind.process(Stimulus.text("my favorite fruit is apples"))

        resp = mind.process(Stimulus.text("what do you recall about my apples currently?"))

        assert "currently" not in resp.text.lower()
        assert "apples" in resp.text.lower()
