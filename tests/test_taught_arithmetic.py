"""Taught-arithmetic reasoning (mirrors the C# ArithmeticReasoningTests).

Yume answers math questions from what she was TAUGHT (direct fact, or
derived from related taught facts), uses the calculator only as a
transparent fallback, and never learns garbage from unlabeled questions.
"""
from cognify import MindBuilder, Stimulus
from cognify.reasoning.arithmetic import try_teach
from cognify.reasoning.facts import FactParser


def make_mind(**kw):
    return (MindBuilder("Yume")
            .with_seed(7)
            .with_reasoning()
            .with_thinking()
            .with_memory()
            .with_tools()
            .build())


def make_mind_without_tools():
    return (MindBuilder("Yume")
            .with_seed(7)
            .with_reasoning()
            .with_thinking()
            .with_memory()
            .build())


def teach(mind, *lines):
    for line in lines:
        learned = mind.teach(line)
        assert learned, f"teaching failed: {line}"


class TestTaughtFacts:
    def test_taught_fact_is_stored_structured(self):
        facts = FactParser().parse("2 + 2 is 4")
        assert len(facts) == 1
        assert (facts[0].subject, facts[0].predicate, facts[0].object) == ("2+2", "equals", "4")

        worded = FactParser().parse("two plus two is four")
        assert len(worded) == 1
        assert (worded[0].subject, worded[0].predicate, worded[0].object) == ("2+2", "equals", "4")

    def test_wrong_teaching_is_rejected(self):
        assert try_teach("2 + 2 is 5") is None
        assert try_teach("2 + 2 is 4") is not None

    def test_taught_fact_is_answered_without_calculator(self):
        mind = make_mind()
        teach(mind, "2 + 2 = 4")

        resp = mind.process(Stimulus.text("Hello Yume, what is 2 + 2?"))

        assert not any("calculator" in t for t in resp.tool_results)
        assert "4" in resp.text
        assert "taught" in resp.text.lower()


class TestDerivation:
    def test_untaught_sum_is_derived_by_counting_on(self):
        mind = make_mind()
        teach(mind, "2 + 1 = 3", "3 + 1 = 4")

        resp = mind.process(Stimulus.text("what is 2 + 2?"))

        assert not any("calculator" in t for t in resp.tool_results)
        assert "4" in resp.text
        assert "counting on" in resp.text.lower()

    def test_commutative_taught_fact_is_recognized(self):
        mind = make_mind()
        teach(mind, "2 + 3 = 5")

        resp = mind.process(Stimulus.text("what is 3 + 2?"))

        assert not any("calculator" in t for t in resp.tool_results)
        assert "5" in resp.text
        assert "same as" in resp.text.lower()

    def test_subtraction_is_derived_from_taught_addition(self):
        mind = make_mind()
        teach(mind, "2 + 1 = 3")

        resp = mind.process(Stimulus.text("what is 3 - 1?"))

        assert not any("calculator" in t for t in resp.tool_results)
        assert "2" in resp.text

    def test_number_words_are_understood(self):
        mind = make_mind()
        teach(mind, "2 + 2 = 4")

        resp = mind.process(Stimulus.text("what is two plus two?"))

        assert "4" in resp.text


class TestWordProblems:
    def test_word_problem_is_reasoned_from_taught_arithmetic(self):
        mind = make_mind()
        teach(mind, "1 + 1 = 2")

        resp = mind.process(Stimulus.text(
            "if I have 1 apple and buy another one, how many apples do I have?"))

        assert not any("calculator" in t for t in resp.tool_results)
        assert "2 apples" in resp.text

    def test_word_problem_with_taught_subtraction(self):
        mind = make_mind()
        teach(mind, "3 + 2 = 5")

        resp = mind.process(Stimulus.text(
            "I have 5 cookies and I eat 2, how many cookies do I have?"))

        assert "3 cookies" in resp.text

    def test_unanswerable_word_problem_falls_back_transparently(self):
        mind = make_mind()  # nothing taught at all
        resp = mind.process(Stimulus.text(
            "if I have 4 apples and buy 3 more, how many apples do I have?"))
        # Taught reasoning could not work this out, so the calculator fills
        # in — but it must SAY so openly instead of passing mechanical math
        # off as knowledge the mind reasoned out itself.
        assert "not been taught" in resp.text
        assert "7" in resp.text

    def test_unanswerable_word_problem_without_tools_admits_not_knowing(self):
        mind = (MindBuilder("Yume").with_seed(7)
                .with_reasoning()
                .enable_tool_use(False)
                .build())
        resp = mind.process(Stimulus.text(
            "if I have 4 apples and buy 3 more, how many apples do I have?"))
        assert "7" not in resp.text


class TestCalculatorFallback:
    def test_untaught_question_falls_back_transparently(self):
        mind = make_mind()
        resp = mind.process(Stimulus.text("what is 2 + 3?"))

        assert any("calculator" in t for t in resp.tool_results)
        assert "not been taught" in resp.text.lower()
        assert "2 + 3 = 5" in resp.text

    def test_untaught_question_without_tools_admits_not_knowing(self):
        mind = make_mind_without_tools()
        resp = mind.process(Stimulus.text("what is 2 + 3?"))

        assert resp.tool_results == []
        assert "5" not in resp.text
        assert "learned" not in resp.text.lower()

    def test_user_can_explicitly_demand_the_calculator(self):
        mind = make_mind()
        teach(mind, "2 + 2 = 4")

        resp = mind.process(Stimulus.text("use the calculator to check what is 2 + 2"))

        assert any("calculator" in t for t in resp.tool_results)
        assert "2 + 2 = 4" in resp.text


class TestNoGarbageFacts:
    def test_unlabeled_questions_never_become_facts(self):
        parser = FactParser()
        assert parser.parse("yume what is 2 + 2") == []
        assert parser.parse("tell me what 2 + 2 is") == []
        assert parser.parse("hello yume what is 2 plus 2") == []


class TestNoVerbatimRepeats:
    def test_consecutive_teaching_confirmations_never_repeat(self):
        mind = make_mind_without_tools()
        first = mind.process(Stimulus.text("1 + 1 = 2"))
        second = mind.process(Stimulus.text("2 + 1 = 3"))
        third = mind.process(Stimulus.text("3 + 1 = 4"))

        assert first.text != second.text
        assert second.text != third.text
        assert first.text != third.text
