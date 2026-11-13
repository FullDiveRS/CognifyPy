"""Parity tests: identity questions and arithmetic routing (mirrors the C# suite)."""
from cognify import MindBuilder, Stimulus


def make_mind(**kw):
    return (MindBuilder("Yume")
            .with_seed(7)
            .with_reasoning()
            .with_tools()
            .build())


class TestIdentityAnswer:
    def test_introduce_yourself_answers_with_name(self):
        mind = make_mind()
        resp = mind.process(Stimulus.text("Hello, Introduce yourself."))
        assert "Yume" in resp.text
        assert "CognifyPy" in resp.text

    def test_who_are_you_answers_with_name(self):
        mind = make_mind()
        resp = mind.process(Stimulus.text("who are you?"))
        assert "Yume" in resp.text

    def test_identity_answer_high_confidence(self):
        mind = make_mind()
        resp = mind.process(Stimulus.text("tell me about yourself"))
        assert resp.confidence >= 0.6


class TestArithmeticRouting:
    def test_arithmetic_question_uses_calculator(self):
        mind = make_mind()
        resp = mind.process(Stimulus.text("Hello Yume, what is 2 + 2?"))
        assert any("calculator" in t for t in resp.tool_results)
        assert any("2 + 2 = 4" in t for t in resp.tool_results)

    def test_answer_to_arithmetic_found_mid_sentence(self):
        mind = make_mind()
        resp = mind.process(Stimulus.text("what is the answer to 12 * 3?"))
        assert "36" in resp.text

    def test_arithmetic_question_does_not_claim_learning(self):
        mind = make_mind()
        resp = mind.process(Stimulus.text("Hello Yume, what is 2 + 2?"))
        assert "learned" not in resp.text.lower()
        assert mind.kb.size() == 0
