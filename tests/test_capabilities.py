"""Tests for embeddings, multilingual, tool-use, and code capabilities."""
import pytest

from cognify.capabilities.code import (CodeCapability, analyze_python,
                                       detect_language)
from cognify.capabilities.embeddings import EmbeddingService
from cognify.capabilities.multilingual import MultilingualCapability
from cognify.capabilities.tool_use import (SafeMathError, ToolRegistry,
                                           ToolUseCapability, safe_math, tool)
from cognify.core.stimulus import Stimulus


class TestEmbeddings:
    def test_service_similarity(self):
        svc = EmbeddingService(dims=64)
        assert svc.similarity("a dog runs", "a dog runs fast") > \
               svc.similarity("a dog runs", "tax law is complex")

    def test_nearest_ranking(self):
        svc = EmbeddingService(dims=64)
        items = [("cat", "the cat meows"), ("car", "the car drives fast"),
                 ("math", "algebra and calculus")]
        ranked = svc.nearest("a fast car on the road", items, k=2)
        assert ranked[0][0] == "car"

    def test_dims_property(self):
        assert EmbeddingService(dims=48).dims == 48

    def test_capability_embeds_into_ctx(self):
        from cognify.core.context import CognitiveContext
        from cognify.emotions.emotion_state import EmotionalState
        from cognify.emotions.personality import Personality

        svc = EmbeddingService(dims=64)
        ctx = CognitiveContext(cycle=1, stimulus=Stimulus.text("hello world"),
                               personality=Personality(), emotions=EmotionalState())
        percept = svc.handle(Stimulus.text("hello world"), ctx)
        assert percept is not None
        assert ctx.embedding is not None
        assert len(ctx.embedding) == 64


class TestMultilingual:
    @pytest.fixture
    def ml(self):
        return MultilingualCapability()

    def test_english(self, ml):
        lang, conf = ml.detect("What is the weather today my friend?")
        assert lang == "en"

    def test_spanish(self, ml):
        lang, conf = ml.detect("Hola, gracias por todo")
        assert lang == "es" and conf > 0.5

    def test_french(self, ml):
        lang, _ = ml.detect("Bonjour merci beaucoup le matin")
        assert lang == "fr"

    def test_german(self, ml):
        lang, _ = ml.detect("Hallo, danke sehr, das ist gut")
        assert lang == "de"

    def test_chinese_by_script(self, ml):
        lang, conf = ml.detect("你好，谢谢")
        assert lang == "zh" and conf > 0.8

    def test_japanese_by_kana(self, ml):
        lang, conf = ml.detect("こんにちは、ありがとうございます")
        assert lang == "ja" and conf > 0.8

    def test_korean_by_hangul(self, ml):
        lang, _ = ml.detect("안녕하세요 감사합니다")
        assert lang == "ko"

    def test_russian_by_cyrillic(self, ml):
        lang, _ = ml.detect("привет, спасибо большое")
        assert lang == "ru"

    def test_confidence_bounds(self, ml):
        for text in ("hello there", "hola amigo", "bonjour mon ami", ""):
            _, conf = ml.detect(text)
            assert 0.0 <= conf <= 1.0

    def test_phrasebook(self, ml):
        assert ml.greet("es") == "¡Hola"
        assert ml.phrase("thanks", "fr") == "Merci"
        assert ml.translate("cat", "de") == "katze"

    def test_unknown_is_safe(self, ml):
        lang, conf = ml.detect("12345 !!!")
        assert lang in ("en", "unknown")

    def test_capability_sets_ctx(self):
        from cognify.core.context import CognitiveContext
        from cognify.emotions.emotion_state import EmotionalState
        from cognify.emotions.personality import Personality

        ml = MultilingualCapability()
        ctx = CognitiveContext(cycle=1, stimulus=Stimulus.text("Hola, gracias"),
                               personality=Personality(), emotions=EmotionalState())
        percept = ml.handle(Stimulus.text("Hola, gracias"), ctx)
        assert percept.language == "es"
        assert ctx.scratch["language"] == "es"
        assert "Hola" in ctx.scratch["localized_greeting"]


class TestSafeMath:
    def test_basic(self):
        assert safe_math("2 + 3 * 4") == 14
        assert safe_math("(2 + 3) * 4") == 20
        assert safe_math("2 ** 10") == 1024
        assert safe_math("-5 + 1") == -4

    def test_division_by_zero(self):
        with pytest.raises(SafeMathError):
            safe_math("1 / 0")

    def test_rejects_names(self):
        with pytest.raises(SafeMathError):
            safe_math("__import__('os').system('ls')")

    def test_rejects_calls(self):
        with pytest.raises(SafeMathError):
            safe_math("pow(2, 3)")

    def test_rejects_strings(self):
        with pytest.raises(SafeMathError):
            safe_math("'a' + 'b'")


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()

        @tool
        def add_things(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        reg.register(add_things)
        spec = reg.get("add things")
        assert spec is not None
        assert spec.params == {"a": "int", "b": "int"}
        assert "Add two numbers" in spec.description

    def test_register_rejects_plain_function(self):
        reg = ToolRegistry()
        with pytest.raises(TypeError):
            reg.register(lambda x: x)

    def test_match_by_description(self):
        reg = ToolRegistry()

        @tool(name="weather lookup", description="check the weather forecast rain")
        def weather(city: str) -> str:
            return "sunny"

        reg.register(weather)
        hits = reg.match("what is the weather like today")
        assert hits and hits[0].name == "weather lookup"


class TestToolPlanning:
    @pytest.fixture
    def cap(self):
        reg = ToolRegistry()
        from cognify.capabilities.tool_use import builtin_tools
        for fn in builtin_tools():
            reg.register(fn)
        return ToolUseCapability(reg)

    def test_math_plan(self, cap):
        plans = cap.plan(Stimulus.text("What is 12 * 12?"))
        assert plans and plans[0][0] == "calculator"

    def test_time_plan(self, cap):
        plans = cap.plan(Stimulus.text("What time is it right now?"))
        assert plans and plans[0][0] == "current time"

    def test_conversion_plan(self, cap):
        plans = cap.plan(Stimulus.text("convert 5 km to miles please"))
        assert plans and plans[0][0] == "unit convert"

    def test_explicit_call_syntax(self, cap):
        plans = cap.plan(Stimulus.text("@calculator(expression=3+3)"))
        assert plans and plans[0][0] == "calculator"

    def test_no_plan_for_chitchat(self, cap):
        assert cap.plan(Stimulus.text("hello my friend")) == []

    def test_execution_records_invocation(self, cap):
        from cognify.core.context import CognitiveContext
        from cognify.emotions.emotion_state import EmotionalState
        from cognify.emotions.personality import Personality

        ctx = CognitiveContext(cycle=1, stimulus=Stimulus.text("What is 7 * 6?"),
                               personality=Personality(), emotions=EmotionalState())
        cap.handle(Stimulus.text("What is 7 * 6?"), ctx)
        assert len(ctx.tool_invocations) == 1
        inv = ctx.tool_invocations[0]
        assert inv.success and "42" in str(inv.result)

    def test_failure_captured(self):
        reg = ToolRegistry()

        @tool(name="fragile", description="always fails on purpose")
        def fragile(x: int) -> int:
            raise RuntimeError("nope")

        reg.register(fragile)
        cap = ToolUseCapability(reg)
        from cognify.core.context import CognitiveContext
        from cognify.emotions.emotion_state import EmotionalState
        from cognify.emotions.personality import Personality

        ctx = CognitiveContext(cycle=1, stimulus=Stimulus.text("use fragile please"),
                               personality=Personality(), emotions=EmotionalState())
        cap.handle(Stimulus.text("@fragile(x=1)"), ctx)
        inv = ctx.tool_invocations[0]
        assert not inv.success and "nope" in inv.error


class TestCodeCapability:
    def test_detect_language(self):
        assert detect_language("def foo():\n    return 1\nprint(foo)") == "python"
        assert detect_language("public class Foo { static void Main() {} }") in ("c#", "java")

    def test_analyze_python(self):
        code = ('def double(x):\n    return x * 2\n'
                'class Widget:\n    """A widget."""\n    pass\n'
                'import os\n')
        a = analyze_python(code)
        assert a.language == "python"
        assert [f["name"] for f in a.functions] == ["double"]
        assert a.classes == ["Widget"]
        assert "os" in a.imports
        assert any("docstring" in i for i in a.issues)   # double lacks one

    def test_lint_long_line_and_bare_except(self):
        code = ("def f():\n"
                "    try:\n"
                "        pass\n"
                "    except:\n"
                "        pass\n"
                + "#" + "x" * 120 + "\n")
        a = analyze_python(code)
        assert any("bare" in i for i in a.issues)
        assert any("100 characters" in i for i in a.issues)
        assert a.quality < 100

    def test_syntax_error_reported(self):
        a = analyze_python("def broken(:\n    pass")
        assert any("syntax error" in i for i in a.issues)

    def test_generate_known_template(self):
        cap = CodeCapability()
        code, name, expl = cap.generate("a function that reverses a string")
        assert name == "reverse_string"
        assert "[::-1]" in code
        compile(code, "<gen>", "exec")               # generated code is valid

    def test_generate_fibonacci(self):
        cap = CodeCapability()
        code, name, _ = cap.generate("fibonacci sequence function")
        assert name == "fibonacci"
        ns = {}
        exec(compile(code, "<gen>", "exec"), ns)     # noqa: S102 - trusted template
        assert ns[name](7) == 13

    def test_generate_stub_fallback(self):
        cap = CodeCapability()
        code, name, _ = cap.generate("quantum entangle two qubits")
        assert "NotImplementedError" in code

    def test_can_handle_generation_request(self):
        cap = CodeCapability()
        assert cap.can_handle(Stimulus.text("Write a function that adds two numbers"))
        assert not cap.can_handle(Stimulus.text("hello there friend"))

    def test_handle_generation(self):
        cap = CodeCapability()
        p = cap.handle(Stimulus.text("Write a function that adds two numbers"))
        assert p.kind == "generation" and p.language == "python"
        assert "def " in p.details["code"]

    def test_handle_analysis(self):
        cap = CodeCapability()
        snippet = "def foo():\n    return 42\n\nclass A:\n    pass\n"
        p = cap.handle(Stimulus.code(snippet))
        assert p.kind == "analysis"
        assert "foo" in p.details["functions"]
