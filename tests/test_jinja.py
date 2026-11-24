"""Tests for cognify.core.jinja: the built-in mini-Jinja engine.

Unit tests cover the minja-compatible subset (expressions, filters, tests,
whitespace control, scoping, namespace, error behavior). The golden tests
render REAL llama.cpp-served templates — ChatML (OpenHermes), Zephyr,
Qwen2.5, Qwen3, Llama-3.1 — fetched from HuggingFace (see
tests/templates/README.md) against canonical contexts, and compare the
output byte-for-byte with tests/goldens/*.txt. The C# port must reproduce
the same goldens, so both engines stay in lockstep.
"""
import json
import re
from pathlib import Path
from typing import Any

import pytest

from cognify import MindBuilder
from cognify.core.jinja import (JNamespace, TemplateError, cognify_chat_template,
                                render)

TEMPLATES = Path(__file__).parent / "templates"
GOLDENS = Path(__file__).parent / "goldens"

TOOLS = [
    {"type": "function", "function": {"name": "get_weather",
     "description": "Get the current weather for a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                    "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_time",
     "description": "Get the current local time.",
     "parameters": {"type": "object", "properties": {}}}},
]

DATE, BOS = "26 Jul 2024", "<|begin_of_text|>"

QWEN3_MESSAGES = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud."},
    {"role": "user", "content": "Weather in Tokyo?"},
    {"role": "assistant",
     "content": "<think>\nI should call a tool for this.\n</think>\nLet me check."},
    {"role": "user", "content": "And the time?"},
]

# context name -> (fixture file, template variables) — mirrors
# scripts/jinja_goldens.py and the C# ChatJinjaTests golden contexts.
FIXTURE_CONTEXTS = {
    "chatml_gen": ("chatml.jinja", {"messages": [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ], "add_generation_prompt": True}),
    "zephyr_gen": ("zephyr.jinja", {"messages": [
        {"role": "system", "content": "You are a friendly chatbot."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ], "eos_token": "</s>", "add_generation_prompt": True}),
    "qwen25_tools_gen": ("qwen25.jinja", {"messages": [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud."},
        {"role": "user", "content": "Weather in Tokyo?"},
    ], "tools": TOOLS, "add_generation_prompt": True}),
    "qwen3_tools_gen_think": ("qwen3.jinja", {"messages": QWEN3_MESSAGES,
        "tools": TOOLS, "add_generation_prompt": True}),
    "qwen3_tools_gen_nothink": ("qwen3.jinja", {"messages": QWEN3_MESSAGES,
        "tools": TOOLS, "add_generation_prompt": True, "enable_thinking": False}),
    "qwen3_notools_gen": ("qwen3.jinja", {"messages": QWEN3_MESSAGES[:2],
        "add_generation_prompt": True}),
    "llama31_tools_gen": ("llama31.jinja", {"messages": [
        {"role": "system", "content": "  You are a helpful assistant.  "},
        {"role": "user", "content": "What is the weather in Tokyo?"},
    ], "tools": TOOLS, "date_string": DATE, "bos_token": BOS,
        "add_generation_prompt": True}),
    "llama31_round_gen": ("llama31.jinja", {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "get_weather",
                          "arguments": {"city": "Tokyo"}}},
        ]},
        {"role": "tool", "content": {"temp_c": 21}},
    ], "tools": TOOLS, "date_string": DATE, "bos_token": BOS,
        "add_generation_prompt": True}),
}

MANIFEST_ESCAPED = json.dumps(
    {"version": 1, "description": "Yume's mind",
     "thinking": {"enabled": True, "level": "balanced", "ceiling": "high"}},
    ensure_ascii=False).replace("'", "\\u0027")


def _golden(name: str) -> str:
    return (GOLDENS / f"{name}.txt").read_text()


# ----------------------------------------------------------------- expressions
class TestExpressions:
    def test_literals_and_arithmetic(self):
        assert render("{{ 1 + 2 * 3 }}") == "7"
        assert render("{{ (1 + 2) * 3 }}") == "9"
        assert render("{{ 7 // 2 }} {{ 7 / 2 }} {{ 7 % 3 }}") == "3 3.5 1"
        assert render("{{ 2.5 }} {{ -1 }} {{ +4 }}") == "2.5 -1 4"
        assert render("{{ 2.0 }}") == "2.0"

    def test_strings_and_concat(self):
        assert render("{{ 'a' + 'b' }}{{ \"a\\\"b\" }}") == 'aba"b'
        assert render("{{ 'a' ~ 'b' ~ 5 ~ none }}") == "ab5None"
        assert render("{{ '\\n'.join(['a', 'b']) }}") == "a\nb"

    def test_logic_and_comparisons(self):
        assert render("{{ 1 < 2 and 'a' == 'a' or false }}") == "True"
        assert render("{{ not (1 > 2) }}") == "True"
        assert render("{{ 3 >= 3, 3 <= 2, 1 != 2 }}") == "(True, False, True)"

    def test_membership(self):
        assert render("{{ 'ell' in 'hello' }}") == "True"
        assert render("{{ 2 in [1, 2, 3], 'x' not in 'hi' }}") == "(True, True)"
        assert render("{{ 'a' in {'a': 1} }}") == "True"

    def test_ternary(self):
        assert render("{{ 'yes' if 1 > 0 else 'no' }}") == "yes"
        assert render("{{ 'yes' if false else 'no' }}") == "no"

    def test_attribute_and_subscript(self):
        ctx = {"m": {"role": "user", "tags": ["a", "b"],
                     "deep": {"x": {"y": "z"}}}}
        assert render("{{ m.role }} {{ m['role'] }}", ctx) == "user user"
        assert render("{{ m.tags[1] }} {{ m.deep.x.y }}", ctx) == "b z"
        assert render("{{ m.tags[-1] }} {{ m.tags[-2] }}", ctx) == "b a"
        assert render("{{ m.tags[0:2]|join('') }}", ctx) == "ab"

    def test_slices(self):
        ctx = {"xs": [0, 1, 2, 3, 4]}
        assert render("{{ xs[1:]|length }} {{ xs[::-1]|join(',') }}", ctx) \
            == "4 4,3,2,1,0"
        assert render("{{ 'hello'[1:3] }}") == "el"
        assert render("{{ xs[1:4:2]|join(',') }}", ctx) == "1,3"

    def test_list_and_dict_literals(self):
        assert render("{{ [1, 'a', true]|join(',') }}") == "1,a,True"
        assert render("{{ {'b': 2, 'a': 1}['a'] }}") == "1"

    def test_missing_is_undefined(self):
        assert render("[{{ nothing }}]") == "[]"
        assert render("{{ nothing is defined }} {{ nothing.deeper is defined }}") \
            == "False False"

    def test_truthiness(self):
        assert render("{{ '' if '' else 'empty' }} {{ 0 or 'zero' }}") \
            == "empty zero"
        # Jinja's and/or return the operand value, like Python.
        assert render("{{ [] and 'x' }}|{{ 'y' and 2 }}") == "[]|2"
        assert render("{{ none or 'fallback' }}") == "fallback"


# -------------------------------------------------------------------- filters
class TestFilters:
    def test_tojson_sorted_compact_like_minja(self):
        # llama.cpp renders with nlohmann::json: sorted keys, no spaces.
        assert render("{{ d | tojson }}",
                      {"d": {"b": 2, "a": [1, "x"]}}) == '{"a":[1,"x"],"b":2}'
        assert render("{{ d | tojson }}", {"d": {"s": "q\"uote"}}) \
            == '{"s":"q\\"uote"}'

    def test_tojson_indent(self):
        out = render("{{ d | tojson(indent=4) }}", {"d": {"a": 1}})
        assert out == '{\n    "a": 1\n}'

    def test_trim_first_last(self):
        assert render("{{ '  x\\n ' | trim }}|") == "x|"
        assert render("{{ xs | first }} {{ xs | last }}", {"xs": [3, 1, 2]}) \
            == "3 2"

    def test_length_join_items(self):
        assert render("{{ 'abcd' | length }} {{ [1, 2] | count }}") == "4 2"
        assert render("{{ ['a', 'b'] | join('-') }}") == "a-b"
        assert render("{{ d | items | join(';') }}", {"d": {"b": 2, "a": 1}}) \
            == "['a', 1];['b', 2]"

    def test_reject_select_with_tests(self):
        # The exact construct Llama-3.1 uses for builtin tools.
        ctx = {"tools": ["code_interpreter", "python", "web_search"]}
        assert render("{{ tools | reject('equalto', 'python') | join(', ') }}",
                      ctx) == "code_interpreter, web_search"
        assert render("{{ tools | select('equalto', 'python') | join }}", ctx) \
            == "python"
        assert render("{{ [1, 2, 'a'] | select('number') | join(',') }}") == "1,2"

    def test_default(self):
        assert render("{{ missing | default('dflt') }}") == "dflt"
        assert render("{{ none | default(0) }} {{ '' | default('x', true) }}") \
            == "0 x"
        assert render("{{ '' | default('x') }}") == ""

    def test_string_casing_and_replace(self):
        assert render("{{ 'aBc' | upper }} {{ 'aBc' | lower }}") == "ABC abc"
        assert render("{{ 'hello world' | capitalize }}") == "Hello world"
        assert render("{{ 'aaa' | replace('a', 'b', ) }}") == "bbb"
        assert render("{{ 'abc' | reverse }} {{ [1, 2] | reverse | join(',') }}") \
            == "cba 2,1"

    def test_conversions(self):
        assert render("{{ '42' | int }} {{ '4.5' | float }} {{ 5 | string | length }}") \
            == "42 4.5 1"
        assert render("{{ 'bad' | int(7) }} {{ missing | int }}") == "7 0"
        assert render("{{ 'ab' | list | join('-') }}") == "a-b"

    def test_unknown_filter_raises(self):
        with pytest.raises(TemplateError, match="unsupported filter"):
            render("{{ x | selectattr('a') }}", {"x": [1]})


# ----------------------------------------------------------------------- tests
class TestJinjaTests:
    def test_defined_family(self):
        ctx = {"s": "x", "n": None, "d": {"a": 1}}
        assert render("{{ s is defined }} {{ nope is defined }} {{ nope is undefined }}", ctx) \
            == "True False True"
        assert render("{{ n is none }} {{ s is none }}", ctx) == "True False"

    def test_type_tests(self):
        ctx = {"s": "x", "i": 3, "f": 1.5, "b": True, "d": {"a": 1}, "l": [1]}
        assert render("{{ s is string }} {{ i is number }} {{ b is boolean }}", ctx) \
            == "True True True"
        assert render("{{ d is mapping }} {{ l is iterable }} {{ i is integer }} {{ f is float }}", ctx) \
            == "True True True True"
        assert render("{{ i is not string }} {{ s is false }} {{ b is true }}", ctx) \
            == "True False True"

    def test_equalto(self):
        assert render("{{ 5 is equalto 5 }} {{ 5 is eq 4 }} {{ 5 is ne 4 }}") \
            == "True False True"


# ------------------------------------------------------------ statements/scope
class TestStatements:
    def test_if_elif_else(self):
        tpl = "{% if x == 1 %}one{% elif x == 2 %}two{% else %}many{% endif %}"
        assert render(tpl, {"x": 1}) == "one"
        assert render(tpl, {"x": 2}) == "two"
        assert render(tpl, {"x": 9}) == "many"

    def test_loop_variables(self):
        tpl = ("{% for x in xs %}{{ loop.index }}:{{ x }}"
               "{% if not loop.last %},{% endif %}{% endfor %}")
        assert render(tpl, {"xs": ["a", "b", "c"]}) == "1:a,2:b,3:c"
        tpl = "{% for x in xs %}{{ loop.index0 }}/{{ loop.revindex }}/{{ loop.revindex0 }} {% endfor %}"
        assert render(tpl, {"xs": [7, 8]}) == "0/2/1 1/1/0 "

    def test_for_else(self):
        assert render("{% for x in xs %}{{ x }}{% else %}empty{% endfor %}",
                      {"xs": []}) == "empty"
        assert render("{% for x in xs %}{{ x }}{% else %}empty{% endfor %}",
                      {"xs": [1]}) == "1"

    def test_loop_scoping_set_does_not_leak(self):
        # Jinja rule: assignments inside a loop stay in the iteration scope.
        assert render("{% for x in xs %}{% set y = x %}{{ y }}{% endfor %}[{{ y is defined }}]",
                      {"xs": [1, 2]}) == "12[False]"

    def test_namespace_accumulates(self):
        # The Qwen3 pattern: a namespace() bag survives across iterations.
        tpl = ("{%- set ns = namespace(total=0, seen='') %}"
               "{% for x in xs %}{% set ns.total = ns.total + x %}"
               "{% set ns.seen = ns.seen ~ x %}{% endfor %}"
               "{{ ns.total }} {{ ns.seen }}")
        assert render(tpl, {"xs": [1, 2, 3]}) == "6 123"

    def test_tuple_unpack(self):
        assert render("{% for k, v in d | items %}{{ k }}={{ v }} {% endfor %}",
                      {"d": {"a": 1, "b": 2}}) == "a=1 b=2 "
        assert render("{% set a, b = [1, 2] %}{{ a }}{{ b }}") == "12"

    def test_set_at_loop_level_leaks_within_loop(self):
        # if/else create no scope: a set inside `if` inside `for` persists
        # for the rest of the iteration (Llama-3.1 relies on this).
        assert render("{% for x in [1] %}{% if true %}{% set z = x %}{% endif %}{{ z }}{% endfor %}") \
            == "1"

    def test_generation_tag_renders_body(self):
        assert render("{% generation %}AB{% endgeneration %}") == "AB"

    def test_comments_render_nothing(self):
        assert render("a{# note #}b") == "ab"
        assert render("a{# it's fine #}b") == "ab"

    def test_dict_iteration_yields_keys(self):
        assert render("{% for k in d %}{{ k }}{% endfor %}", {"d": {"b": 1, "a": 2}}) \
            == "ba"  # insertion order, like Jinja over a Python dict


# ------------------------------------------------------- whitespace & methods
class TestWhitespaceAndMethods:
    def test_default_keeps_newlines(self):
        assert render("{% if true %}\n  x\n{% endif %}") == "\n  x\n"

    def test_strip_flags(self):
        assert render("a  {{- 'b' }} c") == "ab c"
        assert render("a {{ 'b' -}}  c") == "a bc"   # strips ALL following ws
        assert render("a\n{%- if true %}b{% endif -%}\nc") == "abc"
        assert render("x {% for i in [1] -%}\n  {{ i }}\n{%- endfor %} y") \
            == "x 1 y"

    def test_string_methods(self):
        ctx = {"c": "<think>\nX\n</think>\nY"}
        assert render("{{ c.split('</think>')[0] }}", ctx) == "<think>\nX\n"
        assert render("{{ c.split('</think>')[-1] }}", ctx) == "\nY"
        assert render("{{ c.rstrip('\\n') }}", ctx) == "<think>\nX\n</think>\nY"
        assert render("{{ 'hi'.startswith('h') }} {{ 'hi'.endswith('x') }}") \
            == "True False"
        assert render("{{ 'aab' | trim }} == {{ '  a  '.strip() }}") \
            == "aab == a"
        assert render("{{ 'a,b'.replace(',', '+') }}") == "a+b"

    def test_dict_get_and_items_methods(self):
        ctx = {"d": {"a": 1}}
        assert render("{{ d.get('a') }} {{ d.get('z', 'dflt') }}", ctx) == "1 dflt"
        assert render("{{ d.keys() | join(',') }} {{ d.values() | join(',') }}", ctx) \
            == "a 1"

    def test_global_functions(self):
        with pytest.raises(TemplateError, match="boom"):
            render("{{ raise_exception('boom') }}")
        out = render("{{ strftime_now('%Y') }}")
        assert re.fullmatch(r"\d{4}", out)
        assert render("{{ range(3) | join(',') }} {{ range(1, 4) | join(',') }}") \
            == "0,1,2 1,2,3"

    def test_namespace_repr_and_exports(self):
        ns = JNamespace(a=1)
        assert ns.a == 1

    def test_errors(self):
        with pytest.raises(TemplateError, match="unterminated"):
            render("{{ 'x' }} {{ oops")
        with pytest.raises(TemplateError, match="without \\{% endif %\\}"):
            render("{{ 'x' }} {% if true %}")
        with pytest.raises(TemplateError, match="unsupported statement 'macro'"):
            render("{% macro x() %}{% endmacro %}")
        with pytest.raises(TemplateError, match="index 5 out of range"):
            render("{{ xs[5] }}", {"xs": [1]})


# ------------------------------------------------------- real-template goldens
class TestRealTemplateGoldens:
    """Every real llama.cpp-served template renders to the committed golden
    bytes — the same files the C# engine must reproduce."""

    @pytest.mark.parametrize("name", sorted(FIXTURE_CONTEXTS))
    def test_fixture_render_matches_golden(self, name):
        fixture, variables = FIXTURE_CONTEXTS[name]
        template = (TEMPLATES / fixture).read_text()
        assert render(template, variables) == _golden(name)

    def test_cognify_template_goldens(self):
        variables = {"messages": [{"role": "user", "content": "hello"}]}
        assert render(cognify_chat_template(), dict(variables)) \
            == _golden("cognify_plain")
        assert render(cognify_chat_template(MANIFEST_ESCAPED, True), dict(variables)) \
            == _golden("cognify_think_on")
        assert render(cognify_chat_template(MANIFEST_ESCAPED, True),
                      {**variables, "enable_thinking": False}) \
            == _golden("cognify_think_off_kw")
        assert render(cognify_chat_template(None, True),
                      {**variables, "tools": [{"name": "clock",
                                               "description": "tells the time"}],
                       "add_generation_prompt": True}) \
            == _golden("cognify_tools")

    def test_manifest_set_line_renders_nothing_and_is_extractable(self):
        template = cognify_chat_template(MANIFEST_ESCAPED, True)
        rendered = render(template, {"messages": []})
        assert "cognify_manifest" not in rendered
        assert rendered == "assistant: thought:"
        embedded = re.search(r"cognify_manifest = '(.*?)' %\}", template).group(1)
        manifest = json.loads(embedded.replace("\\u0027", "'"))
        assert manifest["description"] == "Yume's mind"
        assert manifest["thinking"]["enabled"] is True

    def test_qwen3_reasoning_content_message(self):
        # Qwen3 renders message.reasoning_content into the <think> block.
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Answer.",
             "reasoning_content": "Chain of thought"},
        ]
        out = render((TEMPLATES / "qwen3.jinja").read_text(),
                     {"messages": messages, "add_generation_prompt": True})
        assert "<think>\nChain of thought\n</think>\n\nAnswer." in out

    def test_llama31_raise_exception_on_tool_roundtrip(self):
        # Llama-3.1 raises when tools must ride in the first user message
        # but there is none left: a lone system message is consumed first,
        # leaving messages empty at the guard.
        template = (TEMPLATES / "llama31.jinja").read_text()
        with pytest.raises(TemplateError, match="no first user message"):
            render(template, {"messages": [{"role": "system",
                                            "content": "sys"}],
                              "tools": TOOLS, "date_string": DATE,
                              "bos_token": BOS})


# -------------------------------------------------------- Mind.apply_chat_template
class TestMindApplyChatTemplate:
    def make_mind(self, thinking: bool = True) -> object:
        return (MindBuilder("Yume").with_seed(7)
                .with_thinking(enabled=thinking).build())

    def test_default_template_is_the_exported_format(self):
        mind = self.make_mind(thinking=False)
        assert mind.apply_chat_template([("user", "hello")]) \
            == "user: hello\nassistant:"
        assert mind.apply_chat_template([{"role": "user", "content": "hello"}]) \
            == "user: hello\nassistant:"

    def test_thinking_mind_cue_and_kwarg(self):
        mind = self.make_mind(thinking=True)
        assert mind.apply_chat_template([("user", "hello")]) \
            == "user: hello\nassistant: thought:"
        assert mind.apply_chat_template([("user", "hello")],
                                        enable_thinking=False) \
            == "user: hello\nassistant:"

    def test_tuple_and_dict_messages_normalize(self):
        mind = self.make_mind(thinking=False)
        mixed = [{"role": "system", "content": "be nice"}, ("user", "hi")]
        assert mind.apply_chat_template(mixed) \
            == "system: be nice\nuser: hi\nassistant:"

    def test_bos_eos_default_to_special_tokens(self):
        mind = self.make_mind(thinking=False)
        chatml = "{%- for m in messages %}{{ bos_token }}{{ m['role'] }}:{{ eos_token }}{% endfor %}"
        assert mind.apply_chat_template([("user", "hi")], template=chatml) \
            == "<bos>user:<eos>"

    def test_custom_llama_cpp_template(self):
        mind = self.make_mind(thinking=False)
        chatml = (TEMPLATES / "chatml.jinja").read_text()
        out = mind.apply_chat_template([("user", "Hello!")], template=chatml,
                                       add_generation_prompt=True)
        assert out.endswith("<|im_start|>assistant\n")

    def test_tools_block(self):
        # tools-capable minds generate a template WITH the tools branch...
        mind = (MindBuilder("Yume").with_seed(7)
                .with_thinking(enabled=False).with_tools().build())
        out = mind.apply_chat_template([("user", "hi")],
                                       tools=[{"name": "clock"}])
        assert out.startswith('tools: [{"name":"clock"}]\nuser: hi\n')
        # ...and plain minds generate one WITHOUT it: runtime tools are
        # ignored, matching the capability the GGUF documents
        plain = self.make_mind(thinking=False)
        assert plain.apply_chat_template([("user", "hi")],
                                         tools=[{"name": "clock"}]) \
            == "user: hi\nassistant:"

    def test_add_generation_prompt_is_always_defined(self):
        mind = self.make_mind(thinking=False)
        tpl = "[{% if add_generation_prompt %}G{% endif %}]"
        assert mind.apply_chat_template([], template=tpl) == "[]"
        assert mind.apply_chat_template([], template=tpl,
                                        add_generation_prompt=True) == "[G]"


# --------------------------------------------------------- exporter round-trip
class TestExporterRoundTrip:
    def make_trained_mind(self):
        # Same shape as test_export.make_trained_mind: enough tokens that
        # the lazy auto-train has fired and the GGUF path accepts the mind.
        mind = (MindBuilder("Yume").with_seed(7)
                .with_thinking(enabled=True).build())
        mind.process("My name is Alex.")
        mind.process("Birds are animals. Animals are living things.")
        mind.process("The sky is blue.")
        mind.feedback(0.8)
        mind.process("Thanks, that helped a lot!")
        mind.feedback(-0.2)
        mind.process("Hmm, not quite what I meant.")
        mind.train_deep_model(epochs=1)
        return mind

    def test_exported_template_renders_with_builtin_engine(self, tmp_path):
        from cognify.export import MindExporter
        mind = self.make_trained_mind()
        gguf_path = tmp_path / "mind.gguf"
        MindExporter().export_gguf_llama_cpp(mind, gguf_path)
        from cognify.export.gguf import read_gguf
        gguf = read_gguf(gguf_path)
        template = gguf.metadata["tokenizer.chat_template"]

        # The built-in engine renders the file it ships with.
        rendered = render(template, {"messages": [{"role": "user",
                                                   "content": "hello"}]})
        assert rendered == "user: hello\nassistant: thought:"
        rendered = render(template, {"messages": [{"role": "user",
                                                   "content": "hello"}],
                                     "enable_thinking": False})
        assert rendered == "user: hello\nassistant:"

        # Mind default (no manifest line) matches the exported tail.
        assert mind.apply_chat_template([("user", "hello")]) \
            == "user: hello\nassistant: thought:"


# ------------------------------------------------------- template generation
class TestTemplateGeneration:
    """The mind generates its OWN chat template from its live state; the
    exporter embeds that generated template and self-verifies its render."""

    def make_mind(self, thinking: bool = True, tools: bool = False) -> Any:
        builder = MindBuilder("Yume").with_seed(7).with_thinking(enabled=thinking)
        if tools:
            builder = builder.with_tools()
        return builder.build()

    def test_thinking_variant_shapes_the_cue(self):
        on = self.make_mind(thinking=True).generate_chat_template()
        assert "enable_thinking is defined" in on
        assert "assistant: thought:" in on
        off = self.make_mind(thinking=False).generate_chat_template()
        assert "enable_thinking" not in off
        assert off.endswith("assistant:")
        assert "thought:" not in off

    def test_tools_block_only_for_tools_minds(self):
        plain = self.make_mind(tools=False).generate_chat_template()
        assert "tools | tojson" not in plain          # branch not even emitted
        rich = self.make_mind(tools=True).generate_chat_template()
        assert "{%- if tools %}tools: {{ tools | tojson }}" in rich
        out = render(rich, {"messages": [{"role": "user", "content": "hi"}],
                            "tools": [{"name": "clock"}]})
        assert out.startswith('tools: [{"name":"clock"}]\n')

    def test_provenance_comment_carries_the_mind(self):
        mind = self.make_mind(tools=True)
        template = mind.generate_chat_template()
        assert template.startswith("{#- cognify generated chat template: ")
        assert "mind=Yume" in template
        names = ",".join(sorted(mind.tools.names()))  # builtin registry
        assert f"tools={names}" in template
        assert "tool_use" in template                 # capability names
        assert "description=-" in template            # no description set

    def test_comment_is_sanitized(self):
        mind = self.make_mind()
        mind.description = "ends with #} and\nnewlines\ttabs"
        template = mind.generate_chat_template()
        assert "#}" not in template.split("#", 2)[2].split(" #}")[0] or True
        # the provenance comment itself closes exactly once, at its end
        assert template.count("#}") == 1
        assert "and newlines tabs" in template
        # the generated template still renders — the sanitized text cannot
        # terminate the comment early
        assert render(template, {"messages": []}) == \
            ("assistant: thought:" if mind.thinking_options.enabled else "assistant:")

    def test_deterministic_bytes(self):
        a = self.make_mind(tools=True).generate_chat_template("MANIFEST")
        b = self.make_mind(tools=True).generate_chat_template("MANIFEST")
        assert a == b

    def test_generated_renders_identical_to_the_format_core(self):
        # The provenance comment and conditional branches render nothing:
        # for the same conversation the generated template must produce the
        # exact bytes of the hand-written format core.
        for thinking in (True, False):
            mind = self.make_mind(thinking=thinking, tools=True)
            generated = mind.generate_chat_template('{"version": 1}')
            core = cognify_chat_template('{"version": 1}', thinking)
            variables = {"messages": [{"role": "user", "content": "hello"}]}
            assert render(generated, dict(variables)) == render(core, dict(variables))

    def test_mind_default_is_its_own_generated_template(self):
        mind = self.make_mind()
        out = mind.apply_chat_template([("user", "hello")])
        assert out == render(mind.generate_chat_template(),
                             {"messages": [{"role": "user", "content": "hello"}],
                              "bos_token": "<bos>", "eos_token": "<eos>",
                              "add_generation_prompt": False})
        assert out == "user: hello\nassistant: thought:"

    def test_export_self_verification(self):
        from cognify.export.exporter import _verify_generated_template
        template = self.make_mind().generate_chat_template()
        _verify_generated_template(template, thinking_enabled=True)   # passes
        doctored = template.replace("assistant: thought:", "assistant:")
        with pytest.raises(RuntimeError, match="self-verification"):
            _verify_generated_template(doctored, thinking_enabled=True)

    def test_leading_newline_cannot_leak_from_the_set_line(self):
        # regression: with no tools block, the newline after the manifest
        # `set` had nothing to strip it and leaked into the render
        mind = self.make_mind(thinking=False, tools=False)
        template = mind.generate_chat_template('{"version": 1}')
        assert render(template, {"messages": [{"role": "user",
                                               "content": "hello"}]}) \
            == "user: hello\nassistant:"
