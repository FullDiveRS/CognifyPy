"""ToolUse: function-calling with registry, intent matching, and safe exec."""
from __future__ import annotations

import ast
import datetime
import inspect
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.stimulus import Stimulus
from ..util import tokenize
from .base import Capability, ToolInvocation

_MATH_RE = re.compile(r"(\d[\d\s\.,]*[\+\-\*/x\^%]+\s*\d)")
_TOOL_CALL_RE = re.compile(r"@(\w+)\((.*)\)")
# A randomness cue must accompany a "random number" tool match — the word
# "number" alone appears in almost every arithmetic lesson.
_RANDOM_CUE_RE = re.compile(r"\b(random|randomly|dice|roll|coin|flip|toss|lottery|shuffle)\b", re.I)
_TIME_RE = re.compile(r"\b(what(?:'s| is) the )?(current |now |)time\b|\bwhat('| i)?s the date\b|\btoday'?s? date\b", re.I)
_CONVERT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(km|mi|miles?|kg|lbs?|pounds?|c|celsius|f|fahrenheit|°?c|°?f)\b.*\b(?:in|to|as)\s+(km|mi|miles?|kg|lbs?|pounds?|c|celsius|f|fahrenheit|°?c|°?f)\b", re.I)


class SafeMathError(ValueError):
    pass


def safe_math(expression: str) -> float:
    """Evaluate arithmetic via a whitelisted AST walk (no eval, no names)."""
    node = ast.parse(expression.strip(), mode="eval")

    def visit(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return visit(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult,
                                                          ast.Div, ast.Pow, ast.Mod,
                                                          ast.FloorDiv)):
            left, right = visit(n.left), visit(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                if right == 0:
                    raise SafeMathError("division by zero")
                return left / right
            if isinstance(n.op, ast.Mod):
                return left % right
            return left ** right
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = visit(n.operand)
            return v if isinstance(n.op, ast.UAdd) else -v
        raise SafeMathError(f"disallowed expression element: {type(n).__name__}")

    return visit(node)


@dataclass
class ToolSpec:
    name: str
    description: str
    params: Dict[str, str]
    func: Callable[..., Any]

    def keywords(self) -> List[str]:
        toks = tokenize(self.name) + tokenize(self.description)
        return [t for t in toks if len(t) > 2]


def tool(func: Optional[Callable] = None, *, name: str = "",
         description: str = ""):
    """Decorator that marks a function as a mind-tool (introspects signature)."""
    def wrap(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        params = {p.name: (p.annotation.__name__ if p.annotation is not inspect.Parameter.empty
                           and isinstance(p.annotation, type) else "any")
                  for p in sig.parameters.values()}
        fn.__tool__ = ToolSpec(name=name or fn.__name__.replace("_", " "),
                               description=description or (fn.__doc__ or "").strip().split("\n")[0],
                               params=params, func=fn)
        return fn
    return wrap(func) if func is not None else wrap


class ToolRegistry:
    """Holds ToolSpecs; the planner matches stimuli against descriptions."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, func_or_spec) -> ToolSpec:
        if isinstance(func_or_spec, ToolSpec):
            spec = func_or_spec
        elif hasattr(func_or_spec, "__tool__"):
            spec = func_or_spec.__tool__
        else:
            raise TypeError("register a @tool-decorated function or a ToolSpec")
        self._tools[spec.name] = spec
        return spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return sorted(self._tools.keys())

    def all(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def match(self, text: str, threshold: float = 0.2) -> List[ToolSpec]:
        """Rank tools by keyword overlap with the text."""
        toks = set(tokenize(text))
        scored: List[Tuple[float, ToolSpec]] = []
        for spec in self._tools.values():
            kws = set(spec.keywords())
            if not kws:
                continue
            matched = len(toks & kws)
            overlap = matched / max(1, min(len(kws), 4))
            if matched and overlap >= threshold:
                scored.append((overlap, spec))
        scored.sort(key=lambda p: -p[0])
        return [s for _, s in scored]


# --------------------------------------------------------------------- builtins
def _calculator(expression: str = "") -> str:
    expr = expression.replace("x", "*").replace("^", "**")
    value = safe_math(expr)
    return f"{expr.strip()} = {value:g}"


def _current_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _random_number(low: float = 1, high: float = 100) -> int:
    if low > high:
        low, high = high, low   # a guessed/inverted range still rolls
    return random.randint(int(low), int(high))


def _text_stats(text: str) -> str:
    words = len(text.split())
    chars = len(text)
    return f"{words} words, {chars} characters"


_CONVERSIONS = {
    ("km", "mi"): 0.621371, ("mi", "km"): 1.609344,
    ("kg", "lb"): 2.20462, ("lb", "kg"): 0.453592,
}

def _norm_unit(u: str) -> str:
    u = u.strip().lower().replace("°", "")
    if u.startswith("mile"):
        return "mi"
    if u.startswith("pound"):
        return "lb"
    return u[:2]


def _unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    f, t = _norm_unit(from_unit), _norm_unit(to_unit)
    if f == t:
        return f"{value:g} {t} = {value:g} {t}"
    if f in ("c", "f") and t in ("c", "f"):
        if f == "c" and t == "f":
            return f"{value:g}C = {value * 9 / 5 + 32:g}F"
        return f"{value:g}F = {(value - 32) * 5 / 9:g}C"
    pair = (f, t)
    if pair in _CONVERSIONS:
        return f"{value:g} {f} = {value * _CONVERSIONS[pair]:g} {t}"
    raise ValueError(f"no conversion {f} -> {t}")


class ToolUseCapability(Capability):
    """Plans and executes tool calls grounded in the registry."""

    name = "tool_use"
    description = "Decides when to call registered tools and runs them safely."
    phase = "tools"

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def can_handle(self, stimulus: Stimulus, ctx=None) -> bool:
        return self.plan(stimulus, ctx) != []

    @staticmethod
    def _explicit_calculator_request(text: str) -> bool:
        lower = (text or "").lower()
        return ("calculator" in lower
                or lower.startswith("calculate")
                or lower.startswith("compute"))

    def plan(self, stimulus: Stimulus, ctx=None) -> List[Tuple[str, Dict[str, Any], str]]:
        """Return [(tool_name, args, reason)] planned invocations."""
        text = stimulus.text or ""
        plans: List[Tuple[str, Dict[str, Any], str]] = []
        if not text:
            return plans

        # explicit @tool(arg) syntax
        m = _TOOL_CALL_RE.search(text)
        if m:
            spec = self.registry.get(m.group(1))
            if spec:
                args = self._parse_args(m.group(2), spec)
                plans.append((spec.name, args, "explicit @tool call"))
                return plans

        # Reasoning-first arithmetic: when the reasoning step already answered
        # a math question from taught knowledge, the calculator must NOT fire
        # on top of it — unless the user explicitly asks for the calculator.
        math_answered = bool(ctx is not None and ctx.scratch.get("math_answered"))
        explicit_calc = self._explicit_calculator_request(text)

        # arithmetic expression -> calculator
        if _MATH_RE.search(text):
            expr = self._extract_expression(text)
            if expr and (not math_answered or explicit_calc):
                plans.append(("calculator", {"expression": expr}, "arithmetic detected"))
                return plans

        # unit conversion
        m = _CONVERT_RE.search(text)
        if m:
            plans.append(("unit convert", {"value": float(m.group(1)),
                                           "from_unit": m.group(2),
                                           "to_unit": m.group(3)},
                          "unit conversion requested"))
            return plans

        # time / date
        if _TIME_RE.search(text):
            plans.append(("current time", {}, "time question"))
            return plans

        # description-keyword match
        for spec in self.registry.match(text)[:1]:
            args = self._guess_args(text, spec)
            if spec.name == "calculator":
                # Only a real, extracted expression counts ("2 + 3", or a
                # quoted one). A word problem ("...1 apple and buy another
                # one, how many...") matched on keywords must never reach
                # the calculator with empty args.
                expr_arg = args.get("expression")
                if not (isinstance(expr_arg, str) and expr_arg.strip()
                        and (_MATH_RE.search(expr_arg)
                             or re.search(r"\d\s*[\+\-\*/x\^]", expr_arg))):
                    continue
            if spec.name == "random number" and not _RANDOM_CUE_RE.search(text):
                # "number" appears in almost every math lesson — a keyword
                # match alone must not roll dice with two numbers scavenged
                # from the text ("10 based number system, starts at 0").
                continue
            plans.append((spec.name, args, "matched tool description"))
        return plans

    def handle(self, stimulus: Stimulus, ctx=None) -> Optional[Any]:
        """Execute planned tools; records ToolInvocations into ctx."""
        results: List[ToolInvocation] = []
        math_answered = bool(ctx is not None and ctx.scratch.get("math_answered"))
        math_attempted = bool(ctx is not None and ctx.scratch.get("math_attempted"))
        explicit_calc = self._explicit_calculator_request(stimulus.text or "")
        for name, args, _reason in self.plan(stimulus, ctx):
            spec = self.registry.get(name)
            if spec is None:
                results.append(ToolInvocation(name, args, success=False,
                                              error="unknown tool"))
                continue
            try:
                value = spec.func(**args)
                if (name == "calculator" and value is not None
                        and math_attempted and not explicit_calc and not math_answered):
                    # transparency beats a silent crutch
                    value = (f"I have not been taught that one yet, so I used "
                             f"the calculator: {value}")
                results.append(ToolInvocation(spec.name, args, value, True))
            except Exception as exc:            # noqa: BLE001 - tools are untrusted
                results.append(ToolInvocation(spec.name, args, None, False, str(exc)))
        if ctx is not None:
            ctx.tool_invocations.extend(results)
        return results[0].summary() if results else None

    # ------------------------------------------------------------------ arg extraction
    @staticmethod
    def _extract_expression(text: str) -> str:
        # must start with a digit so we don't capture stray spaces/words
        m = re.search(r"\d[\d\s\.,\+\-\*/x\^%()]*", text)
        if not m:
            return ""
        expr = m.group(0).strip()
        return expr if _MATH_RE.search(expr) or re.search(r"\d\s*[\+\-\*/x\^]", expr) else ""

    @staticmethod
    def _parse_args(raw: str, spec: ToolSpec) -> Dict[str, Any]:
        """Parse 'k=v, k2="v2"' or positional arguments."""
        args: Dict[str, Any] = {}
        if not raw.strip():
            return args
        tokens = re.findall(r'"([^"]*)"|\'([^\']*)\'|([^\s,]+)', raw)
        values = [a or b or c for a, b, c in tokens]
        param_names = list(spec.params.keys())
        idx = 0
        for v in values:
            if "=" in v and v.split("=", 1)[0] in param_names:
                k, val = v.split("=", 1)
                args[k] = _coerce(val)
            elif idx < len(param_names):
                args[param_names[idx]] = _coerce(v)
                idx += 1
        return args

    @staticmethod
    def _guess_args(text: str, spec: ToolSpec) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        numbers = [float(n.replace(",", "")) for n in re.findall(r"-?\d+(?:\.\d+)?", text)]
        for pname in spec.params:
            if pname in ("low", "high") and len(numbers) >= 2:
                args["low"], args["high"] = min(numbers[:2]), max(numbers[:2])
            elif pname in ("expression", "text", "query"):
                quoted = re.search(r'"([^"]+)"', text)
                if quoted:
                    args[pname] = quoted.group(1)
                elif pname == "expression" and numbers:
                    continue
                elif pname == "text" and len(text.split()) > 1:
                    args[pname] = text
        return args


def _coerce(v: str) -> Any:
    v = v.strip().strip('"').strip("'")
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return v


def builtin_tools() -> List[Callable]:
    """The default toolset registered when tool_use is enabled."""
    calc = tool(_calculator, name="calculator",
                description="calculate arithmetic expressions and math")
    now = tool(_current_time, name="current time",
               description="current time and today's date clock")
    rnd = tool(_random_number, name="random number",
               description="random number between low and high dice roll")
    stats = tool(_text_stats, name="text stats",
                 description="count words and characters of text")
    conv = tool(_unit_convert, name="unit convert",
                description="convert units km mi kg lb celsius fahrenheit")
    return [calc, now, rnd, stats, conv]
