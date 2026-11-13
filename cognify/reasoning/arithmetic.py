"""ArithmeticEngine: reasoning with TAUGHT arithmetic — no calculator needed.

Mirrors the C# Cognify ArithmeticEngine:
- Teaching: "2 + 2 = 4" / "two plus two is four" become ("2+2", equals, "4")
  facts (only when the stated result is actually correct).
- Questions: "what is 2 + 2?" is looked up first; unknown ones are DERIVED
  from related taught facts (counting on via "n+1" facts, chunked jumps,
  commutativity, repeated addition/subtraction).
- Word problems: "if I have 1 apple and buy another one, how many apples do
  I have?" — quantities + transfer verbs, evaluated step by step from
  taught knowledge.

Only when this engine comes up empty does the caller (tool capability) fall
back to the calculator — and it says so transparently when it does.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .facts import Fact

MAX_STEPS = 32

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_WORD_ALT = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
_NUMBER_RUN = re.compile(rf"\b(?P<run>{_WORD_ALT})(?:[\s-](?P<run2>{_WORD_ALT}))*\b")
_MULTIWORD_OPS = re.compile(r"\b(?:multiplied by|divided by)\b", re.I)
_AND_BETWEEN_NUMBERS = re.compile(r"(?<=\d)\s+and\s+(?=\d)")
_X_BETWEEN_NUMBERS = re.compile(r"(?<=\d)\s+x\s+(?=\d)", re.I)

_TEACH = re.compile(
    r"^(?:the answer to\s+)?(?P<a>-?\d+(?:\.\d+)?)\s*(?P<op>[+\-*/])\s*(?P<b>-?\d+(?:\.\d+)?)\s*"
    r"(?:=|is|equals|makes|gives|will be)\s*(?P<r>-?\d+(?:\.\d+)?)$", re.I)
_EXPRESSION = re.compile(r"(?P<a>-?\d+(?:\.\d+)?)\s*(?P<op>[+\-*/])\s*(?P<b>-?\d+(?:\.\d+)?)")
_QUESTIONISH = re.compile(r"\b(what|whats|how much|how many|calculate|compute|solve|tell me)\b|\?", re.I)
_BARE = re.compile(r"^-?\d+(?:\.\d+)?(?:\s*[+\-*/]\s*-?\d+(?:\.\d+)?)+$")

_HOW_MANY = re.compile(r"\bhow (?:many|much|may)\b", re.I)   # "how may" is a common typo
_INITIAL = re.compile(
    r"\b(?:i have|we have|i(?:'ve)? got|you have|there are|there is|i started with|i start with|she has|he has)\s+"
    r"(?P<n>-?\d+(?:\.\d+)?|no\b|zero\b|an?\b)(?:\s+(?P<unit>[a-z]+))?", re.I)
_GAIN = re.compile(
    r"\b(?:buy|bought|get|gets|got|receive|receives|received|find|finds|found|gain|gains|gained|"
    r"add|adds|added|make|makes|made|bake|bakes|baked|pick|picks|picked)\s+"
    r"(?:another(?:\s+one)?|one more|(?P<n>\d+)(?:\s+more)?|"
    r"an?\s+(?!lot\b|few\b|couple\b|bit\b|number\b|handful\b)[a-z]+|one\b)", re.I)
_LOSE = re.compile(
    r"\b(?:eat|eats|ate|lose|loses|lost|give away|gives away|gave away|sell|sells|sold|"
    r"break|breaks|broke|drop|drops|dropped)\s+"
    r"(?:(?P<n>\d+(?:\.\d+)?)|an?\s+(?!lot\b|few\b|couple\b|bit\b)[a-z]+|one\b)", re.I)
_NON_UNITS = {"of", "more", "less", "them", "it", "to", "and",
              "idea", "clue", "doubt", "chance"}


@dataclass
class ArithmeticResult:
    """Outcome of an arithmetic reasoning pass."""
    expression: str
    result: str
    reply: str
    directly_taught: bool
    confidence: float
    steps: Tuple[str, ...] = field(default_factory=tuple)


def _num(value: float) -> str:
    r = round(value)
    if abs(value - r) < 1e-9:
        return str(int(r))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _parse_num(text: str) -> float:
    return float(text)


def _is_countable(value: float) -> bool:
    return abs(value - round(value)) < 1e-9 and abs(value) <= 1000


def _key(a: float, op: str, b: float) -> str:
    return f"{_num(a)}{op}{_num(b)}"


def _display(expr: str) -> str:
    return (expr.replace("+", " + ").replace("-", " - ")
                .replace("*", " * ").replace("/", " / ")
                .replace("  ", " ").strip())


def _plural(unit: str, amount: str) -> str:
    if amount == "1" and unit.endswith("s") and not unit.endswith("ss"):
        return unit[:-1]
    if amount != "1" and not unit.endswith("s"):
        return unit + "s"
    return unit


def _quantity(raw: str) -> float:
    """"no" -> 0, "a"/"an" -> 1, digits parse directly."""
    if raw in ("no", "zero"):
        return 0.0
    if raw in ("a", "an"):
        return 1.0
    return float(raw)


def normalize(text: str) -> str:
    """Lowercase + number words to digits + operator words to symbols."""
    if not text or not text.strip():
        return ""
    lower = text.lower().replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-")

    lower = _MULTIWORD_OPS.sub(
        lambda m: " * " if m.group(0).lower().startswith("multiplied") else " / ", lower)

    def words_to_digits(m: "re.Match[str]") -> str:
        words = [w for w in re.split(r"[\s-]+", m.group(0)) if w]
        if not all(w in _NUMBER_WORDS for w in words):
            return m.group(0)
        total, current = 0, 0
        for w in words:
            v = _NUMBER_WORDS[w]
            if v < 10 and current >= 20:
                current += v                      # twenty-one
            else:
                total += current
                current = v
        return f" {total + current} "

    lower = _NUMBER_RUN.sub(words_to_digits, lower)
    lower = _AND_BETWEEN_NUMBERS.sub(" + ", lower)
    lower = _X_BETWEEN_NUMBERS.sub(" * ", lower)
    lower = re.sub(r"\b(plus)\b", " + ", lower)
    lower = re.sub(r"\b(minus|less)\b", " - ", lower)
    lower = re.sub(r"\btimes\b", " * ", lower)
    lower = re.sub(r"\bover\b", " / ", lower)
    return re.sub(r"\s+", " ", lower).strip()


def try_teach(sentence: str) -> Optional[Fact]:
    """Parse a taught arithmetic statement into a (\"2+2\", equals, \"4\") fact."""
    if not sentence or not sentence.strip():
        return None
    text = normalize(sentence.strip().rstrip(".,!?"))
    m = _TEACH.match(text)
    if not m:
        return None
    a, b, r = (_parse_num(m.group(g)) for g in ("a", "b", "r"))
    op = m.group("op")
    expected = {"+": a + b, "-": a - b, "*": a * b, "/": (a / b if b else float("nan"))}.get(op, float("nan"))
    if expected != expected or abs(expected - r) > 1e-9:
        return None
    return Fact(_key(a, op, b), "equals", _num(r), 0.95, "taught-arithmetic")


def looks_like_arithmetic_statement(sentence: str) -> bool:
    """True when a statement carries an arithmetic expression (teaching or mistake)."""
    if not sentence or not sentence.strip():
        return False
    return bool(_EXPRESSION.search(normalize(sentence)))


class ArithmeticEngine:
    """Answers arithmetic from taught facts only; derivation on top of them.

    A taught number line ("0, 1, 2, 3, 4, 5, 6, 7, 8, 9") becomes the mind's
    counting ladder — counting along it is reasoning from teaching, so sums
    that stay on the line never need the calculator.
    """

    def __init__(self) -> None:
        self._ladder: Optional[Tuple[float, float]] = None

    def looks_like_arithmetic_question(self, text: str) -> bool:
        return bool(text and text.strip()
                    and _EXPRESSION.search(normalize(text))
                    and _QUESTIONISH.search(text))

    def looks_like_word_problem(self, text: str) -> bool:
        return bool(text and text.strip() and _HOW_MANY.search(text))

    # ------------------------------------------------------------- questions
    def try_answer_question(self, kb, text: str) -> Optional[ArithmeticResult]:
        if not text or not text.strip():
            return None
        normalized = normalize(text)
        if not _QUESTIONISH.search(text) and not _BARE.match(normalized.rstrip("?. ")):
            return None

        best = None
        for m in _EXPRESSION.finditer(normalized):
            if best is None or m.end() - m.start() > best.end() - best.start():
                best = m
        if best is None:
            return None

        a, b = _parse_num(best.group("a")), _parse_num(best.group("b"))
        op = best.group("op")
        expr = _key(a, op, b)

        taught = self._lookup(kb, expr)
        if taught is not None:
            return ArithmeticResult(expr, taught,
                                    f"{_display(expr)} is {taught} - you taught me that one.",
                                    True, 0.95)

        if op == "+":
            return self._derive_add(kb, expr, a, b)
        if op == "-":
            return self._derive_sub(kb, expr, a, b)
        if op == "*":
            return self._derive_mul(kb, expr, a, b)
        if op == "/":
            return self._derive_div(kb, expr, a, b)
        return None

    # --------------------------------------------------------- word problems
    def _plan_word_problem(self, text: str):
        """Extracts a word problem's setup: (start, explicit_start, unit,
        events) with events in reading order — or None when the text has no
        "how many" question or no starting quantity at all."""
        if not text or not _HOW_MANY.search(text):
            return None
        lowered = normalize(text)

        # Several candidate starts may appear ("there is a 10 based number
        # system ... if i have no apples"). Prefer explicit quantities over
        # bare articles, then unit-bearing matches, then the later one — the
        # real setup usually sits closest to the "how many" question.
        init = None
        init_score = -1
        for m in _INITIAL.finditer(lowered):
            score = self._start_score(m)
            if score > init_score or (score == init_score and m.start() > init.start()):
                init, init_score = m, score
        if init is None:
            return None

        start = _quantity(init.group("n"))
        explicit_start = init.group("n")[0].isdigit() or init.group("n") in ("no", "zero")
        unit = init.group("unit")
        if unit is not None and (unit in _NON_UNITS or len(unit) < 3):
            unit = None

        events: List[Tuple[int, str, float, str]] = []
        for m in _GAIN.finditer(lowered):
            n = _parse_num(m.group("n")) if m.group("n") else 1.0
            events.append((m.start(), "+", n, m.group(0).strip()))
        for m in _LOSE.finditer(lowered):
            n = _parse_num(m.group("n")) if m.group("n") else 1.0
            events.append((m.start(), "-", n, m.group(0).strip()))
        events.sort(key=lambda e: e[0])
        return start, explicit_start, unit, events

    @staticmethod
    def _stated_quantity_result(start: float, unit) -> ArithmeticResult:
        unit0 = f" {_plural(unit, _num(start))}" if unit else ""
        return ArithmeticResult(
            _num(start), _num(start),
            f"You start with {_num(start)}{unit0}, so you still have "
            f"{_num(start)}{unit0} - that is what you told me.",
            False, 0.7)

    def try_word_problem(self, kb, text: str) -> Optional[ArithmeticResult]:
        plan = self._plan_word_problem(text)
        if plan is None:
            return None
        start, explicit_start, unit, events = plan

        # No transfers at all: "I have 3 apples, how many apples do I have?"
        # is answered from the stated quantity — but only when it was stated
        # explicitly ("a dog" alone is too weak to answer from).
        if not events:
            return self._stated_quantity_result(start, unit) if explicit_start else None

        steps: List[str] = []
        cur = start
        for _, op, amount, label in events:
            expr = _key(cur, op, amount)
            value = self._lookup(kb, expr)
            if value is None:
                derived = (self._derive_add(kb, expr, cur, amount) if op == "+"
                           else self._derive_sub(kb, expr, cur, amount))
                if derived is None:
                    return None
                value = derived.result
                steps.extend(derived.steps)
            unit_phrase = f" {_plural(unit, value)}" if unit else ""
            steps.append(f"{label}: {_num(cur)} {op} {_num(amount)} = {value}{unit_phrase}")
            cur = float(value)

        start_unit = f" {_plural(unit, _num(start))}" if unit else ""
        result_unit = f" {_plural(unit, _num(cur))}" if unit else ""
        reply = (f"You start with {_num(start)}{start_unit}, and all of that works out to "
                 f"{_num(cur)}{result_unit} - {steps[-1]}.")
        return ArithmeticResult(
            " ".join(f"{op}{_num(n)}" for _, op, n, _ in events),
            _num(cur), reply, False, 0.75, tuple(steps))

    def try_word_problem_mechanically(self, text: str) -> Optional[ArithmeticResult]:
        """Solves a word problem the calculator's way: same extraction as
        try_word_problem, but every step is plain arithmetic instead of
        taught knowledge. This is the transparent fallback the tool phase
        uses when taught reasoning could not work the problem out — the mind
        says openly that it leaned on the calculator."""
        plan = self._plan_word_problem(text)
        if plan is None:
            return None
        start, explicit_start, unit, events = plan

        if not events:
            return self._stated_quantity_result(start, unit) if explicit_start else None

        steps: List[str] = []
        cur = start
        for _, op, amount, label in events:
            nxt = cur + amount if op == "+" else cur - amount
            unit_phrase = f" {_plural(unit, _num(nxt))}" if unit else ""
            steps.append(f"{label}: {_num(cur)} {op} {_num(amount)} = {_num(nxt)}{unit_phrase}")
            cur = nxt

        start_unit = f" {_plural(unit, _num(start))}" if unit else ""
        result_unit = f" {_plural(unit, _num(cur))}" if unit else ""
        reply = (f"You start with {_num(start)}{start_unit}, and all of that works out to "
                 f"{_num(cur)}{result_unit} - {steps[-1]}.")
        return ArithmeticResult(
            " ".join(f"{op}{_num(n)}" for _, op, n, _ in events),
            _num(cur), reply, False, 0.6, tuple(steps))

    @staticmethod
    def _start_score(m: "re.Match[str]") -> int:
        """Explicit quantities (digits, "no") beat bare articles; units help."""
        n = m.group("n")
        score = 2 if (n[0].isdigit() or n in ("no", "zero")) else 0
        unit = m.group("unit")
        if unit is not None and unit not in _NON_UNITS and len(unit) >= 3:
            score += 1
        return score

    # ----------------------------------------------------- the counting ladder
    def teach_sequence(self, text: str) -> bool:
        """Remember a taught counting sequence ("0, 1, 2, ..., 9") as the
        mind's number line. At least five numbers running upward by one."""
        if not text or not text.strip():
            return False
        taught = False
        for part in re.split(r"(?<=[.!?;])\s+", text.strip()):
            tokens = [float(m.group(0)) for m in
                      re.finditer(r"-?\d+(?:\.\d+)?", normalize(part))]
            i = 0
            while i < len(tokens):
                j = i
                while j + 1 < len(tokens) and abs(tokens[j + 1] - tokens[j] - 1) < 1e-9:
                    j += 1
                if j - i + 1 >= 5 and _is_countable(tokens[i]) and _is_countable(tokens[j]):
                    self._remember_ladder(tokens[i], tokens[j])
                    taught = True
                i = j + 1
        return taught

    def _remember_ladder(self, minimum: float, maximum: float) -> None:
        # Overlapping or touching ranges extend the line; a disjoint run is
        # a new number line and replaces the old one.
        if self._ladder is None or minimum > self._ladder[1] + 1 or maximum < self._ladder[0] - 1:
            self._ladder = (minimum, maximum)
        else:
            self._ladder = (min(self._ladder[0], minimum), max(self._ladder[1], maximum))

    # ------------------------------------------------------------ derivation
    def _lookup(self, kb, expr: str) -> Optional[str]:
        facts = kb.about(expr, "equals")
        return facts[0].object if facts else None

    def _find_jump(self, kb, cur: float, op: str, maximum: float) -> Optional[Tuple[float, float]]:
        k = float(int(maximum))
        while k >= 1:
            value = self._lookup(kb, _key(cur, op, k))
            if value is not None:
                return k, float(value)
            if op == "-":
                inverse = self._lookup_inverse(kb, cur, k)
                if inverse is not None:
                    return k, float(inverse)
            if k == 1 and self._ladder is not None and _is_countable(cur):
                lo, hi = self._ladder
                if lo - 1e-9 <= cur <= hi + 1e-9:
                    step = cur + 1 if op == "+" else cur - 1
                    if lo - 1e-9 <= step <= hi + 1e-9:
                        return 1.0, step
            k -= 1
        return None

    def _lookup_inverse(self, kb, cur: float, k: float) -> Optional[str]:
        """Taught "p + j = cur" gives cur - j = p and cur - p = j."""
        for f in kb.facts():
            if f.predicate != "equals" or f.object != _num(cur):
                continue
            m = re.match(r"^(?P<p>-?\d+(?:\.\d+)?)\+(?P<j>-?\d+(?:\.\d+)?)$", f.subject)
            if not m:
                continue
            p, j = _parse_num(m.group("p")), _parse_num(m.group("j"))
            if abs(j - k) < 1e-9:
                return _num(p)
            if abs(p - k) < 1e-9:
                return _num(j)
        return None

    def _derive_add(self, kb, expr: str, a: float, b: float) -> Optional[ArithmeticResult]:
        steps: List[str] = []
        if _is_countable(a) and _is_countable(b) and b >= 0:
            if b == 0:
                return ArithmeticResult(expr, _num(a),
                                        f"{_display(expr)} is {_num(a)} - adding zero changes nothing.",
                                        False, 0.7)
            # "no apples and i buy an apple": starting from NOTHING and adding
            # b leaves b. Zero is not an unknown quantity here — the word
            # problem itself stated it ("no apples" = none), so this is an
            # identity the mind can reason out without ever being taught it.
            if a == 0:
                steps.append(f"{_num(0)} + {_num(b)} = {_num(b)} (starting from nothing)")
                return ArithmeticResult(
                    expr, _num(b),
                    f"{_display(expr)} is {_num(b)} - starting from nothing and "
                    f"adding {_num(b)} gives {_num(b)}.",
                    False, 0.8, tuple(steps))
            cur, remaining = a, b
            while remaining > 0 and len(steps) < MAX_STEPS:
                jump = self._find_jump(kb, cur, "+", remaining)
                if jump is None:
                    break
                k, landed = jump
                steps.append(f"{_num(cur)} + {_num(k)} = {_num(landed)}"
                             + (" (counting on)" if k == 1 else ""))
                cur, remaining = landed, remaining - k
            if remaining == 0 and steps:
                return ArithmeticResult(
                    expr, _num(cur),
                    f"{_display(expr)} is {_num(cur)} - I worked it out from what you "
                    f"taught me: " + ", then ".join(steps) + ".",
                    False, 0.8, tuple(steps))

        flipped = _key(b, "+", a)
        mirror = self._lookup(kb, flipped)
        if mirror is not None:
            return ArithmeticResult(
                expr, mirror,
                f"{_display(expr)} is {mirror} - same as {_display(flipped)}, which you taught me.",
                False, 0.78)
        return None

    def _derive_sub(self, kb, expr: str, a: float, b: float) -> Optional[ArithmeticResult]:
        if not (_is_countable(a) and _is_countable(b)) or a < 0 or b < 0:
            return None
        if b == 0:
            return ArithmeticResult(expr, _num(a),
                                    f"{_display(expr)} is {_num(a)} - taking zero away changes nothing.",
                                    False, 0.7)
        steps: List[str] = []
        cur, remaining = a, b
        while remaining > 0 and len(steps) < MAX_STEPS:
            jump = self._find_jump(kb, cur, "-", remaining)
            if jump is None:
                return None
            k, landed = jump
            steps.append(f"{_num(cur)} - {_num(k)} = {_num(landed)}")
            cur, remaining = landed, remaining - k
        if not steps:
            return None
        return ArithmeticResult(
            expr, _num(cur),
            f"{_display(expr)} is {_num(cur)} - I worked backwards from what you "
            f"taught me: " + ", then ".join(steps) + ".",
            False, 0.8, tuple(steps))

    def _derive_mul(self, kb, expr: str, a: float, b: float) -> Optional[ArithmeticResult]:
        if not (_is_countable(a) and _is_countable(b)) or b < 0 or b > 12 or abs(a) > 1000:
            return None
        if b == 0:
            return ArithmeticResult(expr, "0",
                                    f"{_display(expr)} is 0 - {_num(a)} added together zero times is still 0.",
                                    False, 0.7)
        steps: List[str] = []
        cur = a
        for _ in range(2, int(b) + 1):
            add = self._derive_add(kb, _key(cur, "+", a), cur, a)
            if add is None:
                return None
            steps.append(f"{_num(cur)} + {_num(a)} = {add.result}")
            cur = float(add.result)
        return ArithmeticResult(
            expr, _num(cur),
            f"{_display(expr)} is {_num(cur)} - that is {_num(a)} added together "
            f"{_num(b)} times: " + ", ".join(steps) + ".",
            False, 0.75, tuple(steps))

    def _derive_div(self, kb, expr: str, a: float, b: float) -> Optional[ArithmeticResult]:
        if not (_is_countable(a) and _is_countable(b)) or b <= 0 or a < 0 or abs(a) > 400:
            return None
        cur, times = a, 0
        while cur > 0 and times <= MAX_STEPS:
            if cur < b:
                return None
            sub = self._derive_sub(kb, _key(cur, "-", b), cur, b)
            if sub is None:
                return None
            cur, times = float(sub.result), times + 1
        if cur != 0:
            return None
        return ArithmeticResult(
            expr, _num(times),
            f"{_display(expr)} is {_num(times)} - I took {_num(b)} away from {_num(a)} "
            f"{_num(times)} time(s) until nothing was left.",
            False, 0.7)
