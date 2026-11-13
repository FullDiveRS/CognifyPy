"""Code capability: real AST analysis, linting, and template generation."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.stimulus import Stimulus
from .base import Capability, CodePercept

LANG_HINTS: List[Tuple[str, List[str]]] = [
    ("python", ["def ", "import ", "from ", "print(", "elif", "__name__", "self.", ":"]),
    ("javascript", ["function ", "const ", "let ", "=>", "console.log", "var "]),
    ("c#", ["using ", "namespace ", "public class", "Console.WriteLine", "static void"]),
    ("java", ["public class", "System.out.println", "static void main"]),
    ("sql", ["select ", "from ", "where ", "insert into", "create table"]),
    ("json", ["{", "}", ":"]),
]

_GENERATORS: List[Tuple[List[str], str, str]] = [
    (["add", "sum", "plus"], "add_numbers",
     'def add_numbers(a, b):\n    """Return the sum of a and b."""\n    return a + b'),
    (["reverse", "backwards"], "reverse_string",
     'def reverse_string(s):\n    """Return s reversed."""\n    return s[::-1]'),
    (["palindrome"], "is_palindrome",
     'def is_palindrome(s):\n    """Return True if s reads the same both ways."""\n'
     '    s = s.lower().replace(" ", "")\n    return s == s[::-1]'),
    (["fibonacci", "fib"], "fibonacci",
     'def fibonacci(n):\n    """Return the n-th Fibonacci number (0-indexed)."""\n'
     '    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a'),
    (["prime"], "is_prime",
     'def is_prime(n):\n    """Return True if n is a prime number."""\n'
     '    if n < 2:\n        return False\n'
     '    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n'
     '    return True'),
    (["count", "words"], "count_words",
     'def count_words(text):\n    """Count whitespace-separated words in text."""\n'
     '    return len(text.split())'),
    (["factorial"], "factorial",
     'def factorial(n):\n    """Return n! for non-negative integers."""\n'
     '    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result'),
    (["maximum", "max", "largest", "biggest"], "find_max",
     'def find_max(values):\n    """Return the largest value in a non-empty list."""\n'
     '    best = values[0]\n    for v in values[1:]:\n        if v > best:\n            best = v\n    return best'),
    (["average", "mean"], "average",
     'def average(values):\n    """Return the arithmetic mean of a list."""\n'
     '    return sum(values) / len(values) if values else 0'),
    (["greet", "hello", "welcome"], "greet",
     'def greet(name):\n    """Return a friendly greeting."""\n    return f"Hello, {name}!"'),
    (["sort", "order"], "sort_values",
     'def sort_values(values):\n    """Return a new sorted list (ascending)."""\n'
     '    return sorted(values)'),
    (["capitalize", "title"], "capitalize_words",
     'def capitalize_words(text):\n    """Capitalize each word in text."""\n'
     '    return " ".join(w.capitalize() for w in text.split())'),
]


@dataclass
class CodeAnalysis:
    language: str = "unknown"
    lines: int = 0
    functions: List[Dict[str, Any]] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    quality: int = 100

    def summary(self) -> str:
        if self.language == "python" and self.functions:
            fn_names = ", ".join(f["name"] for f in self.functions[:3])
            return (f"Python code with {len(self.functions)} function(s) "
                    f"({fn_names}), quality {self.quality}/100.")
        return (f"{self.language} code, {self.lines} lines, "
                f"quality {self.quality}/100.")


def detect_language(code: str) -> str:
    for lang, hints in LANG_HINTS:
        hits = sum(1 for h in hints if h in code)
        if hits >= 2:
            return lang
    return "unknown"


def analyze_python(code: str) -> CodeAnalysis:
    """Genuine AST-based analysis + lightweight lint for Python code."""
    result = CodeAnalysis(language="python", lines=len(code.splitlines()))
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        result.issues.append(f"syntax error: {exc.msg} (line {exc.lineno})")
        result.quality = max(0, 60 - 10 * len(result.issues))
        return result

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            has_doc = ast.get_docstring(node) is not None
            result.functions.append({"name": node.name, "args": args,
                                     "docstring": has_doc,
                                     "lines": (node.end_lineno or node.lineno) - node.lineno + 1})
            if not has_doc:
                result.issues.append(f"function '{node.name}' has no docstring")
        elif isinstance(node, ast.ClassDef):
            result.classes.append(node.name)
        elif isinstance(node, ast.Import):
            result.imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            result.imports.append(node.module or ".")
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            result.issues.append("bare 'except:' — catch specific exceptions")
        elif isinstance(node, ast.arguments):
            for default in node.defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    result.issues.append("mutable default argument")

    for i, line in enumerate(code.splitlines(), 1):
        if len(line) > 100:
            result.issues.append(f"line {i} exceeds 100 characters")
        if line.rstrip() != line:
            result.issues.append(f"line {i} has trailing whitespace")

    result.quality = max(0, 100 - 12 * len(result.issues))
    return result


class CodeCapability(Capability):
    """Detects, analyzes, explains, and generates small pieces of code."""

    name = "code"
    description = "Understands and writes small programs (real AST analysis)."
    phase = "perceive"

    def can_handle(self, stimulus: Stimulus, ctx=None) -> bool:
        text = stimulus.text or ""
        if stimulus.modality.value == "code":
            return True
        wants_generation = re.search(
            r"\b(write|create|generate|make)\b.*\b(function|program|code|script)\b", text, re.I)
        wants_analysis = "```" in text or self._looks_like_code(text)
        return bool(wants_generation or wants_analysis)

    def handle(self, stimulus: Stimulus, ctx=None) -> Optional[CodePercept]:
        text = stimulus.text or ""
        gen = re.search(
            r"(?:write|create|generate|make)\s+(?:me\s+)?(?:a\s+)?(?:function|program|script)?"
            r"[^a-z]*?(?:that\s+)?([\w \-]+)", text, re.I)
        if gen and re.search(r"\b(write|create|generate|make)\b.*\b(function|program|code|script)\b",
                             text, re.I):
            code, name, explanation = self.generate(gen.group(1).strip(" .?!"))
            return CodePercept(
                description=f"I wrote a Python function '{name}': {explanation}",
                confidence=0.8, kind="generation", language="python",
                details={"code": code, "function": name, "explanation": explanation},
            )

        code_block = self._extract_block(text)
        lang = detect_language(code_block)
        if lang == "python":
            analysis = analyze_python(code_block)
        else:
            analysis = self._heuristic_analysis(code_block, lang)
        return CodePercept(
            description=f"Analyzed {lang} code: {analysis.summary()}",
            confidence=0.75 if lang != "unknown" else 0.4,
            kind="analysis", language=lang,
            details={"lines": analysis.lines,
                     "functions": [f["name"] for f in analysis.functions],
                     "classes": analysis.classes, "imports": analysis.imports,
                     "issues": analysis.issues, "quality": analysis.quality},
        )

    # ------------------------------------------------------------------ generation
    def generate(self, spec: str) -> Tuple[str, str, str]:
        """Match a spec like 'add two numbers' to a template."""
        spec_l = spec.lower()
        best, best_score = None, 0
        for keywords, name, code in _GENERATORS:
            score = sum(1 for k in keywords if k in spec_l)
            if score > best_score:
                best, best_score = (name, code), score
        if best is None:
            name = re.sub(r"\W+", "_", spec.strip())[:30] or "task"
            code = (f'def {name}(input):\n    """TODO: {spec}"""\n'
                    f'    raise NotImplementedError')
            return code, name, "a starting stub for your task"
        doc_parts = best[1].split('"""')
        explanation = doc_parts[1].strip().rstrip(".") if len(doc_parts) > 1 else "does the job"
        return best[1], best[0], explanation

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _extract_block(text: str) -> str:
        m = re.search(r"```(?:\w+)?\s*\n(.*?)```", text, re.S)
        if m:
            return m.group(1)
        return text

    @staticmethod
    def _looks_like_code(text: str) -> bool:
        markers = ["def ", "import ", "return ", "class ", "function", "const ",
                   "public ", "{", "}", ";", "=>", ": "]
        hits = sum(1 for m in markers if m in text)
        newlines = text.count("\n")
        return hits >= 3 and (newlines >= 2 or text.strip().startswith(("def", "class")))

    @staticmethod
    def _heuristic_analysis(code: str, lang: str) -> CodeAnalysis:
        result = CodeAnalysis(language=lang or "unknown", lines=len(code.splitlines()))
        result.functions = [{"name": m, "args": [], "docstring": False, "lines": 0}
                            for m in re.findall(r"(?:def|function)\s+(\w+)", code)]
        result.classes = re.findall(r"class\s+(\w+)", code)
        result.imports = re.findall(r"(?:import|using|#include)\s+([\w.<>]+)", code)
        for i, line in enumerate(code.splitlines(), 1):
            if len(line) > 100:
                result.issues.append(f"line {i} exceeds 100 characters")
        result.quality = max(0, 100 - 12 * len(result.issues))
        return result
