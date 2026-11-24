"""Built-in mini-Jinja engine for chat templates.

A dependency-free Jinja subset that renders the templates llama.cpp's
``minja`` accepts — the same subset real model templates (ChatML, Qwen2.5,
Qwen3, Llama-3.1, Zephyr) actually use. It backs ``Mind.apply_chat_template``
so a mind can format conversations with any llama.cpp-style template, not
just its own exported one.

Supported (mirrors minja / llama.cpp ``--jinja``):

* ``{{ expr }}`` output, ``{% if/elif/else/endif %}``, ``{% for x in seq %}``
  (with ``loop.index/index0/first/last/length/revindex/revindex0`` and
  for-``else``), ``{% set x = expr %}`` (plain, tuple and ``ns.attr``
  targets), ``{% generation %}`` blocks (llama.cpp extension; body renders),
  ``{# comments #}``, and ``-`` whitespace control on every tag.
* Literals, arithmetic (``+ - * // / %``), string concat (``+`` and ``~``),
  comparisons, ``in`` / ``not in``, ``and/or/not``, ternary
  (``a if b else c``), attribute/subscript access, slices (``messages[1:]``,
  ``messages[::-1]``), negative indices, calls with keyword arguments.
* Filters: ``tojson`` (sorted keys like llama.cpp's nlohmann dump, optional
  ``indent``), ``trim``, ``length``/``count``, ``first``, ``last``,
  ``join``, ``items``, ``reject``/``select`` (with tests), ``default``,
  ``upper``, ``lower``, ``capitalize``, ``replace``, ``reverse``,
  ``string``, ``list``, ``int``, ``float``.
* Tests (``is``): ``defined``, ``undefined``, ``none``, ``string``,
  ``number``, ``integer``, ``float``, ``boolean``, ``true``, ``false``,
  ``mapping``, ``iterable``, ``sequence``, ``equalto``/``eq``, ``ne``.
* Global functions: ``raise_exception``, ``strftime_now``, ``namespace``,
  ``range``. String methods ``startswith/endswith/split/rstrip/lstrip/
  strip/replace/upper/lower``; dict methods ``get/keys/values/items``.

Jinja scoping rule: assignments inside a ``for`` loop stay in that
iteration's scope (that is why templates use ``namespace()`` to accumulate
state across iterations); ``if`` bodies do not create a scope. Missing
variables are silently undefined: they render as ``""``, test false for
``is defined``, and truth-test false — exactly like Jinja/minja.

Not supported (clear errors, like minja without the callback hooks):
macros, includes, extends/blocks, ``{% raw %}``, the ``map``/``selectattr``
filters, and auto-escaping.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["render", "TemplateError", "JUndefined", "UNDEFINED",
           "JNamespace", "cognify_chat_template", "generate_chat_template",
           "TEMPLATE_GENERATOR"]


class TemplateError(ValueError):
    """A template could not be parsed or rendered."""


class JUndefined:
    """Jinja's undefined value: falsy, renders empty, silently propagates.

    Accessing an attribute or item of it returns it again, so
    ``message.reasoning_content is string`` works without a key check."""
    __slots__ = ()

    def __bool__(self) -> bool:
        return False

    def __str__(self) -> str:
        return ""

    def __repr__(self) -> str:
        return "undefined"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, JUndefined)

    def __ne__(self, other: Any) -> bool:
        return not isinstance(other, JUndefined)

    def __hash__(self) -> int:
        return hash("undefined")


UNDEFINED = JUndefined()


class JNamespace:
    """The ``namespace(...)`` template function result: a mutable attribute
    bag, the Jinja-sanctioned way to accumulate state across loop
    iterations."""
    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in sorted(self.__dict__.items()))
        return f"namespace({inner})"


# ---------------------------------------------------------------------------
# Lexer: template text -> tokens (kind, payload, left_strip, right_strip)
# ---------------------------------------------------------------------------

_OPENERS = {"{{": "var", "{%": "stmt", "{#": "comment"}
_CLOSERS = {"var": "}}", "stmt": "%}", "comment": "#}"}


def _lex(source: str) -> List[Tuple[str, str, bool, bool]]:
    tokens: List[Tuple[str, str, bool, bool]] = []
    i, n = 0, len(source)
    while i < n:
        # Next tag opener (first of the three that occurs from here).
        hit: Optional[Tuple[int, str, str]] = None
        for opener, kind in _OPENERS.items():
            k = source.find(opener, i)
            if k != -1 and (hit is None or k < hit[0]):
                hit = (k, kind, opener)
        if hit is None:
            tokens.append(("text", source[i:], False, False))
            break
        k, kind, opener = hit
        if k > i:
            tokens.append(("text", source[i:k], False, False))
        left = k + 2 < n and source[k + 2] == "-"
        start = k + 3 if left else k + 2
        closer = _CLOSERS[kind]
        q, payload_end, right = start, None, False
        quote: Optional[str] = None
        while q < n:
            c = source[q]
            if kind != "comment" and quote:  # inside a string literal: tag
                if c == "\\":               # closers never count; comments
                    q += 2                  # are plain text (apostrophes!)
                    continue
                if c == quote:
                    quote = None
                q += 1
                continue
            if kind != "comment" and c in "'\"":
                quote = c
                q += 1
                continue
            if source.startswith(closer, q):
                if q > start and source[q - 1] == "-":
                    right, payload_end = True, q - 1
                else:
                    payload_end = q
                break
            q += 1
        if payload_end is None:
            raise TemplateError(f"unterminated {kind} tag starting at offset {k}")
        payload = source[start:payload_end].strip()
        tokens.append((kind, payload, left, right))
        i = q + len(closer)
    return tokens


# ---------------------------------------------------------------------------
# Expression lexer/parser (recursive descent, Jinja precedence)
# ---------------------------------------------------------------------------

class _Tok:
    __slots__ = ("kind", "value", "pos")

    def __init__(self, kind: str, value: Any, pos: int) -> None:
        self.kind, self.value, self.pos = kind, value, pos

    def __repr__(self) -> str:
        return f"<{self.kind} {self.value!r}>"


_EXPR_TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<num>\d+\.\d+|\.\d+|\d+)
  | (?P<str>'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*")
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op>==|!=|<=|>=|//|[-+*/%~|\[\](){}.,:<>=])
""", re.VERBOSE)

_STR_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f",
                "'": "'", '"': '"', "\\": "\\"}


def _unescape(body: str, quote: str) -> str:
    out, i = [], 0
    while i < len(body):
        c = body[i]
        if c == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt == "u" and i + 5 < len(body):
                out.append(chr(int(body[i + 2:i + 6], 16)))
                i += 6
                continue
            out.append(_STR_ESCAPES.get(nxt, nxt))
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _lex_expr(src: str) -> List[_Tok]:
    toks: List[_Tok] = []
    pos = 0
    while pos < len(src):
        m = _EXPR_TOKEN.match(src, pos)
        if m is None:
            raise TemplateError(f"bad character {src[pos]!r} in expression: {src!r}")
        pos = m.end()
        if m.lastgroup == "ws":
            continue
        if m.lastgroup == "num":
            text = m.group("num")
            toks.append(_Tok("num", float(text) if "." in text else int(text), pos))
        elif m.lastgroup == "str":
            body = m.group("str")[1:-1]
            toks.append(_Tok("str", _unescape(body, m.group("str")[0]), pos))
        elif m.lastgroup == "name":
            toks.append(_Tok("name", m.group("name"), pos))
        else:
            toks.append(_Tok("op", m.group("op"), pos))
    toks.append(_Tok("end", None, pos))
    return toks


# AST nodes: plain tuples ("kind", ...) keep the engine small.
#   ("lit", value)  ("name", name)  ("attr", node, name)  ("item", node, key)
#   ("slice", node, start, stop, step)  ("call", node, args, kwargs)
#   ("filter", node, name, args, kwargs)  ("not", node)  ("neg", node)
#   ("pos", node)  ("binop", op, left, right)  ("concat", parts)
#   ("and", l, r)  ("or", l, r)  ("ternary", cond, then, else)
#   ("test", node, name, negated, args)  ("list", items)  ("dict", pairs)


class _ExprParser:
    def __init__(self, toks: List[_Tok], source: str) -> None:
        self.toks = toks
        self.i = 0
        self.source = source

    # -- token helpers ----------------------------------------------------
    def _peek(self) -> _Tok:
        return self.toks[self.i]

    def _next(self) -> _Tok:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def _accept_op(self, *ops: str) -> Optional[str]:
        tok = self._peek()
        if tok.kind == "op" and tok.value in ops:
            self.i += 1
            return tok.value
        return None

    def _accept_name(self, *names: str) -> Optional[str]:
        tok = self._peek()
        if tok.kind == "name" and tok.value in names:
            self.i += 1
            return tok.value
        return None

    def _expect_op(self, op: str) -> None:
        if self._accept_op(op) is None:
            raise TemplateError(
                f"expected {op!r} in expression: {self.source!r}")

    # -- grammar ------------------------------------------------------------
    def parse(self) -> Any:
        node = self._ternary()
        if self._peek().kind == "op" and self._peek().value == ",":
            items = [node]
            while self._accept_op(","):
                if self._peek().kind == "end":
                    break
                items.append(self._ternary())
            node = ("tuple", items)
        if self._peek().kind != "end":
            raise TemplateError(f"unexpected {self._peek().value!r} in expression: {self.source!r}")
        return node

    def _ternary(self) -> Any:
        expr = self._or()
        if self._accept_name("if"):
            cond = self._or()
            self._expect_name_else()
            return ("ternary", cond, expr, self._ternary())
        return expr

    def _expect_name_else(self) -> None:
        if self._accept_name("else") is None:
            raise TemplateError(f"expected 'else' in expression: {self.source!r}")

    def _test_arg_follows(self) -> bool:
        # `is equalto 5` / `is divisibleby 3` — a bare test argument follows
        # unless the next token is a reserved word or another test name.
        tok = self._peek()
        if tok.kind in ("num", "str"):
            return True
        if tok.kind == "op" and tok.value in ("[", "{"):
            return True
        if tok.kind == "name":
            return tok.value not in _RESERVED_AFTER_TEST
        return False

    def _or(self) -> Any:
        node = self._and()
        while self._accept_name("or"):
            node = ("or", node, self._and())
        return node

    def _and(self) -> Any:
        node = self._not()
        while self._accept_name("and"):
            node = ("and", node, self._not())
        return node

    def _not(self) -> Any:
        if self._accept_name("not"):
            return ("not", self._not())
        return self._compare()

    def _compare(self) -> Any:
        node = self._concat()
        while True:
            for op in ("==", "!=", "<=", ">=", "<", ">"):
                if self._accept_op(op):
                    node = ("binop", op, node, self._concat())
                    break
            else:
                if self._accept_name("in"):
                    node = ("binop", "in", node, self._concat())
                elif self._peek().kind == "name" and self._peek().value == "not" \
                        and self.i + 1 < len(self.toks) \
                        and self.toks[self.i + 1].kind == "name" \
                        and self.toks[self.i + 1].value == "in":
                    self.i += 2
                    node = ("not", ("binop", "in", node, self._concat()))
                elif self._accept_name("is"):
                    negated = self._accept_name("not") is not None
                    tok = self._next()
                    if tok.kind != "name":
                        raise TemplateError(f"expected test name after 'is': {self.source!r}")
                    args: List[Any] = []
                    if self._accept_op("("):
                        args = self._call_args(")")
                    elif self._test_arg_follows():
                        args = [self._unary()]
                    node = ("test", node, tok.value, negated, args)
                else:
                    return node

    def _concat(self) -> Any:
        parts = [self._math1()]
        while self._accept_op("~"):
            parts.append(self._math1())
        return parts[0] if len(parts) == 1 else ("concat", parts)

    def _math1(self) -> Any:
        node = self._math2()
        while True:
            op = self._accept_op("+", "-")
            if op is None:
                return node
            node = ("binop", op, node, self._math2())

    def _math2(self) -> Any:
        node = self._unary()
        while True:
            op = self._accept_op("*", "/", "//", "%")
            if op is None:
                return node
            node = ("binop", op, node, self._unary())

    def _unary(self) -> Any:
        if self._accept_op("-"):
            return ("neg", self._unary())
        if self._accept_op("+"):
            return ("pos", self._unary())
        return self._postfix()

    def _postfix(self) -> Any:
        node = self._atom()
        while True:
            if self._accept_op("."):
                tok = self._next()
                if tok.kind != "name":
                    raise TemplateError(f"expected name after '.': {self.source!r}")
                if self._peek().kind == "op" and self._peek().value == "(":
                    node = ("call", ("attr", node, tok.value), *self._call_with_parens())
                else:
                    node = ("attr", node, tok.value)
            elif self._accept_op("["):
                node = self._subscript(node)
            elif self._accept_op("|"):
                node = self._filter(node)
            elif self._peek().kind == "op" and self._peek().value == "(":
                node = ("call", node, *self._call_with_parens())
            else:
                return node

    def _call_with_parens(self) -> Tuple[List[Any], List[Tuple[str, Any]]]:
        self._expect_op("(")
        return self._call_args(")")

    def _call_args(self, closer: str) -> Tuple[List[Any], List[Tuple[str, Any]]]:
        args: List[Any] = []
        kwargs: List[Tuple[str, Any]] = []
        if self._accept_op(closer):
            return args, kwargs
        while True:
            if self._peek().kind == "name" \
                    and self.i + 1 < len(self.toks) \
                    and self.toks[self.i + 1].kind == "op" \
                    and self.toks[self.i + 1].value == "=":
                name = self._next().value
                self.i += 1                      # '='
                kwargs.append((name, self._ternary()))
            else:
                args.append(self._ternary())
            if self._accept_op(","):
                if self._peek().kind == "op" and self._peek().value == closer:
                    break                      # trailing comma, like Jinja
                continue
            self._expect_op(closer)
            return args, kwargs
        self._expect_op(closer)
        return args, kwargs

    def _subscript(self, node: Any) -> Any:
        # '[' already consumed: [expr] | [start:] | [start:stop] |
        # [start:stop:step] | [:stop] | [:stop:step] | [::step] | [::]
        start: Any = None
        stop: Any = None
        step: Any = None
        if self._accept_op(":"):
            # empty start
            if self._accept_op(":"):
                # [::...]
                if self._peek().kind != "op" or self._peek().value != "]":
                    step = self._ternary()
            else:
                if self._peek().kind != "op" or self._peek().value != "]":
                    stop = self._ternary()
                if self._accept_op(":"):
                    if self._peek().kind != "op" or self._peek().value != "]":
                        step = self._ternary()
            self._expect_op("]")
            return ("slice", node, start, stop, step)
        start = self._ternary()
        if self._accept_op("]"):
            return ("item", node, start)
        self._expect_op(":")
        if self._accept_op(":"):
            # [start::...]
            if self._peek().kind != "op" or self._peek().value != "]":
                step = self._ternary()
        else:
            if self._peek().kind != "op" or self._peek().value != "]":
                stop = self._ternary()
            if self._accept_op(":"):
                if self._peek().kind != "op" or self._peek().value != "]":
                    step = self._ternary()
        self._expect_op("]")
        return ("slice", node, start, stop, step)

    def _filter(self, node: Any) -> Any:
        tok = self._next()
        if tok.kind != "name":
            raise TemplateError(f"expected filter name after '|': {self.source!r}")
        args: List[Any] = []
        kwargs: List[Tuple[str, Any]] = []
        if self._peek().kind == "op" and self._peek().value == "(":
            args, kwargs = self._call_with_parens()
        return ("filter", node, tok.value, args, kwargs)

    def _atom(self) -> Any:
        tok = self._peek()
        if tok.kind == "num" or tok.kind == "str":
            self.i += 1
            return ("lit", tok.value)
        if tok.kind == "name":
            self.i += 1
            if tok.value == "true":
                return ("lit", True)
            if tok.value == "false":
                return ("lit", False)
            if tok.value == "none":
                return ("lit", None)
            return ("name", tok.value)
        op = self._accept_op("[", "{", "(")
        if op == "[":
            items: List[Any] = []
            if not self._accept_op("]"):
                while True:
                    items.append(self._ternary())
                    if self._accept_op(","):
                        if self._accept_op("]"):
                            break
                        continue
                    self._expect_op("]")
                    break
            return ("list", items)
        if op == "{":
            pairs: List[Tuple[Any, Any]] = []
            if not self._accept_op("}"):
                while True:
                    key = self._ternary()
                    self._expect_op(":")
                    pairs.append((key, self._ternary()))
                    if self._accept_op(","):
                        if self._accept_op("}"):
                            break
                        continue
                    self._expect_op("}")
                    break
            return ("dict", pairs)
        if op == "(":
            inner = self._ternary()
            self._expect_op(")")
            return inner
        raise TemplateError(f"unexpected {tok.value!r} in expression: {self.source!r}")


def _parse_expr(src: str) -> Any:
    return _ExprParser(_lex_expr(src), src).parse()


# ---------------------------------------------------------------------------
# Template parser: token stream -> node tree
# ---------------------------------------------------------------------------
#   ("text", s)  ("out", expr)  ("if", [(cond|None, body), ...])
#   ("for", targets, iter, body, else_body)  ("set", targets, expr, attr)
#   ("generation", body)


class _TemplateParser:
    """Token stream -> node tree. Strip flags follow Jinja: a tag's ``-``
    left flag right-trims the preceding text, its ``-`` right flag
    left-trims the next text token (cleared by any intervening tag)."""

    def __init__(self, tokens: List[Tuple[str, str, bool, bool]]) -> None:
        self.tokens = tokens
        self.i = 0
        self._pending_lstrip = False

    def parse(self) -> List[Any]:
        body, stop, _ = self._body(())
        if stop is not None:
            raise TemplateError(f"unexpected {{% {stop} %}}")
        return body

    def _body(self, stops: Tuple[str, ...]) -> Tuple[List[Any], Optional[str], Optional[str]]:
        body: List[Any] = []
        while self.i < len(self.tokens):
            kind, payload, left, right = self.tokens[self.i]
            self.i += 1
            if kind == "text":
                self._emit_text(body, payload)
                continue
            self._pending_lstrip = False          # an intervening tag absorbs
            if left and body and body[-1][0] == "text":
                body[-1] = ("text", body[-1][1].rstrip())
            if kind == "comment":
                self._pending_lstrip = right
                continue
            if kind == "var":
                body.append(("out", _parse_expr(payload)))
                self._pending_lstrip = right
                continue
            # stmt
            word, _, rest = payload.partition(" ")
            rest = rest.strip()
            if word in stops:
                self._pending_lstrip = right
                return body, word, rest
            # This tag's right-strip applies to the first text INSIDE the
            # block it opens — set before dispatching nested bodies.
            self._pending_lstrip = right
            if word == "if":
                body.append(self._if(_parse_expr(rest)))
            elif word == "for":
                body.append(self._for(rest))
            elif word == "set":
                body.append(self._set(rest))
            elif word == "generation":
                inner, stop, _ = self._body(("endgeneration",))
                if stop != "endgeneration":
                    raise TemplateError("{% generation %} without {% endgeneration %}")
                body.append(("generation", inner))
            elif word in ("elif", "else", "endif", "endfor", "endgeneration"):
                raise TemplateError(f"unexpected {{% {word} %}}")
            else:
                raise TemplateError(
                    f"unsupported statement {word!r} (built-in engine supports "
                    f"if/for/set/generation, not macros or includes)")
        return body, None, None

    def _emit_text(self, body: List[Any], payload: str) -> None:
        if self._pending_lstrip:
            payload = payload.lstrip()
            self._pending_lstrip = False
        if payload:
            body.append(("text", payload))

    def _if(self, cond: Any) -> Any:
        branches: List[Tuple[Any, List[Any]]] = []
        body, stop, rest = self._body(("elif", "else", "endif"))
        branches.append((cond, body))
        while stop == "elif":
            cond2 = _parse_expr(rest) if rest else None
            body2, stop, rest = self._body(("elif", "else", "endif"))
            branches.append((cond2, body2))
        if stop == "else":
            else_body, stop, _ = self._body(("endif",))
            branches.append((None, else_body))
        if stop != "endif":
            raise TemplateError("{% if %} without {% endif %}")
        return ("if", branches)

    def _for(self, rest: str) -> Any:
        # targets 'in' expression — targets: name | (a, b) | a, b
        parser = _ExprParser(_lex_expr(rest), rest)
        targets: List[str] = []
        parenthesized = parser._accept_op("(")
        while True:
            tok = parser._next()
            if tok.kind != "name":
                raise TemplateError(f"bad loop target in {rest!r}")
            targets.append(tok.value)
            if not parser._accept_op(","):
                break
        if parenthesized and not parser._accept_op(")"):
            raise TemplateError(f"bad loop target in {rest!r}")
        if parser._accept_name("in") is None:
            raise TemplateError(f"missing 'in' in {{% for %}}: {rest!r}")
        iter_expr = parser._ternary()
        if parser._peek().kind != "end":
            raise TemplateError(f"trailing tokens in {{% for %}}: {rest!r}")
        body, stop, _ = self._body(("else", "endfor"))
        else_body: List[Any] = []
        if stop == "else":
            else_body, stop, _ = self._body(("endfor",))
        if stop != "endfor":
            raise TemplateError("{% for %} without {% endfor %}")
        return ("for", targets, iter_expr, body, else_body)

    def _set(self, rest: str) -> Any:
        # target = expression — target: name | ns.attr | a, b | (a, b)
        parser = _ExprParser(_lex_expr(rest), rest)
        targets: List[str] = []
        parenthesized = parser._accept_op("(")
        while True:
            tok = parser._next()
            if tok.kind != "name":
                raise TemplateError(f"bad assignment target in {{% set %}}: {rest!r}")
            targets.append(tok.value)
            if not parser._accept_op(","):
                break
        if parenthesized and not parser._accept_op(")"):
            raise TemplateError(f"bad assignment target in {{% set %}}: {rest!r}")
        if len(targets) == 1 and not parenthesized:
            target: Any = ("name", targets[0])
            while parser._accept_op("."):
                tok = parser._next()
                if tok.kind != "name":
                    raise TemplateError("bad attribute assignment target")
                target = ("attr", target, tok.value)
            if parser._accept_op("("):
                raise TemplateError("bad assignment target (call not allowed)")
        else:
            target = ("tuple", [("name", n) for n in targets])
        if not parser._accept_op("="):
            raise TemplateError(f"missing '=' in {{% set %}}: {rest!r}")
        value = parser._ternary()
        if parser._peek().kind != "end":
            raise TemplateError(f"trailing tokens in {{% set %}}: {rest!r}")
        return ("set", target, value)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

_TESTS = {"defined", "undefined", "none", "string", "number", "integer",
          "float", "boolean", "true", "false", "mapping", "iterable",
          "sequence", "equalto", "eq", "ne", "in", "divisibleby"}

# After `is <test>`, these names never start a bare test argument.
_RESERVED_AFTER_TEST = _TESTS | {"and", "or", "not", "if", "else", "is",
                                 "true", "false", "none"}


def _truthy(value: Any) -> bool:
    if isinstance(value, JUndefined):
        return False
    return bool(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _str(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, JUndefined):
        return ""
    if value is True:
        return "True"
    if value is False:
        return "False"
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, JNamespace):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_repr(v) for v in value) + "]"
    if isinstance(value, tuple):
        inner = ", ".join(_repr(v) for v in value)
        return "(" + inner + ("," if len(value) == 1 else "") + ")"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_repr(k)}: {_repr(v)}" for k, v in value.items()) + "}"
    return str(value)


def _repr(value: Any) -> str:
    if isinstance(value, str):
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    return _str(value)


def _equals(a: Any, b: Any) -> bool:
    if isinstance(a, JUndefined) or isinstance(b, JUndefined):
        return isinstance(a, JUndefined) and isinstance(b, JUndefined)
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if _is_number(a) and _is_number(b):
        return a == b
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equals(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a.keys()) == set(b.keys()) and all(_equals(a[k], b[k]) for k in a)
    return a is b


def _compare_values(op: str, a: Any, b: Any) -> bool:
    if op == "==":
        return _equals(a, b)
    if op == "!=":
        return not _equals(a, b)
    if isinstance(a, JUndefined) or isinstance(b, JUndefined) \
            or a is None or b is None:
        raise TemplateError(f"cannot order-compare {op} with None/undefined")
    if (_is_number(a) and _is_number(b)) or (isinstance(a, str) and isinstance(b, str)):
        if op == "<":
            return a < b
        if op == ">":
            return a > b
        if op == "<=":
            return a <= b
        return a >= b
    raise TemplateError(f"cannot order-compare {type(a).__name__} {op} {type(b).__name__}")


def _contains(item: Any, container: Any) -> bool:
    if isinstance(container, str):
        if not isinstance(item, str):
            raise TemplateError("'in' on strings needs a string operand")
        return item in container
    if isinstance(container, list):
        return any(_equals(item, v) for v in container)
    if isinstance(container, dict):
        return any(_equals(item, k) for k in container.keys())
    raise TemplateError(f"'in' not supported on {type(container).__name__}")


def _length(value: Any) -> int:
    if isinstance(value, (str, list, dict)):
        return len(value)
    raise TemplateError(f"object of type {type(value).__name__} has no length")


def _index(seq: Any, key: Any) -> Any:
    if isinstance(seq, dict):
        for k in seq.keys():
            if _equals(k, key):
                return seq[k]
        return UNDEFINED
    if isinstance(seq, (list, str)):
        if not _is_number(key):
            raise TemplateError(f"{type(seq).__name__} indices must be integers")
        i = int(key)
        if -len(seq) <= i < len(seq):
            return seq[i]
        raise TemplateError(f"{type(seq).__name__} index {i} out of range (length {len(seq)})")
    if isinstance(seq, JUndefined):
        return UNDEFINED
    raise TemplateError(f"{type(seq).__name__} is not subscriptable")


def _slice_of(seq: Any, start: Any, stop: Any, step: Any) -> Any:
    if not isinstance(seq, (list, str)):
        raise TemplateError(f"cannot slice {type(seq).__name__}")
    s = slice(int(start) if start is not None else None,
              int(stop) if stop is not None else None,
              int(step) if step is not None else None)
    return seq[s]


def _attr(value: Any, name: str) -> Any:
    if isinstance(value, JNamespace):
        return getattr(value, name, UNDEFINED)
    if isinstance(value, dict):
        for k in value.keys():
            if _equals(k, name):
                return value[k]
        if name in ("get", "keys", "values", "items"):
            return _BoundMethod(value, name)
        return UNDEFINED
    if isinstance(value, JUndefined):
        return UNDEFINED
    if isinstance(value, str) and name in ("upper", "lower", "strip",
                                           "lstrip", "rstrip", "split",
                                           "startswith", "endswith",
                                           "replace", "capitalize", "join"):
        return _BoundMethod(value, name)
    if isinstance(value, list) and name in ("index", "length"):
        return _BoundMethod(value, name) if name == "index" else len(value)
    raise TemplateError(f"{type(value).__name__!s} has no attribute {name!r}")


class _BoundMethod:
    """A string/list/dict method awaiting a call (``m.content.split(x)``)."""
    __slots__ = ("target", "name")

    def __init__(self, target: Any, name: str) -> None:
        self.target, self.name = target, name


def _call_bound(bound: _BoundMethod, args: List[Any], kwargs: List[Tuple[str, Any]]) -> Any:
    target, name = bound.target, bound.name
    if kwargs:
        raise TemplateError(f"method {name!r} takes no keyword arguments")
    if isinstance(target, str):
        if name == "startswith":
            return target.startswith(_expect_str(args, name))
        if name == "endswith":
            return target.endswith(_expect_str(args, name))
        if name == "split":
            if not args:
                return target.split()
            return target.split(_expect_str(args, name))
        if name == "rstrip":
            return target.rstrip(_expect_str(args) if args else None)
        if name == "lstrip":
            return target.lstrip(_expect_str(args) if args else None)
        if name == "strip":
            return target.strip(_expect_str(args) if args else None)
        if name == "replace":
            _arity(args, 2, name)
            return target.replace(args[0], args[1])
        if name == "join":
            if not args or not isinstance(args[0], list):
                raise TemplateError("str.join() needs a list argument")
            return target.join(_str(v) for v in args[0])
        if name == "upper":
            return target.upper()
        if name == "lower":
            return target.lower()
        if name == "capitalize":
            return target.capitalize()
    if isinstance(target, dict):
        if name == "get":
            _arity(args, 1, name, 2)
            found = _index(target, args[0])
            if found is UNDEFINED:
                return args[1] if len(args) > 1 else None
            return found
        if name == "keys":
            return list(target.keys())
        if name == "values":
            return list(target.values())
        if name == "items":
            return [[k, v] for k, v in target.items()]
    if isinstance(target, list) and name == "index":
        _arity(args, 1, name)
        for i, v in enumerate(target):
            if _equals(v, args[0]):
                return i
        raise TemplateError("value not in list")
    raise TemplateError(f"unsupported method {name!r}")


def _expect_str(args: List[Any], name: str = "") -> str:
    if not args or not isinstance(args[0], str):
        raise TemplateError(f"method {name or 'str'!r} needs a string argument")
    return args[0]


def _arity(args: List[Any], exact: int, name: str, max_allowed: Optional[int] = None) -> None:
    ok = len(args) == exact or (max_allowed is not None and exact <= len(args) <= max_allowed)
    if not ok:
        raise TemplateError(f"method {name!r} takes {exact} argument(s)")


# -- filters ---------------------------------------------------------------

def _to_json(value: Any, indent: Optional[int] = None) -> str:
    def plain(v: Any) -> Any:
        if isinstance(v, JNamespace):
            return {k: plain(x) for k, x in sorted(v.__dict__.items())}
        if isinstance(v, JUndefined):
            return None
        if isinstance(v, list):
            return [plain(x) for x in v]
        if isinstance(v, tuple):
            return [plain(x) for x in v]
        if isinstance(v, dict):
            return {str(k): plain(x) for k, x in v.items()}
        return v
    if indent is not None:
        return json.dumps(plain(value), ensure_ascii=False, sort_keys=True,
                          indent=indent)
    return json.dumps(plain(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _apply_filter(name: str, value: Any, args: List[Any], kwargs: List[Tuple[str, Any]]) -> Any:
    kw = dict(kwargs)
    if name == "tojson":
        return _to_json(value, kw.get("indent"))
    if name == "trim":
        _no_args(args, kwargs, name)
        return _expect_value(value, name).strip()
    if name in ("length", "count"):
        _no_args(args, kwargs, name)
        return _length(value)
    if name == "first":
        _no_args(args, kwargs, name)
        return _index(_expect_seq(value, name), 0)
    if name == "last":
        _no_args(args, kwargs, name)
        return _index(_expect_seq(value, name), -1)
    if name == "join":
        if not isinstance(value, list):
            raise TemplateError(f"filter 'join' needs a list, got {type(value).__name__}")
        sep = args[0] if args else kw.get("d", "")
        attr = kw.get("attribute")
        parts = [_str(_attr(v, attr)) if attr is not None else _str(v)
                 for v in value]
        return sep.join(parts)
    if name == "items":
        _no_args(args, kwargs, name)
        if not isinstance(value, dict):
            raise TemplateError(f"filter 'items' needs a dict, got {type(value).__name__}")
        return [[k, v] for k, v in sorted(value.items(), key=lambda kv: _str(kv[0]))]
    if name == "reject":
        return _select_or_reject(value, args, kwargs, keep=False)
    if name == "select":
        return _select_or_reject(value, args, kwargs, keep=True)
    if name == "default":
        dv = args[0] if args else kw.get("d", "")
        boolean = kw.get("boolean", args[1] if len(args) > 1 else False)
        if value is None or isinstance(value, JUndefined) \
                or (boolean and not _truthy(value)):
            return dv
        return value
    if name == "upper":
        _no_args(args, kwargs, name)
        return _expect_value(value, name).upper()
    if name == "lower":
        _no_args(args, kwargs, name)
        return _expect_value(value, name).lower()
    if name == "capitalize":
        _no_args(args, kwargs, name)
        return _expect_value(value, name).capitalize()
    if name == "replace":
        _arity(args, 2, name)
        return _expect_value(value, name).replace(args[0], args[1])
    if name == "reverse":
        _no_args(args, kwargs, name)
        if isinstance(value, str):
            return value[::-1]
        if isinstance(value, list):
            return list(reversed(value))
        raise TemplateError(f"filter 'reverse' needs a list or string")
    if name == "string":
        _no_args(args, kwargs, name)
        return _str(value)
    if name == "list":
        _no_args(args, kwargs, name)
        if isinstance(value, str):
            return list(value)
        if isinstance(value, list):
            return list(value)
        if isinstance(value, dict):
            return list(value.keys())
        raise TemplateError(f"filter 'list' cannot convert {type(value).__name__}")
    if name == "int":
        return _to_number(value, int, args, name)
    if name == "float":
        return _to_number(value, float, args, name)
    if name == "sort":
        _no_args(args, kwargs, name)
        if not isinstance(value, list):
            raise TemplateError("filter 'sort' needs a list")
        return sorted(value, key=_sort_key)
    raise TemplateError(
        f"unsupported filter {name!r} (built-in engine supports: tojson, "
        f"trim, length, count, first, last, join, items, reject, select, "
        f"default, upper, lower, capitalize, replace, reverse, string, "
        f"list, int, float, sort)")


def _sort_key(v: Any) -> Any:
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else _str(v)


def _expect_value(value: Any, name: str) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    raise TemplateError(f"filter {name!r} needs a string, got {type(value).__name__}")


def _expect_seq(value: Any, name: str) -> Any:
    if isinstance(value, (list, str)):
        return value
    raise TemplateError(f"filter {name!r} needs a list or string")


def _no_args(args: List[Any], kwargs: List[Tuple[str, Any]], name: str) -> None:
    if args or kwargs:
        raise TemplateError(f"filter {name!r} takes no arguments")


def _to_number(value: Any, cast: type, args: List[Any], name: str) -> Any:
    fallback = args[0] if args else (0 if cast is int else 0.0)
    try:
        if isinstance(value, str):
            return cast(value.strip())
        if _is_number(value):
            return cast(value)
        if value is None or isinstance(value, JUndefined):
            return fallback
        if isinstance(value, bool):
            return cast(value)
    except ValueError:
        return fallback
    raise TemplateError(f"filter {name!r} cannot convert {type(value).__name__}")


def _select_or_reject(value: Any, args: List[Any], kwargs: List[Tuple[str, Any]], keep: bool) -> Any:
    if not isinstance(value, list):
        raise TemplateError("reject/select need a list")
    if not args or not isinstance(args[0], str):
        raise TemplateError("reject/select need a test name")
    test_name, test_args = args[0], list(args[1:])
    out = []
    for v in value:
        if _run_test(test_name, v, test_args) == keep:
            out.append(v)
    return out


def _run_test(name: str, value: Any, args: List[Any]) -> bool:
    if name not in _TESTS:
        raise TemplateError(f"unsupported test {name!r}")
    if name == "defined":
        return not isinstance(value, JUndefined)
    if name == "undefined":
        return isinstance(value, JUndefined)
    if name == "none":
        return value is None
    if name == "string":
        return isinstance(value, str)
    if name == "number":
        return _is_number(value)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "float":
        return isinstance(value, float)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "true":
        return value is True
    if name == "false":
        return value is False
    if name == "mapping":
        return isinstance(value, dict)
    if name == "iterable" or name == "sequence":
        return isinstance(value, (list, str, dict))
    if name in ("equalto", "eq"):
        _arity(args, 1, name)
        return _equals(value, args[0])
    if name == "ne":
        _arity(args, 1, name)
        return not _equals(value, args[0])
    if name == "in":
        _arity(args, 1, name)
        return _contains(value, args[0])
    if name == "divisibleby":
        _arity(args, 1, name)
        return _is_number(value) and _is_number(args[0]) and int(value) % int(args[0]) == 0
    return False


# -- global template functions ----------------------------------------------

def _raise_exception(args: List[Any]) -> Any:
    _arity(args, 1, "raise_exception")
    raise TemplateError(str(args[0]))


def _strftime_now(args: List[Any]) -> Any:
    _arity(args, 1, "strftime_now")
    return time.strftime(args[0])


def _range(args: List[Any]) -> Any:
    if not args or len(args) > 3:
        raise TemplateError("range() takes 1-3 arguments")
    return list(range(*[int(a) for a in args]))


_BUILTINS: Dict[str, Any] = {
    "raise_exception": _raise_exception,
    "strftime_now": _strftime_now,
    "namespace": lambda args, kwargs: JNamespace(
        **{k: v for k, v in kwargs}),
    "range": _range,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class _Renderer:
    def __init__(self, template: str) -> None:
        self.template = template
        self.out: List[str] = []

    def render(self, variables: Dict[str, Any]) -> str:
        tokens = _lex(self.template)
        tree = _TemplateParser(tokens).parse()
        scopes: List[Dict[str, Any]] = [dict(variables)]
        self._exec(tree, scopes)
        return "".join(self.out)

    # -- scope helpers ------------------------------------------------------
    def _lookup(self, scopes: List[Dict[str, Any]], name: str) -> Any:
        for scope in reversed(scopes):
            if name in scope:
                return scope[name]
        if name in _BUILTINS:
            return name
        return UNDEFINED

    def _assign(self, scopes: List[Dict[str, Any]], name: str, value: Any) -> None:
        scopes[-1][name] = value

    # -- statements ----------------------------------------------------------
    def _exec(self, body: List[Any], scopes: List[Dict[str, Any]]) -> None:
        for node in body:
            kind = node[0]
            if kind == "text":
                self.out.append(node[1])
            elif kind == "out":
                self.out.append(_str(self._eval(node[1], scopes)))
            elif kind == "if":
                self._exec_if(node, scopes)
            elif kind == "for":
                self._exec_for(node, scopes)
            elif kind == "set":
                self._exec_set(node, scopes)
            elif kind == "generation":
                self._exec(node[1], scopes)
            else:
                raise TemplateError(f"unknown node {kind}")

    def _exec_if(self, node: Any, scopes: List[Dict[str, Any]]) -> None:
        for cond, body in node[1]:
            if cond is None or _truthy(self._eval(cond, scopes)):
                self._exec(body, scopes)
                return

    def _exec_for(self, node: Any, scopes: List[Dict[str, Any]]) -> None:
        _, targets, iter_expr, body, else_body = node
        seq = self._eval(iter_expr, scopes)
        if seq is None or isinstance(seq, JUndefined):
            raise TemplateError("cannot iterate over None/undefined")
        if isinstance(seq, dict):
            items = list(seq.keys())          # Jinja iterates dict keys
        elif isinstance(seq, list):
            items = list(seq)
        elif isinstance(seq, str):
            items = list(seq)
        else:
            raise TemplateError(f"cannot iterate over {type(seq).__name__}")
        if not items:
            self._exec(else_body, scopes)
            return
        length = len(items)
        for index, item in enumerate(items):
            loop_scope: Dict[str, Any] = {
                "loop": {
                    "index": index + 1, "index0": index,
                    "first": index == 0, "last": index == length - 1,
                    "length": length,
                    "revindex": length - index,
                    "revindex0": length - index - 1,
                },
            }
            if len(targets) == 1:
                loop_scope[targets[0]] = item
            else:
                unpacked = item if isinstance(item, list) else [item]
                if len(unpacked) != len(targets):
                    raise TemplateError(
                        f"cannot unpack loop item into {len(targets)} targets")
                for name, value in zip(targets, unpacked):
                    loop_scope[name] = value
            scopes.append(loop_scope)
            self._exec(body, scopes)
            scopes.pop()

    def _exec_set(self, node: Any, scopes: List[Dict[str, Any]]) -> None:
        _, target, value_expr = node
        value = self._eval(value_expr, scopes)
        if target[0] == "tuple":
            values = value if isinstance(value, list) else [value]
            if len(values) != len(target[1]):
                raise TemplateError("tuple assignment length mismatch")
            for name_node, v in zip(target[1], values):
                self._assign(scopes, name_node[1], v)
            return
        if target[0] == "attr":
            base = self._eval(target[1], scopes)
            if isinstance(base, JNamespace):
                setattr(base, target[2], value)
                return
            if isinstance(base, dict):
                base[target[2]] = value
                return
            raise TemplateError(
                "{% set %} on attributes needs a namespace() or dict target")
        self._assign(scopes, target[1], value)

    # -- expressions ----------------------------------------------------------
    def _eval(self, node: Any, scopes: List[Dict[str, Any]]) -> Any:
        kind = node[0]
        if kind == "lit":
            return node[1]
        if kind == "name":
            return self._lookup(scopes, node[1])
        if kind == "attr":
            return _attr(self._eval(node[1], scopes), node[2])
        if kind == "item":
            return _index(self._eval(node[1], scopes),
                          self._eval(node[2], scopes))
        if kind == "slice":
            return _slice_of(self._eval(node[1], scopes),
                             self._eval(node[2], scopes) if node[2] else None,
                             self._eval(node[3], scopes) if node[3] else None,
                             self._eval(node[4], scopes) if node[4] else None)
        if kind == "call":
            callee = self._eval(node[1], scopes)
            args = [self._eval(a, scopes) for a in node[2]]
            kwargs = [(k, self._eval(v, scopes)) for k, v in node[3]]
            if isinstance(callee, _BoundMethod):
                return _call_bound(callee, args, kwargs)
            if isinstance(callee, str):
                if callee in _BUILTINS:
                    if callee == "namespace":
                        return _BUILTINS[callee](args, kwargs)
                    return _BUILTINS[callee](args)
            raise TemplateError(f"not callable: {callee!r}")
        if kind == "filter":
            value = self._eval(node[1], scopes)
            args = [self._eval(a, scopes) for a in node[3]]
            kwargs = [(k, self._eval(v, scopes)) for k, v in node[4]]
            return _apply_filter(node[2], value, args, kwargs)
        if kind == "not":
            return not _truthy(self._eval(node[1], scopes))
        if kind == "neg":
            v = self._eval(node[1], scopes)
            if not _is_number(v):
                raise TemplateError("unary '-' needs a number")
            return -v
        if kind == "pos":
            v = self._eval(node[1], scopes)
            if not _is_number(v):
                raise TemplateError("unary '+' needs a number")
            return +v
        if kind == "and":
            left = self._eval(node[1], scopes)
            return left if not _truthy(left) else self._eval(node[2], scopes)
        if kind == "or":
            left = self._eval(node[1], scopes)
            return left if _truthy(left) else self._eval(node[2], scopes)
        if kind == "ternary":
            return self._eval(node[2], scopes) if _truthy(self._eval(node[1], scopes)) \
                else self._eval(node[3], scopes)
        if kind == "concat":
            return "".join(_str(self._eval(p, scopes)) for p in node[1])
        if kind == "binop":
            return self._eval_binop(node, scopes)
        if kind == "test":
            value = self._eval(node[1], scopes)
            args = [self._eval(a, scopes) for a in node[4]]
            result = _run_test(node[2], value, args)
            return not result if node[3] else result
        if kind == "list":
            return [self._eval(item, scopes) for item in node[1]]
        if kind == "tuple":
            return tuple(self._eval(item, scopes) for item in node[1])
        if kind == "dict":
            return {self._eval(k, scopes): self._eval(v, scopes)
                    for k, v in node[1]}
        raise TemplateError(f"unknown expression node {kind}")

    def _eval_binop(self, node: Any, scopes: List[Dict[str, Any]]) -> Any:
        _, op, left_node, right_node = node
        left = self._eval(left_node, scopes)
        right = self._eval(right_node, scopes)
        if op == "==":
            return _equals(left, right)
        if op == "!=":
            return not _equals(left, right)
        if op == "in":
            return _contains(left, right)
        if op in ("<", ">", "<=", ">="):
            return _compare_values(op, left, right)
        if op == "+":
            if _is_number(left) and _is_number(right):
                return left + right
            if isinstance(left, str) and isinstance(right, str):
                return left + right
            if isinstance(left, list) and isinstance(right, list):
                return left + right
            raise TemplateError(
                f"cannot add {type(left).__name__} and {type(right).__name__}")
        if op == "-":
            if _is_number(left) and _is_number(right):
                return left - right
            raise TemplateError(
                f"cannot subtract {type(left).__name__} from {type(right).__name__}")
        if op == "*":
            if _is_number(left) and _is_number(right):
                return left * right
            if isinstance(left, str) and _is_number(right):
                return left * int(right)
            raise TemplateError(
                f"cannot multiply {type(left).__name__} and {type(right).__name__}")
        if op == "/":
            if _is_number(left) and _is_number(right):
                if right == 0:
                    raise TemplateError("division by zero")
                return left / right
            raise TemplateError("division needs numbers")
        if op == "//":
            if _is_number(left) and _is_number(right):
                if right == 0:
                    raise TemplateError("division by zero")
                return left // right
            raise TemplateError("division needs numbers")
        if op == "%":
            if _is_number(left) and _is_number(right):
                if right == 0:
                    raise TemplateError("modulo by zero")
                return left % right
            raise TemplateError("modulo needs numbers")
        raise TemplateError(f"unknown operator {op}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render(source: str, variables: Optional[Dict[str, Any]] = None, **extra: Any) -> str:
    """Render ``source`` with the built-in engine.

    ``variables`` (or keyword arguments) supply the template context —
    ``messages``, ``tools``, ``add_generation_prompt``, ``bos_token``, …
    Missing names behave like Jinja's undefined (render ``""``).
    """
    merged: Dict[str, Any] = dict(variables or {})
    merged.update(extra)
    return _Renderer(source).render(merged)


def cognify_chat_template(manifest_json: Optional[str] = None,
                          thinking_enabled: bool = False,
                          tools_block: bool = True) -> str:
    """Cognify's own llama.cpp chat template — the format core the generator
    composes minds' templates from. The shape the GGUF exporter used to embed
    by hand: an optional capability-manifest ``set`` (renders nothing;
    extractable from llama-server's GET /props), an optional OpenAI-style
    tools block, ``role: content`` lines, and the assistant cue — which
    honors Qwen3's ``enable_thinking`` kwarg when the mind thinks.
    ``manifest_json=None`` (the runtime default) omits the manifest line;
    ``tools_block=False`` drops the tools branch (minds without the tools
    capability get a template that cannot even express tools).
    """
    parts: List[str] = []
    if manifest_json is not None:
        # The newline after the set is source formatting only — emit it only
        # when a left-stripping tag follows; with no tools block the bare
        # newline would otherwise leak into the render.
        parts.append("{%- set cognify_manifest = '" + manifest_json + "' %}"
                     + ("\n" if tools_block else ""))
    if tools_block:
        parts.append("{%- if tools %}tools: {{ tools | tojson }}\n"
                     "{% endif %}")
    parts.append("{% for m in messages %}{{ m['role'] }}: "
                 "{{ m['content'] }}\n{% endfor %}")
    if thinking_enabled:
        parts.append("{%- if enable_thinking is defined and not enable_thinking %}"
                     "assistant:{% else %}assistant: thought:{% endif %}")
    else:
        parts.append("assistant:")
    return "".join(parts)


TEMPLATE_GENERATOR = "cognify-builtin-jinja/1"


def _sanitize_comment(text: Any) -> str:
    """Jinja comments end at the first ``#}`` — keep template text and
    newlines out of the generated provenance comment."""
    cleaned = " ".join(str(text).split()).replace("#}", "# }")
    return cleaned or "-"


def generate_chat_template(mind: Any, manifest_json: Optional[str] = None) -> str:
    """Synthesize a chat template from a mind's own state.

    The mind IS the model, so the template it ships is generated from what it
    actually carries — not a hand-written constant: a provenance comment
    naming the mind, its thinking switch/level/ceiling, capabilities, tools
    and description (renders nothing, extractable from GET /props next to the
    manifest), then the format core: the optional manifest ``set``, a tools
    branch ONLY when the mind has the tools capability, ``role: content``
    lines, and a thinking-aware assistant cue. Deterministic: the same mind
    state always produces the same template bytes. The built-in engine
    renders the result, llama.cpp's minja accepts it (comment/set/if/for are
    all in the subset), and the exporter self-verifies the render before
    writing it into the GGUF.
    """
    from .options import ThinkingCeiling, ThinkingLevel  # leaf module, safe here

    thinking = bool(mind.thinking_options.enabled)
    capabilities = sorted(c.name for c in mind.capabilities)
    tool_names = sorted(mind.tools.names()) if mind.tools else []
    has_tools = "tool_use" in capabilities or bool(tool_names)
    description = getattr(mind, "description", "") or ""
    comment = (
        "{#- cognify generated chat template: "
        f"mind={_sanitize_comment(mind.name)}"
        f" | thinking={'on' if thinking else 'off'}/"
        f"{ThinkingLevel.parse(mind.thinking_options.level).value}/"
        f"{ThinkingCeiling.parse(mind.thinking_options.ceiling).value}"
        f" | capabilities={','.join(capabilities) if capabilities else 'none'}"
        f" | tools={','.join(tool_names) if tool_names else 'none'}"
        f" | description={_sanitize_comment(description)}"
        " -#}")  # -#} strips the source newline: renders clean for any core
    core = cognify_chat_template(manifest_json, thinking,
                                 tools_block=has_tools)
    return comment + "\n" + core
