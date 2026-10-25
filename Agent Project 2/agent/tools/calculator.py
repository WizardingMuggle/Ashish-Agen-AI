"""
================================================================================
 TOOLS: calculator + clock  —  things LLMs are notoriously bad at
================================================================================

An LLM predicts text; it does not compute. It will often get 37 * 74 right
and 3 * 190 + 2 * 40 + 3 * 65 wrong, with full confidence. It also has no idea
what today's date is (its training data stops at some point in the past).

Both problems are solved the same way: give the model a tool and tell it to
use it. This is the simplest and most convincing example of why tools matter.

SAFETY: never run model-written code with eval(). The calculator parses the
expression into a syntax tree and only allows numbers, + - * / // % ** and
a few whitelisted functions. Anything else is rejected.
"""

from __future__ import annotations

import ast
import math
import operator
from datetime import datetime

_BINARY_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "round": round, "abs": abs, "min": min, "max": max, "sum": lambda *a: sum(a),
    "sqrt": math.sqrt, "ceil": math.ceil, "floor": math.floor,
}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS and not node.keywords):
        return _FUNCTIONS[node.func.id](*[_evaluate(a) for a in node.args])
    raise ValueError(f"not allowed in a calculator expression: {ast.dump(node)[:60]}")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly. Use this for EVERY calculation instead
    of doing maths in your head. Supports + - * / // % ** and parentheses, plus the
    functions round, abs, min, max, sum, sqrt, ceil, floor.

    Args:
        expression: The expression to evaluate, e.g. "(50 * 2 - 34) * 18.5". Plain numbers only: no units, currency symbols or thousands separators.
    """
    tree = ast.parse(expression, mode="eval")
    value = _evaluate(tree)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, float):
        value = round(value, 6)
    return f"{expression} = {value}"


def get_current_datetime() -> str:
    """Return the current local date, time and weekday. Call this whenever the answer
    depends on today's date (ages, deadlines, 'how many months ago')."""
    now = datetime.now().astimezone()
    return (f"{now:%Y-%m-%d %H:%M} ({now:%A}), time zone {now.tzname()} "
            f"(UTC{now:%z}). ISO date: {now.date().isoformat()}")


def months_between(start_date: str, end_date: str) -> str:
    """Count the whole calendar months between two dates (end minus start). Useful for
    equipment age and refresh cycles.

    Args:
        start_date: The earlier date, format YYYY-MM-DD.
        end_date: The later date, format YYYY-MM-DD (use get_current_datetime for today).
    """
    a = datetime.strptime(start_date, "%Y-%m-%d").date()
    b = datetime.strptime(end_date, "%Y-%m-%d").date()
    months = (b.year - a.year) * 12 + (b.month - a.month) - (1 if b.day < a.day else 0)
    return f"{months} whole months between {start_date} and {end_date} ({(b - a).days} days)"
