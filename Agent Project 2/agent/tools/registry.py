"""
================================================================================
 TOOL REGISTRY  —  turn ordinary Python functions into tools an LLM can call
================================================================================

The model never sees your Python code. It only sees a SCHEMA for each tool:

    {
      "type": "function",
      "function": {
        "name": "check_inventory",
        "description": "Look up stock levels for a product ...",
        "parameters": {                         <- JSON Schema
          "type": "object",
          "properties": {"query": {"type": "string", "description": "SKU or name"}},
          "required": ["query"],
          "additionalProperties": false
        }
      }
    }

The name, the description and the parameter descriptions are effectively a
PROMPT: they are all the model knows about when and how to use the tool. A vague
description gives you an agent that picks the wrong tool. Try `/tool <name>`.

Writing schemas by hand is tedious, so the @tool decorator builds them from
the function's signature (type hints -> JSON types, defaults -> optional) and
its docstring (first paragraph -> description, "Args:" section -> parameters).
That is exactly what frameworks like LangChain, the OpenAI Agents SDK and
Pydantic AI do under the hood.

The registry also EXECUTES tool calls safely:
  * parses the JSON arguments the model wrote,
  * validates them against the schema (unknown / missing / wrong-type),
  * catches any exception,
and always returns a string. Errors are returned to the model as
"ERROR: ..." instead of crashing the program, so the model can read the error
and fix its call. That self-correction is a key agent behaviour.
"""

from __future__ import annotations

import inspect
import json
import re
import time
import types
import typing
from dataclasses import dataclass
from typing import Any, Callable, Literal

_JSON_TYPES: dict[Any, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict           # JSON Schema of the arguments
    func: Callable[..., Any]
    side_effect: bool          # True = changes something -> needs human approval
    category: str

    def schema(self) -> dict:
        """The exact structure sent to the OpenAI API in `tools=[...]`."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    ok: bool
    output: str                # what the model will read
    seconds: float
    truncated: bool = False


# ---------------------------------------------------------------------- #
# Building schemas from Python functions
# ---------------------------------------------------------------------- #
def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a docstring into (description, {arg_name: arg_description})."""
    doc = inspect.cleandoc(doc or "")
    main, _, args_part = doc.partition("Args:")
    description = " ".join(main.split())
    arg_docs: dict[str, str] = {}
    current = None
    for line in args_part.splitlines():
        match = re.match(r"\s*(\w+)\s*:\s*(.*)", line)
        if match:
            current = match.group(1)
            arg_docs[current] = match.group(2).strip()
        elif current and line.strip():
            arg_docs[current] += " " + line.strip()
    return description, arg_docs


def _json_type(annotation: Any) -> dict:
    """Python type hint -> JSON Schema fragment."""
    origin = typing.get_origin(annotation)
    if annotation in _JSON_TYPES:
        return {"type": _JSON_TYPES[annotation]}
    if origin is Literal:
        values = list(typing.get_args(annotation))
        return {"type": _JSON_TYPES[type(values[0])], "enum": values}
    if origin in (list, tuple):
        args = typing.get_args(annotation)
        return {"type": "array", "items": _json_type(args[0]) if args else {}}
    if origin in (typing.Union, types.UnionType):
        # Optional[X] / X | None: the parameter is optional, describe it as X.
        inner = [a for a in typing.get_args(annotation) if a is not type(None)]
        return _json_type(inner[0])
    raise TypeError(f"Unsupported tool parameter type: {annotation!r}")


def build_tool(func: Callable[..., Any], side_effect: bool, category: str) -> Tool:
    description, arg_docs = _parse_docstring(func.__doc__ or "")
    hints = typing.get_type_hints(func)
    properties: dict[str, dict] = {}
    required: list[str] = []
    for name, param in inspect.signature(func).parameters.items():
        prop = _json_type(hints[name])
        if name in arg_docs:
            prop["description"] = arg_docs[name]
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            prop["description"] = f"{prop.get('description', '')} (optional, default: {param.default})".strip()
        properties[name] = prop
    parameters = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    return Tool(func.__name__, description, parameters, func, side_effect, category)


# ---------------------------------------------------------------------- #
# The registry
# ---------------------------------------------------------------------- #
class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        self.tools[tool.name] = tool
        return tool

    def add(self, func: Callable[..., Any], side_effect: bool = False, category: str = "general") -> Tool:
        return self.register(build_tool(func, side_effect, category))

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def names(self) -> list[str]:
        return list(self.tools)

    def schemas(self, names: list[str]) -> list[dict]:
        return [self.tools[n].schema() for n in names if n in self.tools]

    # ------------------------------------------------------------------ #
    # Validation: the model can (and sometimes will) write bad arguments
    # ------------------------------------------------------------------ #
    @staticmethod
    def validate(tool: Tool, args: Any) -> str | None:
        """Return an error message, or None if `args` match the schema."""
        if not isinstance(args, dict):
            return "arguments must be a JSON object"
        props = tool.parameters["properties"]
        unknown = sorted(set(args) - set(props))
        if unknown:
            return f"unknown argument(s) {unknown}; allowed: {sorted(props)}"
        missing = [r for r in tool.parameters["required"] if r not in args]
        if missing:
            return f"missing required argument(s) {missing}"
        checks = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
        }
        for key, value in args.items():
            spec = props[key]
            if not checks[spec["type"]](value):
                return f"argument '{key}' must be of type {spec['type']}, got {type(value).__name__}"
            if "enum" in spec and value not in spec["enum"]:
                return f"argument '{key}' must be one of {spec['enum']}, got {value!r}"
        return None

    def parse_arguments(self, name: str, raw_arguments: str | dict) -> tuple[dict, str | None]:
        """Return (arguments, error). `error` is None when the call is well-formed."""
        tool = self.get(name)
        if tool is None:
            return {}, f"ERROR: there is no tool called '{name}'. Available tools: {self.names()}"
        try:
            args = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        except json.JSONDecodeError as exc:
            return {}, f"ERROR: the arguments are not valid JSON ({exc}). Send a JSON object."
        if args is None:
            args = {}
        problem = self.validate(tool, args)
        if problem:
            return {}, f"ERROR: invalid arguments for {name}: {problem}"
        return args, None

    # ------------------------------------------------------------------ #
    # Execution: always returns a ToolResult, never raises
    # ------------------------------------------------------------------ #
    def execute(self, name: str, raw_arguments: str | dict, max_chars: int) -> ToolResult:
        start = time.perf_counter()

        def done(ok: bool, output: str) -> ToolResult:
            truncated = len(output) > max_chars
            if truncated:
                output = output[:max_chars] + f"\n...[truncated: {len(output) - max_chars} more characters]"
            return ToolResult(ok, output, time.perf_counter() - start, truncated)

        args, problem = self.parse_arguments(name, raw_arguments)
        if problem:
            return done(False, problem)
        tool = self.tools[name]
        try:
            result = tool.func(**args)
        except Exception as exc:  # the tool itself failed -> tell the model why
            return done(False, f"ERROR: {type(exc).__name__}: {exc}")
        if not isinstance(result, str):
            result = json.dumps(result, indent=1, ensure_ascii=False, default=str)
        return done(True, result)
