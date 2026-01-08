import json
from typing import Optional

from .types import ToolCall


def parse_tool_call(text: str) -> Optional[ToolCall]:
    """
    Attempt to parse a tool call from model output.
    Returns None if not a valid tool call.
    """

    text = text.strip()

    if not text.startswith("{"):
        return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if "tool_call" not in data:
        return None

    tool_call = data["tool_call"]

    if not isinstance(tool_call, dict):
        return None

    if "name" not in tool_call or "arguments" not in tool_call:
        return None

    return tool_call
