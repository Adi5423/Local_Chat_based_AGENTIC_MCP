import json
from typing import Optional
from .types import ToolCall


def parse_tool_call(text: str) -> Optional[ToolCall]:
    """
    Parse a tool call ONLY if the entire response is valid JSON.
    This prevents accidental tool calls embedded in explanations.
    """

    text = text.strip()

    # Hard rule: tool calls must be pure JSON, nothing else
    if not (text.startswith("{") and text.endswith("}")):
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
