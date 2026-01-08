from typing import Any, Dict, TypedDict


class ToolInputSchema(TypedDict):
    """JSON-schema-like definition for tool inputs"""
    type: str
    properties: Dict[str, Any]
    required: list[str]


class ToolOutputSchema(TypedDict):
    """JSON-schema-like definition for tool outputs"""
    type: str
    properties: Dict[str, Any]


class ToolCall(TypedDict):
    """LLM → Tool call request"""
    name: str
    arguments: Dict[str, Any]


class ToolResult(TypedDict):
    """Tool → Agent result"""
    name: str
    result: Any
