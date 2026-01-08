from typing import Dict, List

from .base import MCPTool


class MCPToolRegistry:
    """
    Central registry for all MCP tools.

    The agent and LLM never access tools directly —
    they only see what this registry exposes.
    """

    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}

    def register(self, tool: MCPTool) -> None:
        """
        Register a new tool.

        Raises:
            ValueError if tool name already exists
        """
        if tool.name in self._tools:
            raise ValueError(f"MCP tool '{tool.name}' already registered")

        self._tools[tool.name] = tool

    def get(self, name: str) -> MCPTool:
        """
        Retrieve a tool by name.

        Raises:
            KeyError if tool does not exist
        """
        if name not in self._tools:
            raise KeyError(f"MCP tool '{name}' not found")

        return self._tools[name]

    def list_tools(self) -> List[dict]:
        """
        Return tool metadata for LLM consumption.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
            }
            for tool in self._tools.values()
        ]
