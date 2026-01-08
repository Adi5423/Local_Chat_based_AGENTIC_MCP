from typing import Any, Dict

from .registry import MCPToolRegistry
from .types import ToolCall, ToolResult


class MCPExecutor:
    """
    Executes MCP tools in a controlled way.
    """

    def __init__(self, registry: MCPToolRegistry):
        self.registry = registry

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        tool_name = tool_call["name"]
        arguments = tool_call.get("arguments", {})

        try:
            tool = self.registry.get(tool_name)
        except KeyError:
            return {
                "name": tool_name,
                "result": f"Error: tool '{tool_name}' is not available."
            }
        
        result = await tool.run(arguments)

        return {
            "name": tool_name,
            "result": result,
        }
