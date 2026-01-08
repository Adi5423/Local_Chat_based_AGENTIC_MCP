from abc import ABC, abstractmethod
from typing import Any, Dict

from .types import ToolInputSchema, ToolOutputSchema


class MCPTool(ABC):
    """
    Base class for all MCP tools.

    This mirrors Claude MCP design:
    - Explicit capability declaration
    - Structured input/output
    - No hidden behavior
    """

    # ---- Required metadata ----
    name: str
    description: str

    # ---- Schemas ----
    input_schema: ToolInputSchema
    output_schema: ToolOutputSchema

    @abstractmethod
    async def run(self, arguments: Dict[str, Any]) -> Any:
        """
        Execute the tool.

        Args:
            arguments: Validated arguments matching input_schema

        Returns:
            Any JSON-serializable result matching output_schema
        """
        raise NotImplementedError
