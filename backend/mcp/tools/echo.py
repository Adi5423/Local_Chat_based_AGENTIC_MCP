from mcp.base import MCPTool


class EchoTool(MCPTool):
    name = "echo"
    description = "Echoes back the provided text."

    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"}
        },
        "required": ["text"]
    }

    output_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"}
        }
    }

    async def run(self, arguments):
        return {"text": arguments["text"]}
