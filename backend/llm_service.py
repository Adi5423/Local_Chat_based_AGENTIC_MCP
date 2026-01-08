import httpx
import json
from typing import AsyncGenerator, List, Dict, Optional

from mcp.registry import MCPToolRegistry


class LLMClient:
    """
    Generic LLM client for llama.cpp OpenAI-compatible server.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "qwen2.5-coder.gguf",
        tool_registry: Optional[MCPToolRegistry] = None,
    ):
        self.base_url = base_url
        self.model = model
        self.tool_registry = tool_registry

        self.system_prompt = """You are an expert coding assistant.

You may have access to tools.

When a tool is required, respond ONLY with valid JSON in this exact format:

{
  "tool_call": {
    "name": "<tool_name>",
    "arguments": { "<arg>": "<value>" }
  }
}

Do NOT explain tool calls.
Do NOT use markdown inside tool calls.
IMPORTANT:
- Call a tool ONLY once per step
- AFTER a tool result is provided, you MUST respond in plain text
- DO NOT call the same tool again unless explicitly required
IMPORTANT OUTPUT RULE:
- Never include tool-call JSON inside explanations or markdown
- Tool calls must be the ONLY content of the response when used

"""

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion using llama.cpp OpenAI-compatible API.
        """

        # Build system prompt with tool descriptions
        system_prompt = self.system_prompt

        if self.tool_registry:
            tools = self.tool_registry.list_tools()
            if tools:
                system_prompt += "\n\nAVAILABLE TOOLS:\n"
                for tool in tools:
                    system_prompt += (
                        f"\nTool name: {tool['name']}\n"
                        f"Description: {tool['description']}\n"
                        f"Input schema: {tool['input_schema']}\n"
                    )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *messages,
            ],
            "stream": True,
            "temperature": 0.7,
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
            ) as response:

                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue

                    data = line.removeprefix("data:").strip()

                    if data == "[DONE]":
                        return

                    try:
                        event = json.loads(data)
                        delta = event["choices"][0]["delta"]
                        content = delta.get("content")
                        if content:
                            yield content
                    except Exception:
                        continue
