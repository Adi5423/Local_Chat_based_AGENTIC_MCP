import httpx
from typing import AsyncGenerator, List, Dict
import json


class OllamaService:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "qwen2.5-coder:1.5b"
        self.system_prompt = """You are an expert coding assistant. You provide:
- Clear, well-commented code examples
- Explanations of programming concepts
- Debugging assistance
- Best practices and design patterns
- Code reviews and optimization suggestions

Always format code blocks with proper syntax highlighting.
Be concise but thorough in your explanations."""
    
    async def health_check(self) -> bool:
        """Check if Ollama is running"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/tags", timeout=5.0)
                return response.status_code == 200
        except Exception:
            return False
    
    async def stream_chat(
        self, 
        messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion from Ollama"""
        
        # Prepend system message
        full_messages = [
            {"role": "system", "content": self.system_prompt}
        ] + messages
        
        payload = {
            "model": self.model,
            "messages": full_messages,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": 4096
            }
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "message" in data:
                                content = data["message"].get("content", "")
                                if content:
                                    yield content
                            
                            # Check if done
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
    
    async def get_chat_response(
        self, 
        messages: List[Dict[str, str]]
    ) -> str:
        """Get complete chat response (non-streaming)"""
        full_response = ""
        async for chunk in self.stream_chat(messages):
            full_response += chunk
        return full_response