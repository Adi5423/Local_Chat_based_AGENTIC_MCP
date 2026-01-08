from typing import AsyncGenerator
from database import Database
from llm_service import LLMClient
from models import Message
from mcp.executor import MCPExecutor
from mcp.parser import parse_tool_call


class ChatService:
    def __init__(self, database: Database, llm_service):
        self.db = database
        self.llm = llm_service
        self.mcp_executor = MCPExecutor(llm_service.tool_registry)
    
    async def create_new_chat(self) -> str:
        """Create a new chat session"""
        return await self.db.create_chat()
    
    async def get_context_messages(
        self, 
        chat_id: str, 
        context_length: int = 10
    ) -> list:
        """Retrieve recent messages for context"""
        messages = await self.db.get_chat_history(chat_id, limit=context_length)
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]
    
    async def process_message(
        self,
        chat_id: str,
        user_message: str,
        context_length: int = 10,
    ):
        """
        Agent loop:
        - Send message to LLM
        - Detect tool calls
        - Execute tools
        - Feed results back
        - Continue until final answer
        """

        if not await self.db.chat_exists(chat_id):
            raise ValueError(f"Chat {chat_id} does not exist")

        # Save user message
        await self.db.save_message(chat_id, "user", user_message)

        # Agent state
        max_steps = 5
        step = 0

        messages = await self.get_context_messages(chat_id, context_length)

        while step < max_steps:
            step += 1

            full_response = ""

            async for chunk in self.llm.stream_chat(messages):
                full_response += chunk

            # Try to parse tool call
            tool_call = parse_tool_call(full_response)

            if tool_call is None:
                # Final assistant response — now stream to user
                await self.db.save_message(chat_id, "assistant", full_response)
                yield full_response
                return

            # Tool call detected — do NOT expose to user
            # Prevent trivial or accidental tool calls
            # Allow echo ONLY if user explicitly asked for it
            if tool_call["name"] == "echo":
                if "use the echo tool" not in user_message.lower():
                    messages.append({
                        "role": "system",
                        "content": (
                            "Echo tool should only be used when the user explicitly requests it. "
                            "Respond to the user in natural language instead."
                        )
                    })
                    continue

            
            tool_result = await self.mcp_executor.execute(tool_call)

            # Inject tool result as system message
            messages.append({
                "role": "system",
                "content": (
                    f"Tool '{tool_result['name']}' returned:\n"
                    f"{tool_result['result']}"
                )
            })

        # Safety fallback
        fallback = "Error: agent exceeded maximum reasoning steps."
        await self.db.save_message(chat_id, "assistant", fallback)
        yield fallback

    
    async def get_chat_history(self, chat_id: str) -> list:
        """Get full chat history"""
        if not await self.db.chat_exists(chat_id):
            raise ValueError(f"Chat {chat_id} does not exist")
        
        return await self.db.get_chat_history(chat_id)
    
    async def get_all_chats(self) -> list:
        """Get all chat sessions"""
        return await self.db.get_all_chats()