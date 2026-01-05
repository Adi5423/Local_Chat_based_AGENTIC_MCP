from typing import AsyncGenerator
from database import Database
from llm_service import OllamaService
from models import Message


class ChatService:
    def __init__(self, database: Database, llm_service: OllamaService):
        self.db = database
        self.llm = llm_service
    
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
        context_length: int = 10
    ) -> AsyncGenerator[str, None]:
        """Process user message and stream LLM response"""
        
        # Check if chat exists
        if not await self.db.chat_exists(chat_id):
            raise ValueError(f"Chat {chat_id} does not exist")
        
        # Save user message
        await self.db.save_message(chat_id, "user", user_message)
        
        # Get conversation context
        context = await self.get_context_messages(chat_id, context_length)
        
        # Add current message to context (already saved, so included in history)
        # Stream LLM response
        full_response = ""
        async for chunk in self.llm.stream_chat(context):
            full_response += chunk
            yield chunk
        
        # Save assistant response
        await self.db.save_message(chat_id, "assistant", full_response)
    
    async def get_chat_history(self, chat_id: str) -> list:
        """Get full chat history"""
        if not await self.db.chat_exists(chat_id):
            raise ValueError(f"Chat {chat_id} does not exist")
        
        return await self.db.get_chat_history(chat_id)
    
    async def get_all_chats(self) -> list:
        """Get all chat sessions"""
        return await self.db.get_all_chats()