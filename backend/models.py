from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    message: str
    context_length: Optional[int] = Field(default=10, ge=1, le=50)


class ChatResponse(BaseModel):
    chat_id: str
    message: str
    role: str = "assistant"
    timestamp: datetime


class ChatHistoryResponse(BaseModel):
    chat_id: str
    messages: List[Message]


class ChatListItem(BaseModel):
    chat_id: str
    created_at: datetime
    last_message: Optional[str]
    message_count: int


class NewChatResponse(BaseModel):
    chat_id: str
    message: str = "New chat created"