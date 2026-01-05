from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from sse_starlette.sse import EventSourceResponse
import json

from database import Database
from llm_service import OllamaService
from chat_service import ChatService
from models import (
    ChatRequest, 
    ChatResponse, 
    ChatHistoryResponse,
    ChatListItem,
    NewChatResponse,
    Message
)


# Global instances
db = Database()
llm = OllamaService()
chat_service = ChatService(db, llm)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    await db.init_db()
    
    # Check Ollama connection
    if not await llm.health_check():
        print("WARNING: Ollama is not running. Please start Ollama and ensure qwen2.5-coder is installed.")
        print("Run: ollama pull qwen2.5-coder")
    else:
        print("✓ Connected to Ollama")
    
    yield
    
    # Shutdown (cleanup if needed)
    pass


app = FastAPI(
    title="Local LLM Chat API",
    description="Fully local ChatGPT-like application with Qwen2.5-Coder",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    ollama_status = await llm.health_check()
    return {
        "status": "running",
        "ollama_connected": ollama_status,
        "model": llm.model
    }


@app.post("/chat/new", response_model=NewChatResponse)
async def create_new_chat():
    """Create a new chat session"""
    chat_id = await chat_service.create_new_chat()
    return NewChatResponse(chat_id=chat_id)


@app.post("/chat/{chat_id}/message")
async def send_message(
    chat_id: str = Path(..., description="Chat session ID"),
    request: ChatRequest = ...
):
    """Send a message and stream the response"""
    try:
        async def event_generator():
            async for chunk in chat_service.process_message(
                chat_id, 
                request.message,
                request.context_length
            ):
                yield {
                    "event": "message",
                    "data": json.dumps({"content": chunk})
                }
            
            # Send done event
            yield {
                "event": "done",
                "data": json.dumps({"status": "complete"})
            }
        
        return EventSourceResponse(event_generator())
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@app.get("/chat/{chat_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    chat_id: str = Path(..., description="Chat session ID")
):
    """Retrieve chat history"""
    try:
        messages = await chat_service.get_chat_history(chat_id)
        return ChatHistoryResponse(
            chat_id=chat_id,
            messages=[
                Message(role=msg["role"], content=msg["content"])
                for msg in messages
            ]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/chats", response_model=list[ChatListItem])
async def list_chats():
    """Get all chat sessions"""
    chats = await chat_service.get_all_chats()
    return [
        ChatListItem(
            chat_id=chat["chat_id"],
            created_at=chat["created_at"],
            last_message=chat["last_message"],
            message_count=chat["message_count"]
        )
        for chat in chats
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)