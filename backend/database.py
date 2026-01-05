import aiosqlite
from typing import List, Optional
from datetime import datetime
import uuid


class Database:
    def __init__(self, db_path: str = "chat.db"):
        self.db_path = db_path
    
    async def init_db(self):
        """Initialize database schema"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_chat_id 
                ON messages(chat_id, timestamp)
            """)
            
            await db.commit()
    
    async def create_chat(self) -> str:
        """Create a new chat session"""
        chat_id = str(uuid.uuid4())
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO chats (chat_id) VALUES (?)",
                (chat_id,)
            )
            await db.commit()
        return chat_id
    
    async def chat_exists(self, chat_id: str) -> bool:
        """Check if chat exists"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT 1 FROM chats WHERE chat_id = ?",
                (chat_id,)
            )
            result = await cursor.fetchone()
            return result is not None
    
    async def save_message(
        self, 
        chat_id: str, 
        role: str, 
        content: str
    ) -> dict:
        """Save a message to the database"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO messages (chat_id, role, content) 
                VALUES (?, ?, ?)
                """,
                (chat_id, role, content)
            )
            await db.commit()
            
            # Fetch the inserted message with timestamp
            cursor = await db.execute(
                """
                SELECT role, content, timestamp 
                FROM messages 
                WHERE id = ?
                """,
                (cursor.lastrowid,)
            )
            row = await cursor.fetchone()
            
            return {
                "role": row[0],
                "content": row[1],
                "timestamp": row[2]
            }
    
    async def get_chat_history(
        self, 
        chat_id: str, 
        limit: Optional[int] = None
    ) -> List[dict]:
        """Retrieve chat history"""
        async with aiosqlite.connect(self.db_path) as db:
            if limit:
                query = """
                    SELECT role, content, timestamp 
                    FROM messages 
                    WHERE chat_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """
                cursor = await db.execute(query, (chat_id, limit))
            else:
                query = """
                    SELECT role, content, timestamp 
                    FROM messages 
                    WHERE chat_id = ? 
                    ORDER BY timestamp ASC
                """
                cursor = await db.execute(query, (chat_id,))
            
            rows = await cursor.fetchall()
            
            messages = [
                {
                    "role": row[0],
                    "content": row[1],
                    "timestamp": row[2]
                }
                for row in rows
            ]
            
            # If limited, reverse to get chronological order
            if limit:
                messages.reverse()
            
            return messages
    
    async def get_all_chats(self) -> List[dict]:
        """Get list of all chats with metadata"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT 
                    c.chat_id,
                    c.created_at,
                    (SELECT content FROM messages 
                     WHERE chat_id = c.chat_id 
                     ORDER BY timestamp DESC LIMIT 1) as last_message,
                    (SELECT COUNT(*) FROM messages 
                     WHERE chat_id = c.chat_id) as message_count
                FROM chats c
                ORDER BY c.created_at DESC
            """)
            
            rows = await cursor.fetchall()
            
            return [
                {
                    "chat_id": row[0],
                    "created_at": row[1],
                    "last_message": row[2],
                    "message_count": row[3]
                }
                for row in rows
            ]