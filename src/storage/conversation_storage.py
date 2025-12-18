"""SQLite-based conversation storage for managing chat sessions and history."""
import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Message(BaseModel):
    """Message model for conversation history."""
    id: str
    conversation_id: str
    role: str  # "user" or "assistant"
    content: str
    sources: Optional[List[Dict]] = None
    created_at: datetime


class Conversation(BaseModel):
    """Conversation session model."""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[Message] = []


class ConversationStorage:
    """SQLite storage for conversations and messages."""
    
    def __init__(self, db_path: Path):
        """
        Initialize conversation storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"ConversationStorage initialized with db: {db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize database tables."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Create conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            
            # Create index for faster message queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation 
                ON messages(conversation_id)
            """)
            
            conn.commit()
            logger.info("Database tables initialized")
        finally:
            conn.close()
    
    def create_conversation(self, title: Optional[str] = None) -> Conversation:
        """
        Create a new conversation.
        
        Args:
            title: Optional title for the conversation
            
        Returns:
            Created Conversation object
        """
        conn = self._get_connection()
        try:
            conversation_id = str(uuid.uuid4())
            now = datetime.utcnow()
            title = title or "New Conversation"
            
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation_id, title, now, now)
            )
            conn.commit()
            
            logger.info(f"Created conversation: {conversation_id}")
            return Conversation(
                id=conversation_id,
                title=title,
                created_at=now,
                updated_at=now,
                messages=[]
            )
        finally:
            conn.close()
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Get a conversation with its messages.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Conversation object or None if not found
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Get conversation
            cursor.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Get messages
            cursor.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                (conversation_id,)
            )
            message_rows = cursor.fetchall()
            
            messages = [
                Message(
                    id=msg["id"],
                    conversation_id=msg["conversation_id"],
                    role=msg["role"],
                    content=msg["content"],
                    sources=json.loads(msg["sources"]) if msg["sources"] else None,
                    created_at=datetime.fromisoformat(msg["created_at"]) if isinstance(msg["created_at"], str) else msg["created_at"]
                )
                for msg in message_rows
            ]
            
            return Conversation(
                id=row["id"],
                title=row["title"],
                created_at=datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"],
                updated_at=datetime.fromisoformat(row["updated_at"]) if isinstance(row["updated_at"], str) else row["updated_at"],
                messages=messages
            )
        finally:
            conn.close()
    
    def list_conversations(self, limit: int = 50) -> List[Conversation]:
        """
        List all conversations (without messages for performance).
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            List of Conversation objects (without messages)
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            
            return [
                Conversation(
                    id=row["id"],
                    title=row["title"],
                    created_at=datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"],
                    updated_at=datetime.fromisoformat(row["updated_at"]) if isinstance(row["updated_at"], str) else row["updated_at"],
                    messages=[]
                )
                for row in rows
            ]
        finally:
            conn.close()

    def count_conversations(self) -> int:
        """
        Count total number of conversations.
        
        Returns:
            Total count
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM conversations")
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """
        Update conversation title.
        
        Args:
            conversation_id: ID of the conversation
            title: New title
            
        Returns:
            True if updated successfully
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, datetime.utcnow(), conversation_id)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and its messages.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            True if deleted successfully
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            # Delete messages first (due to foreign key)
            cursor.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,)
            )
            # Delete conversation
            cursor.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            conn.commit()
            logger.info(f"Deleted conversation: {conversation_id}")
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict]] = None
    ) -> Message:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: ID of the conversation
            role: Message role ("user" or "assistant")
            content: Message content
            sources: Optional list of source citations
            
        Returns:
            Created Message object
        """
        conn = self._get_connection()
        try:
            message_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO messages (id, conversation_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (message_id, conversation_id, role, content, json.dumps(sources) if sources else None, now)
            )
            
            # Update conversation's updated_at
            cursor.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id)
            )
            
            conn.commit()
            
            logger.debug(f"Added message to conversation {conversation_id}")
            return Message(
                id=message_id,
                conversation_id=conversation_id,
                role=role,
                content=content,
                sources=sources,
                created_at=now
            )
        finally:
            conn.close()
    
    def get_recent_messages(
        self,
        conversation_id: str,
        limit: int = 10
    ) -> List[Message]:
        """
        Get recent messages from a conversation (for context window).
        
        Args:
            conversation_id: ID of the conversation
            limit: Maximum number of messages to return
            
        Returns:
            List of recent Message objects (oldest first)
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            # Get the most recent N messages, then reverse to get chronological order
            cursor.execute(
                """
                SELECT * FROM (
                    SELECT * FROM messages 
                    WHERE conversation_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                ) ORDER BY created_at ASC
                """,
                (conversation_id, limit * 2)  # *2 to get pairs of user/assistant messages
            )
            rows = cursor.fetchall()
            
            return [
                Message(
                    id=row["id"],
                    conversation_id=row["conversation_id"],
                    role=row["role"],
                    content=row["content"],
                    sources=json.loads(row["sources"]) if row["sources"] else None,
                    created_at=datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"]
                )
                for row in rows
            ]
        finally:
            conn.close()


# Global storage instance
_conversation_storage: Optional[ConversationStorage] = None


def get_conversation_storage() -> ConversationStorage:
    """Get or create global conversation storage instance."""
    global _conversation_storage
    if _conversation_storage is None:
        from config.settings import settings
        db_path = settings.chroma_dir.parent / "conversations.db"
        _conversation_storage = ConversationStorage(db_path)
    return _conversation_storage
