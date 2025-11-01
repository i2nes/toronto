"""Conversation memory manager for Toronto AI Assistant.

Handles session creation, message persistence, and conversation history
for multi-turn chat interactions.
"""
import uuid
from typing import List, Dict, Any, Optional
import structlog

from app import db

logger = structlog.get_logger()


class ConversationManager:
    """Manages chat sessions and conversation history."""

    def __init__(self, context_window_size: int = 6):
        """Initialize the conversation manager.

        Args:
            context_window_size: Number of recent messages to include in context
        """
        self.context_window_size = context_window_size

    def create_session(self, title: Optional[str] = None) -> str:
        """Create a new chat session.

        Args:
            title: Optional title for the session

        Returns:
            The created session ID
        """
        session_id = str(uuid.uuid4())
        db.create_session(session_id, title)
        logger.info("conversation_session_created", session_id=session_id)
        return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Add a message to a session.

        Args:
            session_id: The session to add the message to
            role: Message role ('user' or 'assistant')
            content: The message content
            sources: Optional list of source chunks used for RAG

        Returns:
            ID of the inserted message
        """
        message_id = db.add_message(session_id, role, content, sources)
        logger.info(
            "conversation_message_added",
            session_id=session_id,
            role=role,
            message_id=message_id,
        )
        return message_id

    def get_recent_messages(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get recent messages for a session.

        Args:
            session_id: The session ID to get messages for
            limit: Maximum number of messages (defaults to context_window_size)

        Returns:
            List of message dictionaries in chronological order
        """
        limit = limit or self.context_window_size
        messages = db.get_recent_messages(session_id, limit)
        logger.info(
            "conversation_messages_retrieved",
            session_id=session_id,
            count=len(messages),
        )
        return messages

    def get_all_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a session.

        Args:
            session_id: The session ID to get messages for

        Returns:
            List of all message dictionaries in chronological order
        """
        messages = db.get_messages(session_id)
        logger.info(
            "conversation_all_messages_retrieved",
            session_id=session_id,
            count=len(messages),
        )
        return messages

    def format_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Format recent conversation history for LLM context.

        Args:
            session_id: The session ID to format history for

        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        messages = self.get_recent_messages(session_id)

        # Format for LLM (only role and content, no sources or metadata)
        history = []
        for msg in messages:
            history.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        logger.info(
            "conversation_history_formatted",
            session_id=session_id,
            message_count=len(history),
        )
        return history

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session details.

        Args:
            session_id: The session ID to retrieve

        Returns:
            Session dictionary or None if not found
        """
        return db.get_session(session_id)

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all sessions, most recent first.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session dictionaries
        """
        return db.list_sessions(limit)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages.

        Args:
            session_id: The session ID to delete

        Returns:
            True if deleted, False if not found
        """
        deleted = db.delete_session(session_id)
        if deleted:
            logger.info("conversation_session_deleted", session_id=session_id)
        return deleted

    def update_session_title(self, session_id: str, first_message: str) -> None:
        """Update session title based on first message.

        Creates a brief title from the first user message (max 50 chars).

        Args:
            session_id: The session to update
            first_message: The first user message
        """
        # Create a concise title from the first message
        title = first_message[:50]
        if len(first_message) > 50:
            title = title.rsplit(" ", 1)[0] + "..."

        # Update in database
        conn = db.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (title, session_id),
            )
            conn.commit()
            logger.info("session_title_updated", session_id=session_id, title=title)
        except Exception as e:
            conn.rollback()
            logger.error("session_title_update_failed", error=str(e))
        finally:
            conn.close()
