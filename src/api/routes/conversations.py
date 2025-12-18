"""Conversation management routes."""
import logging

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    ConversationCreate,
    ConversationListResponse,
    ConversationResponse,
    ConversationUpdate,
    MessageSchema,
)
from src.storage.conversation_storage import get_conversation_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationResponse)
async def create_conversation(request: ConversationCreate = None):
    """
    Create a new conversation.
    
    Args:
        request: Optional title for the conversation
        
    Returns:
        Created conversation
    """
    try:
        storage = get_conversation_storage()
        title = request.title if request else None
        conversation = storage.create_conversation(title=title)
        
        return ConversationResponse(
            id=conversation.id,
            title=conversation.title,
            messages=[],
            created_at=conversation.created_at,
            updated_at=conversation.updated_at
        )
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=ConversationListResponse)
async def list_conversations(limit: int = 50):
    """
    List all conversations.
    
    Args:
        limit: Maximum number of conversations to return
        
    Returns:
        List of conversations (without messages)
    """
    try:
        storage = get_conversation_storage()
        conversations = storage.list_conversations(limit=limit)
        
        return ConversationListResponse(
            conversations=[
                ConversationResponse(
                    id=conv.id,
                    title=conv.title,
                    messages=[],
                    created_at=conv.created_at,
                    updated_at=conv.updated_at
                )
                for conv in conversations
            ],
            total=len(conversations)
        )
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str):
    """
    Get a conversation with all its messages.
    
    Args:
        conversation_id: ID of the conversation
        
    Returns:
        Conversation with messages
    """
    try:
        storage = get_conversation_storage()
        conversation = storage.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return ConversationResponse(
            id=conversation.id,
            title=conversation.title,
            messages=[
                MessageSchema(
                    id=msg.id,
                    role=msg.role,
                    content=msg.content,
                    sources=msg.sources,
                    created_at=msg.created_at
                )
                for msg in conversation.messages
            ],
            created_at=conversation.created_at,
            updated_at=conversation.updated_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(conversation_id: str, request: ConversationUpdate):
    """
    Update a conversation's title.
    
    Args:
        conversation_id: ID of the conversation
        request: New title
        
    Returns:
        Updated conversation
    """
    try:
        storage = get_conversation_storage()
        
        if not storage.update_conversation_title(conversation_id, request.title):
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        conversation = storage.get_conversation(conversation_id)
        
        return ConversationResponse(
            id=conversation.id,
            title=conversation.title,
            messages=[
                MessageSchema(
                    id=msg.id,
                    role=msg.role,
                    content=msg.content,
                    sources=msg.sources,
                    created_at=msg.created_at
                )
                for msg in conversation.messages
            ],
            created_at=conversation.created_at,
            updated_at=conversation.updated_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation and all its messages.
    
    Args:
        conversation_id: ID of the conversation
        
    Returns:
        Success message
    """
    try:
        storage = get_conversation_storage()
        
        if not storage.delete_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {"status": "success", "message": "Conversation deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
