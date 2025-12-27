"""Vector store operations - using Zilliz Cloud (Milvus) exclusively.

This module provides the main interface for vector storage operations.
ChromaDB support has been removed in favor of Zilliz Cloud for better
scalability and cloud deployment support.
"""
import logging

logger = logging.getLogger(__name__)


def get_vector_store():
    """
    Get configured Zilliz Cloud vector store instance.
    
    Returns:
        ZillizVectorStore instance
        
    Raises:
        ValueError: If Zilliz credentials are not configured
    """
    from config.settings import settings
    from src.storage.zilliz_store import get_zilliz_store
    
    if not settings.zilliz_uri or not settings.zilliz_token:
        raise ValueError(
            "Zilliz Cloud credentials not configured. "
            "Please set ZILLIZ_URI and ZILLIZ_TOKEN environment variables."
        )
    
    logger.info(f"Using Zilliz Cloud vector store (collection: {settings.zilliz_collection_name})")
    
    return get_zilliz_store(
        uri=settings.zilliz_uri,
        token=settings.zilliz_token,
        collection_name=settings.zilliz_collection_name,
        dimension=settings.embedding.dimension
    )
