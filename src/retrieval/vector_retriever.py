"""Vector similarity retriever using Zilliz Cloud (Milvus)."""
import logging
from typing import Dict, List

from src.embedding.embedder import OpenAIEmbedder

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Vector similarity search retriever."""
    
    def __init__(
        self,
        vector_store,
        embedder: OpenAIEmbedder
    ):
        """
        Initialize vector retriever.
        
        Args:
            vector_store: Vector store instance (ZillizVectorStore)
            embedder: OpenAI embedder for query encoding
        """
        self.vector_store = vector_store
        self.embedder = embedder
        
        logger.info("Initialized VectorRetriever with Zilliz Cloud")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve documents using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents with scores
        """
        # Embed the query
        query_embedding = self.embedder.embed_text(query)
        
        # Search in vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        logger.info(f"Vector search returned {len(results)} results for query: '{query[:50]}...'")
        return results
