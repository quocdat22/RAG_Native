"""BM25 keyword-based retriever."""
import logging
from typing import Dict, List

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25 keyword search retriever."""
    
    def __init__(self):
        """Initialize BM25 retriever."""
        self.corpus = []  # List of documents (text)
        self.metadata = []  # List of metadata dicts
        self.bm25 = None
        
        logger.info("Initialized BM25Retriever")
    
    def index_documents(self, documents: List[Dict]):
        """
        Index documents for BM25 search.
        
        Args:
            documents: List of documents with 'text' and 'metadata' keys
        """
        self.corpus = []
        self.metadata = []
        
        for doc in documents:
            self.corpus.append(doc["text"])
            self.metadata.append(doc.get("metadata", {}))
        
        # Tokenize corpus (simple split by whitespace)
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info(f"Indexed {len(self.corpus)} documents for BM25 search")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve documents using BM25 keyword matching.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents with BM25 scores
        """
        if not self.bm25:
            logger.warning("BM25 index not built, returning empty results")
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                results.append({
                    "text": self.corpus[idx],
                    "metadata": self.metadata[idx],
                    "score": float(scores[idx])
                })
        
        logger.info(f"BM25 search returned {len(results)} results for query: '{query[:50]}...'")
        return results
