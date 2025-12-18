"""Hybrid retriever combining vector and BM25 search."""
import logging
from typing import Dict, List

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever using Reciprocal Rank Fusion (RRF)."""
    
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        k: int = 60  # RRF parameter
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_retriever: Vector similarity retriever
            bm25_retriever: BM25 keyword retriever
            vector_weight: Weight for vector search results
            bm25_weight: Weight for BM25 results
            k: RRF constant (default 60)
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.k = k
        
        logger.info(
            f"Initialized HybridRetriever with "
            f"vector_weight={vector_weight}, bm25_weight={bm25_weight}"
        )
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict]
    ) -> List[Dict]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF formula: score = sum(1 / (k + rank))
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            
        Returns:
            Fused and ranked results
        """
        # Create lookup by text for deduplication
        scores = {}
        texts = {}
        metadata = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            text = result["text"]
            rrf_score = self.vector_weight * (1.0 / (self.k + rank))
            
            scores[text] = scores.get(text, 0) + rrf_score
            texts[text] = text
            metadata[text] = result.get("metadata", {})
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            text = result["text"]
            rrf_score = self.bm25_weight * (1.0 / (self.k + rank))
            
            scores[text] = scores.get(text, 0) + rrf_score
            if text not in texts:
                texts[text] = text
                metadata[text] = result.get("metadata", {})
        
        # Sort by combined score
        sorted_texts = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
        
        # Format results
        fused_results = []
        for text in sorted_texts:
            fused_results.append({
                "text": text,
                "metadata": metadata[text],
                "score": scores[text]
            })
        
        return fused_results
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            
        Returns:
            List of retrieved documents with fused scores
        """
        # Get results from both retrievers
        # Retrieve more than top_k for better fusion
        retrieve_k = min(top_k * 3, 20)
        
        vector_results = self.vector_retriever.retrieve(query, top_k=retrieve_k)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=retrieve_k)
        
        # Fuse results
        fused_results = self._reciprocal_rank_fusion(vector_results, bm25_results)
        
        # Return top-k
        final_results = fused_results[:top_k]
        
        logger.info(
            f"Hybrid search returned {len(final_results)} results "
            f"(from {len(vector_results)} vector + {len(bm25_results)} BM25)"
        )
        
        return final_results
