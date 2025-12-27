"""Search routes."""
import logging

from fastapi import APIRouter, HTTPException

from src.api.schemas import SearchRequest, SearchResponse, SearchResult
from src.embedding.embedder import get_embedder
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_retriever import VectorRetriever
from src.storage.vector_store import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


def _initialize_retrievers():
    """Initialize retrievers for search."""
    vector_store = get_vector_store()
    embedder = get_embedder()
    
    # Vector retriever
    vector_retriever = VectorRetriever(vector_store, embedder)
    
    # BM25 retriever - needs to be indexed with all documents
    bm25_retriever = BM25Retriever()
    
    # Get all documents from vector store for BM25 indexing
    all_results = vector_store.get()
    if all_results["documents"]:
        documents = [
            {"text": text, "metadata": metadata}
            for text, metadata in zip(all_results["documents"], all_results["metadatas"])
        ]
        bm25_retriever.index_documents(documents)
    
    # Hybrid retriever
    from config.settings import settings
    hybrid_retriever = HybridRetriever(
        vector_retriever,
        bm25_retriever,
        vector_weight=settings.retrieval.vector_weight,
        bm25_weight=settings.retrieval.bm25_weight
    )
    
    return vector_retriever, bm25_retriever, hybrid_retriever


@router.post("", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search documents using vector, BM25, or hybrid search.
    
    Args:
        request: Search request with query and parameters
        
    Returns:
        Search results with scores
    """
    try:
        vector_retriever, bm25_retriever, hybrid_retriever = _initialize_retrievers()
        
        # Select retriever based on search type
        if request.search_type == "vector":
            results = vector_retriever.retrieve(request.query, top_k=request.top_k)
        elif request.search_type == "bm25":
            results = bm25_retriever.retrieve(request.query, top_k=request.top_k)
        else:  # hybrid
            results = hybrid_retriever.retrieve(request.query, top_k=request.top_k)
        
        # Format results
        search_results = [
            SearchResult(
                text=result["text"],
                score=result["score"],
                metadata=result.get("metadata", {})
            )
            for result in results
        ]
        
        logger.info(
            f"Search completed: query='{request.query[:50]}...', "
            f"type={request.search_type}, results={len(search_results)}"
        )
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            search_type=request.search_type
        )
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))
