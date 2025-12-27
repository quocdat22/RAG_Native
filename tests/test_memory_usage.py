"""Test memory usage of the RAG system components."""
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.utils.memory_monitor import log_memory_usage, check_memory_limit, format_memory_stats


def test_baseline():
    """Test baseline memory usage."""
    print("\n" + "=" * 60)
    print("1. Baseline Memory (before loading any components)")
    print("=" * 60)
    log_memory_usage("Baseline")
    print(format_memory_stats())


def test_vector_store():
    """Test vector store memory usage."""
    print("\n" + "=" * 60)
    print("2. After Loading Vector Store")
    print("=" * 60)
    from src.storage.vector_store import get_vector_store
    
    vector_store = get_vector_store()
    log_memory_usage("Vector Store")
    print(f"Collection stats: {vector_store.get_collection_stats()}")
    print(format_memory_stats())


def test_embedder():
    """Test embedder memory usage."""
    print("\n" + "=" * 60)
    print("3. After Loading Embedder")
    print("=" * 60)
    from src.embedding.embedder import get_embedder
    
    embedder = get_embedder()
    log_memory_usage("Embedder")
    print(format_memory_stats())


def test_retriever():
    """Test retriever memory usage."""
    print("\n" + "=" * 60)
    print("4. After Initializing Retrievers")
    print("=" * 60)
    from src.storage.vector_store import get_vector_store
    from src.embedding.embedder import get_embedder
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.vector_retriever import VectorRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    
    vector_store = get_vector_store()
    embedder = get_embedder()
    
    vector_retriever = VectorRetriever(vector_store, embedder)
    log_memory_usage("Vector Retriever")
    
    # BM25 - this is the memory-heavy part
    bm25_retriever = BM25Retriever()
    all_results = vector_store.collection.get(limit=5000)
    
    if all_results["documents"]:
        documents = [
            {"text": text, "metadata": metadata}
            for text, metadata in zip(all_results["documents"], all_results["metadatas"])
        ]
        bm25_retriever.index_documents(documents)
        log_memory_usage("BM25 Retriever (indexed)")
    
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
    log_memory_usage("Hybrid Retriever")
    print(format_memory_stats())


def test_llm():
    """Test LLM memory usage."""
    print("\n" + "=" * 60)
    print("5. After Loading LLM Generator")
    print("=" * 60)
    from src.generation.llm import get_generator
    
    generator = get_generator()
    log_memory_usage("LLM Generator")
    print(format_memory_stats())


def test_memory_limit():
    """Test memory limit check."""
    print("\n" + "=" * 60)
    print("6. Memory Limit Check (Render Free Tier: 512MB)")
    print("=" * 60)
    
    # Check against Render free tier limit
    is_ok = check_memory_limit(limit_mb=512, warning_threshold=0.8)
    
    if is_ok:
        print("✅ Memory usage is within safe limits!")
    else:
        print("⚠️ Memory usage is high or exceeded limit!")
    
    print(format_memory_stats())


def main():
    """Run all memory tests."""
    print("\n" + "=" * 60)
    print("RAG System Memory Usage Test")
    print("=" * 60)
    
    try:
        test_baseline()
        test_vector_store()
        test_embedder()
        test_retriever()
        test_llm()
        test_memory_limit()
        
        print("\n" + "=" * 60)
        print("Test Complete!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print("\n⚠️ Test failed. See error above.")


if __name__ == "__main__":
    main()
