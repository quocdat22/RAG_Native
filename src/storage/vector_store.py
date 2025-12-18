"""ChromaDB vector store operations."""
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.ingestion.chunking import Chunk

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """ChromaDB vector store wrapper."""
    
    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "documents"
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(
            f"Initialized ChromaVectorStore at {persist_directory} "
            f"with collection '{collection_name}'"
        )
    
    def add_documents(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        document_id: Optional[str] = None
    ) -> str:
        """
        Add document chunks with embeddings to the vector store.
        
        Args:
            chunks: List of Chunk objects
            embeddings: List of embedding vectors
            document_id: Optional document ID (generated if not provided)
            
        Returns:
            Document ID
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        doc_id = document_id or str(uuid.uuid4())
        
        # Prepare data for ChromaDB
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        documents = [chunk.text for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            metadata = chunk.metadata.copy()
            metadata["document_id"] = doc_id
            metadata["chunk_id"] = chunk.chunk_id
            metadata["token_count"] = chunk.token_count
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} chunks for document {doc_id}")
        return doc_id
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of search results with text, metadata, and scores
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]  # Convert distance to similarity
            })
        
        logger.debug(f"Found {len(formatted_results)} results for query")
        return formatted_results
    
    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            Number of chunks deleted
        """
        # Query for all chunks with this document_id
        results = self.collection.get(
            where={"document_id": document_id}
        )
        
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
            return len(results["ids"])
        
        return 0
    
    def get_all_documents(self) -> List[Dict]:
        """
        Get list of all unique documents in the store.
        
        Returns:
            List of document metadata
        """
        # Get all items
        results = self.collection.get()
        
        # Extract unique documents
        documents = {}
        for metadata in results.get("metadatas", []):
            doc_id = metadata.get("document_id")
            if doc_id and doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "filename": metadata.get("filename"),
                    "file_type": metadata.get("file_type"),
                    "upload_timestamp": metadata.get("upload_timestamp")
                }
        
        return list(documents.values())
    
    def get_document_chunks(self, document_id: str) -> List[Dict]:
        """
        Get all chunks for a specific document.
        
        Args:
            document_id: Document ID to retrieve chunks for
            
        Returns:
            List of chunks with text and metadata
        """
        results = self.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"]
        )
        
        chunks = []
        for i in range(len(results["ids"])):
            chunks.append({
                "chunk_id": results["ids"][i],
                "text": results["documents"][i],
                "metadata": results["metadatas"][i]
            })
            
        # Sort chunks by their original order if available (usually in chunk_id or metadata)
        # Assuming f"{doc_id}_{i}" or metadata has chunk_index
        try:
            chunks.sort(key=lambda x: x["metadata"].get("chunk_id", 0))
        except:
            pass
            
        return chunks
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        documents = self.get_all_documents()
        
        return {
            "total_chunks": count,
            "total_documents": len(documents),
            "collection_name": self.collection_name
        }
    
    def reset(self):
        """Delete all data in the collection (use with caution!)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.warning(f"Reset collection '{self.collection_name}'")


def get_vector_store() -> ChromaVectorStore:
    """Get configured vector store instance."""
    from config.settings import settings
    
    return ChromaVectorStore(
        persist_directory=settings.chroma_dir,
        collection_name="documents"
    )
