"""Zilliz Cloud (Milvus) vector store operations."""
import logging
import uuid
from typing import Dict, List, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

from src.ingestion.chunking import Chunk

logger = logging.getLogger(__name__)


class ZillizVectorStore:
    """Zilliz Cloud vector store wrapper using Milvus SDK."""
    
    def __init__(
        self,
        uri: str,
        token: str,
        collection_name: str = "documents",
        dimension: int = 1536
    ):
        """
        Initialize Zilliz vector store.
        
        Args:
            uri: Zilliz Cloud URI endpoint
            token: Zilliz API token
            collection_name: Name of the collection
            dimension: Embedding vector dimension
        """
        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.dimension = dimension
        
        # Initialize Milvus client
        self.client = MilvusClient(
            uri=uri,
            token=token
        )
        
        # Create or get collection
        self._ensure_collection()
        
        logger.info(
            f"Initialized ZillizVectorStore with collection '{collection_name}' "
            f"(dimension={dimension})"
        )
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        # Check if collection exists
        if self.client.has_collection(self.collection_name):
            logger.info(f"Collection '{self.collection_name}' already exists")
            return
        
        logger.info(f"Creating collection '{self.collection_name}'...")
        
        # Define schema
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True  # Allow dynamic metadata fields
        )
        
        # Add fields
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=100
        )
        
        schema.add_field(
            field_name="document_id",
            datatype=DataType.VARCHAR,
            max_length=100
        )
        
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=65535  # Max text length
        )
        
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dimension
        )
        
        # Create index params for vector search
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",  # Zilliz auto-selects best index
            metric_type="COSINE"
        )
        
        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        
        logger.info(f"✅ Created collection '{self.collection_name}'")
    
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
        
        # Prepare data for insertion
        entities = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Prepare metadata
            metadata = chunk.metadata.dict() if hasattr(chunk.metadata, 'dict') else {}
            
            entity = {
                "id": f"{doc_id}_{i}",
                "document_id": doc_id,
                "text": chunk.text,
                "embedding": embedding,
                "chunk_id": chunk.chunk_id,
                "token_count": chunk.token_count,
                # Flatten metadata fields
                "filename": metadata.get("filename"),
                "file_type": metadata.get("file_type"),
                "page": metadata.get("page_number"),
                "upload_timestamp": metadata.get("upload_timestamp"),
                "authors": str(metadata.get("authors")) if metadata.get("authors") else None,
                "year": metadata.get("year"),
                "keywords": str(metadata.get("keywords")) if metadata.get("keywords") else None,
                "abstract": metadata.get("abstract"),
                "doi": metadata.get("doi"),
                "arxiv_id": metadata.get("arxiv_id"),
                "venue": metadata.get("venue"),
            }
            
            # Remove None values
            entity = {k: v for k, v in entity.items() if v is not None}
            entities.append(entity)
        
        # Insert into collection
        self.client.insert(
            collection_name=self.collection_name,
            data=entities
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
        # Build filter expression if provided
        filter_expr = None
        if filter_dict:
            filter_parts = []
            for key, value in filter_dict.items():
                if isinstance(value, str):
                    filter_parts.append(f'{key} == "{value}"')
                else:
                    filter_parts.append(f'{key} == {value}')
            filter_expr = " && ".join(filter_parts) if filter_parts else None
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["id", "document_id", "text", "filename", "file_type", "page", 
                          "authors", "year", "keywords", "abstract", "doi", "arxiv_id", "venue"],
            filter=filter_expr
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                # Reconstruct metadata
                metadata = {
                    "document_id": hit.get("document_id"),
                    "filename": hit.get("filename"),
                    "file_type": hit.get("file_type"),
                    "page": hit.get("page"),
                }
                
                # Add optional metadata
                if hit.get("authors"):
                    metadata["authors"] = hit.get("authors")
                if hit.get("year"):
                    metadata["year"] = hit.get("year")
                if hit.get("keywords"):
                    metadata["keywords"] = hit.get("keywords")
                if hit.get("abstract"):
                    metadata["abstract"] = hit.get("abstract")
                if hit.get("doi"):
                    metadata["doi"] = hit.get("doi")
                if hit.get("arxiv_id"):
                    metadata["arxiv_id"] = hit.get("arxiv_id")
                if hit.get("venue"):
                    metadata["venue"] = hit.get("venue")
                
                formatted_results.append({
                    "id": hit.get("id"),
                    "text": hit.get("text"),
                    "metadata": metadata,
                    "score": hit.get("distance", 0)  # Cosine similarity score
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
        # Delete by filter
        filter_expr = f'document_id == "{document_id}"'
        
        result = self.client.delete(
            collection_name=self.collection_name,
            filter=filter_expr
        )
        
        deleted_count = result.get("delete_count", 0)
        logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
        return deleted_count
    
    def count_document_chunks(self, document_id: str) -> int:
        """
        Count chunks for a specific document.
        
        Args:
            document_id: Document ID to count chunks for
            
        Returns:
            Number of chunks for this document
        """
        filter_expr = f'document_id == "{document_id}"'
        
        results = self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=["id"],
            limit=10000  # Max limit to count all
        )
        
        return len(results)
    
    def get_all_documents(self) -> List[Dict]:
        """
        Get list of all unique documents in the store.
        
        Returns:
            List of document metadata
        """
        # Query all documents with limit
        results = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["document_id", "filename", "file_type", "upload_timestamp",
                          "authors", "year", "keywords", "abstract", "doi", "arxiv_id", "venue"],
            limit=10000
        )
        
        # Extract unique documents
        documents = {}
        for item in results:
            doc_id = item.get("document_id")
            if doc_id and doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "filename": item.get("filename"),
                    "file_type": item.get("file_type"),
                    "upload_timestamp": item.get("upload_timestamp"),
                    "authors": item.get("authors"),
                    "year": item.get("year"),
                    "keywords": item.get("keywords"),
                    "abstract": item.get("abstract"),
                    "doi": item.get("doi"),
                    "arxiv_id": item.get("arxiv_id"),
                    "venue": item.get("venue")
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
        filter_expr = f'document_id == "{document_id}"'
        
        results = self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=["id", "text", "filename", "file_type", "page", 
                          "authors", "year", "keywords"],
            limit=10000
        )
        
        chunks = []
        for item in results:
            metadata = {
                "filename": item.get("filename"),
                "file_type": item.get("file_type"),
                "page": item.get("page"),
                "authors": item.get("authors"),
                "year": item.get("year"),
                "keywords": item.get("keywords"),
            }
            metadata = {k: v for k, v in metadata.items() if v is not None}
            
            chunks.append({
                "id": item.get("id"),
                "text": item.get("text"),
                "metadata": metadata
            })
        
        return chunks
    
    def count(self) -> int:
        """
        Get total count of chunks in collection.
        
        Returns:
            Total number of chunks
        """
        stats = self.client.get_collection_stats(self.collection_name)
        return stats.get("row_count", 0)
    
    def get(self, limit: Optional[int] = None, **kwargs) -> Dict:
        """
        Get all chunks from collection (ChromaDB-compatible API).
        
        Args:
            limit: Maximum number of chunks to return
            **kwargs: Additional query parameters (ignored for compatibility)
            
        Returns:
            Dictionary with 'documents' and 'metadatas' keys
        """
        # Query all chunks
        results = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["id", "document_id", "text", "filename", "file_type", "page",
                          "authors", "year", "keywords", "abstract", "doi", "arxiv_id", "venue"],
            limit=limit or 10000
        )
        
        # Format results in ChromaDB format
        documents = []
        metadatas = []
        
        for item in results:
            documents.append(item.get("text", ""))
            
            metadata = {
                "document_id": item.get("document_id"),
                "filename": item.get("filename"),
                "file_type": item.get("file_type"),
                "page": item.get("page"),
            }
            
            # Add optional fields if present
            if item.get("authors"):
                metadata["authors"] = item.get("authors")
            if item.get("year"):
                metadata["year"] = item.get("year")
            if item.get("keywords"):
                metadata["keywords"] = item.get("keywords")
            if item.get("abstract"):
                metadata["abstract"] = item.get("abstract")
            if item.get("doi"):
                metadata["doi"] = item.get("doi")
            if item.get("arxiv_id"):
                metadata["arxiv_id"] = item.get("arxiv_id")
            if item.get("venue"):
                metadata["venue"] = item.get("venue")
            
            metadatas.append(metadata)
        
        return {
            "documents": documents,
            "metadatas": metadatas
        }
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        stats = self.client.get_collection_stats(self.collection_name)
        
        # Count unique documents
        all_docs = self.get_all_documents()
        
        return {
            "total_chunks": stats.get("row_count", 0),
            "total_documents": len(all_docs),
            "collection_name": self.collection_name
        }


# Singleton instance
_zilliz_store: Optional[ZillizVectorStore] = None


def get_zilliz_store(
    uri: Optional[str] = None,
    token: Optional[str] = None,
    collection_name: str = "documents",
    dimension: int = 1536
) -> ZillizVectorStore:
    """
    Get or create Zilliz vector store singleton.
    
    Args:
        uri: Zilliz Cloud URI (uses settings if not provided)
        token: Zilliz token (uses settings if not provided)
        collection_name: Collection name
        dimension: Embedding dimension
        
    Returns:
        ZillizVectorStore instance
    """
    global _zilliz_store
    
    if _zilliz_store is None:
        from config.settings import settings
        
        uri = uri or settings.zilliz_uri
        token = token or settings.zilliz_token
        
        if not uri or not token:
            raise ValueError(
                "Zilliz URI and token must be provided or set in environment variables "
                "(ZILLIZ_URI and ZILLIZ_TOKEN)"
            )
        
        _zilliz_store = ZillizVectorStore(
            uri=uri,
            token=token,
            collection_name=collection_name,
            dimension=dimension
        )
    
    return _zilliz_store
