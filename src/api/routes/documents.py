"""Document management routes."""
import logging
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from config.settings import settings
from src.api.schemas import (
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
)
from src.embedding.embedder import get_embedder
from src.ingestion.chunking import smart_chunk_documents
from src.ingestion.loaders import DocumentLoader
from src.storage.vector_store import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document.
    
    Args:
        file: Uploaded file (PDF, DOCX, or TXT)
        
    Returns:
        Upload response with document ID and chunk count
    """
    try:
        # Validate file type
        allowed_extensions = {".pdf", ".docx", ".txt"}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Save uploaded file
        file_path = settings.documents_dir / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"Saved uploaded file: {file.filename}")
        
        # Load document
        pages = DocumentLoader.load(file_path)
        
        # Chunk document
        chunks = smart_chunk_documents(
            pages,
            chunk_size=settings.chunking.size,
            chunk_overlap=settings.chunking.overlap
        )
        
        # Generate embeddings
        embedder = get_embedder()
        texts = [chunk.text for chunk in chunks]
        embeddings = embedder.embed_texts(texts)
        
        # Store in vector database
        vector_store = get_vector_store()
        document_id = vector_store.add_documents(chunks, embeddings)
        
        logger.info(
            f"Processed document {file.filename}: "
            f"{len(chunks)} chunks, document_id={document_id}"
        )
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            chunk_count=len(chunks)
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """
    List all uploaded documents.
    
    Returns:
        List of documents with metadata
    """
    try:
        vector_store = get_vector_store()
        documents = vector_store.get_all_documents()
        
        doc_infos = [
            DocumentInfo(
                document_id=doc["document_id"],
                filename=doc["filename"],
                file_type=doc["file_type"],
                upload_timestamp=doc["upload_timestamp"]
            )
            for doc in documents
        ]
        
        return DocumentListResponse(
            documents=doc_infos,
            total=len(doc_infos)
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(document_id: str):
    """
    Delete a document and all its chunks.
    
    Args:
        document_id: Document ID to delete
        
    Returns:
        Deletion confirmation
    """
    try:
        vector_store = get_vector_store()
        chunks_deleted = vector_store.delete_document(document_id)
        
        if chunks_deleted == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"Deleted document {document_id}: {chunks_deleted} chunks")
        
        return DocumentDeleteResponse(
            document_id=document_id,
            chunks_deleted=chunks_deleted
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))
