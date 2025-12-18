"""Pydantic schemas for API requests and responses."""
from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Response after document upload."""
    document_id: str
    filename: str
    chunk_count: int
    status: str = "success"


class DocumentInfo(BaseModel):
    """Document information."""
    document_id: str
    filename: str
    file_type: str
    upload_timestamp: str


class DocumentListResponse(BaseModel):
    """Response for listing documents."""
    documents: List[DocumentInfo]
    total: int


class DocumentDeleteResponse(BaseModel):
    """Response after document deletion."""
    document_id: str
    chunks_deleted: int
    status: str = "success"


class SearchRequest(BaseModel):
    """Request for document search."""
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(5, ge=1, le=20, description="Number of results")
    search_type: Literal["vector", "bm25", "hybrid"] = Field("hybrid", description="Search method")


class SearchResult(BaseModel):
    """Individual search result."""
    text: str
    score: float
    metadata: Dict


class SearchResponse(BaseModel):
    """Response for search request."""
    query: str
    results: List[SearchResult]
    search_type: str


class SourceCitation(BaseModel):
    """Source citation information."""
    filename: str
    page: int | str
    file_type: str


class ChatRequest(BaseModel):
    """Request for chat/Q&A."""
    query: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    search_type: Literal["vector", "bm25", "hybrid"] = Field("hybrid", description="Search method")


class ChatResponse(BaseModel):
    """Response for chat/Q&A."""
    query: str
    answer: str
    sources: List[SourceCitation]
    search_type: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    collection_stats: Optional[Dict] = None
