"""Text chunking with token-based splitting."""
import logging
from typing import Dict, List, Optional

import tiktoken

from src.ingestion.loaders import DocumentPage

logger = logging.getLogger(__name__)


class Chunk:
    """Represents a text chunk with metadata."""
    
    def __init__(
        self,
        text: str,
        chunk_id: str,
        metadata: Dict,
        token_count: int
    ):
        self.text = text
        self.chunk_id = chunk_id
        self.metadata = metadata
        self.token_count = token_count
    
    def to_dict(self) -> Dict:
        """Convert chunk to dictionary."""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
            "token_count": self.token_count
        }


class TextChunker:
    """Token-based text chunking with overlap."""
    
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            encoding_name: Tiktoken encoding name
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        logger.info(
            f"Initialized TextChunker with chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, metadata: Dict) -> List[Chunk]:
        """
        Split text into chunks based on token count.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of Chunk objects
        """
        # Encode the entire text
        tokens = self.encoding.encode(text)
        
        chunks = []
        start_idx = 0
        chunk_num = 0
        
        while start_idx < len(tokens):
            # Extract chunk tokens
            end_idx = start_idx + self.chunk_size
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Create chunk with metadata
            chunk_id = f"{metadata.get('filename', 'unknown')}_{chunk_num}"
            chunk = Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                metadata=metadata.copy(),
                token_count=len(chunk_tokens)
            )
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start_idx += (self.chunk_size - self.chunk_overlap)
            chunk_num += 1
        
        return chunks
    
    def chunk_documents(self, pages: List[DocumentPage]) -> List[Chunk]:
        """
        Chunk a list of document pages.
        
        Args:
            pages: List of DocumentPage objects
            
        Returns:
            List of Chunk objects
        """
        all_chunks = []
        
        for page in pages:
            # Prepare metadata for this page
            page_metadata = page.metadata.to_dict()
            page_metadata["page_number"] = page.page_number
            
            # Chunk the page content
            chunks = self.chunk_text(page.content, page_metadata)
            all_chunks.extend(chunks)
        
        logger.info(
            f"Created {len(all_chunks)} chunks from {len(pages)} pages "
            f"for document: {pages[0].metadata.filename if pages else 'unknown'}"
        )
        
        return all_chunks


def smart_chunk_documents(
    pages: List[DocumentPage],
    chunk_size: int = 800,
    chunk_overlap: int = 200
) -> List[Chunk]:
    """
    Convenience function to chunk documents with default settings.
    
    Args:
        pages: List of DocumentPage objects
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        
    Returns:
        List of Chunk objects
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_documents(pages)
