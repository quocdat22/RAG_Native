"""Basic unit tests for document loaders."""
import pytest
from pathlib import Path
from src.ingestion.loaders import DocumentLoader, PDFLoader, DOCXLoader, TXTLoader


def test_txt_loader():
    """Test TXT file loading."""
    # This is a basic test structure
    # In production, you would create actual test files
    pass


def test_pdf_loader():
    """Test PDF file loading."""
    # Would need a sample PDF file
    pass


def test_docx_loader():
    """Test DOCX file loading."""
    # Would need a sample DOCX file
    pass


def test_document_loader_type_detection():
    """Test automatic file type detection."""
    # Test that correct loader is selected
    pass


def test_unsupported_file_type():
    """Test that unsupported file types raise ValueError."""
    with pytest.raises(ValueError):
        DocumentLoader.load(Path("test.xyz"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
