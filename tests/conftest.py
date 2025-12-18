"""Test configuration."""
import pytest


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "This is a sample text for testing purposes. It contains multiple sentences."


@pytest.fixture
def api_base_url():
    """API base URL for testing."""
    return "http://localhost:8000"
