# RAG Native - Research Assistant System

A RAG (Retrieval-Augmented Generation) system designed for researchers to search and query their document collections (scientific papers, books, etc.).

## Features

- **Document Upload**: Support for PDF, DOCX, and TXT files
- **Hybrid Search**: Combines vector similarity (embeddings) and keyword search (BM25)
- **Q&A Interface**: Ask questions and get answers with source citations
- **Citation Tracking**: Automatic citation with filename and page numbers

## Tech Stack

- **Backend**: FastAPI + Python 3.11
- **Vector Database**: ChromaDB (embedded)
- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: OpenAI GPT-4o-mini
- **Frontend**: Streamlit

## Prerequisites

- Python 3.11+
- UV package manager
- OpenAI API key

## Setup

1. **Clone and navigate to the project:**
```bash
cd RAG_Native
```

2. **Install UV** (if not already installed):
```bash
pip install uv
```

3. **Create virtual environment and install dependencies:**
```bash
uv venv
uv pip install -e .
```

4. **Configure environment variables:**
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage

### Start the Backend API

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

### Start the Frontend

```bash
uv run streamlit run ui/app.py
```

Then open http://localhost:8501 in your browser.

## Project Structure

```
RAG_Native/
├── config/           # Configuration settings
├── src/
│   ├── ingestion/    # Document loading and chunking
│   ├── embedding/    # OpenAI embeddings wrapper
│   ├── storage/      # ChromaDB vector store
│   ├── retrieval/    # Search (vector, BM25, hybrid)
│   ├── generation/   # LLM response generation
│   └── api/          # FastAPI routes
├── ui/               # Streamlit frontend
├── data/             # Document and database storage
├── logs/             # Application logs
└── tests/            # Unit and integration tests
```

## License

MIT
