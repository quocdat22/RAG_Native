# RAG Native - Changelog

## Version 0.1.0 (Initial Release)

### Features

#### Document Processing
- ✅ PDF document loading with page tracking
- ✅ DOCX document loading with section tracking
- ✅ TXT file support
- ✅ Token-based chunking (500-1000 tokens, configurable)
- ✅ Chunk overlap for context preservation
- ✅ Metadata extraction (filename, page numbers, timestamps)

#### Vector Storage & Retrieval
- ✅ ChromaDB embedded vector database
- ✅ OpenAI text-embedding-3-small embeddings
- ✅ Cosine similarity search
- ✅ BM25 keyword search
- ✅ Hybrid retrieval with Reciprocal Rank Fusion (RRF)
- ✅ Configurable retrieval weights

#### Generation
- ✅ OpenAI GPT-4o-mini for answer generation
- ✅ RAG-specific prompt templates
- ✅ Citation-aware responses
- ✅ Source attribution with filenames and page numbers
- ✅ Streaming response support

#### API (FastAPI)
- ✅ Document upload endpoint
- ✅ Document management (list, delete)
- ✅ Search endpoint (vector/BM25/hybrid)
- ✅ Chat endpoint for Q&A
- ✅ Streaming chat endpoint
- ✅ Health check endpoints
- ✅ CORS middleware
- ✅ Comprehensive logging

#### Frontend (Streamlit)
- ✅ Chat interface with message history
- ✅ Document upload via sidebar
- ✅ Document library management
- ✅ Source citations display
- ✅ Configurable search settings
- ✅ Retrieval parameter controls

#### Configuration
- ✅ Pydantic-based settings management
- ✅ Environment variable configuration
- ✅ Validation for hyperparameters
- ✅ Auto-creation of required directories

#### Developer Experience
- ✅ UV package manager support
- ✅ Clean project structure
- ✅ Type hints throughout
- ✅ Comprehensive logging
- ✅ Error handling
- ✅ Installation verification script

### Technical Stack

- **Language**: Python 3.11+
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Streamlit
- **Vector DB**: ChromaDB (embedded)
- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: OpenAI GPT-4o-mini
- **Retrieval**: Vector + BM25 hybrid
- **Package Manager**: UV

### Known Limitations

- Single-user application (no authentication)
- No conversation history persistence
- No support for images/tables in PDFs yet
- BM25 index rebuilt on each API restart
- English language optimized

### Future Enhancements

Potential areas for improvement:
- [ ] Multi-user support with authentication
- [ ] Conversation history database
- [ ] Advanced PDF parsing (tables, figures)
- [ ] Multiple language support
- [ ] Query expansion and rewriting
- [ ] Document summarization
- [ ] Batch document processing
- [ ] Export functionality (citations, summaries)
- [ ] Docker containerization
- [ ] Cloud deployment guides
