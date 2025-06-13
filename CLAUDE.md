# CLAUDE.md - Project Status and Recent Changes

## Project Overview
This is an LLM RAG (Large Language Model - Retrieval Augmented Generation) system that processes documents and provides semantic search capabilities using embeddings.

## Recent Changes

### Latest Commits (as of current session)
- **4fa14ab**: Add semantic search example and enhance document processing
- **019d9c7**: Add .env.example template with Gemma 3n as default recommendation
- **c7c3709**: Update README with comprehensive llama.cpp server architecture documentation
- **56c6ce0**: Add progress tracking to ingestion process and database info utility
- **4820624**: Enhance query functionality with multi-collection search and improved CLI

### Current Uncommitted Changes

#### Modified Files:

**ingest.py**:
- Enhanced UTF-8 compatibility by adding encoding/decoding with error handling
- Improved text processing to handle problematic characters during PDF extraction
- Line 125: Added UTF-8 encoding/decoding to prevent character encoding issues

**llm_wrapper.py**:
- **Major refactor**: Switched from server-based embedding generation to direct binary execution
- Replaced HTTP API calls to llama.cpp server with direct calls to `llama-embedding` binary
- Added comprehensive text cleaning and preprocessing for embedding generation
- Improved error handling and timeout management (120 seconds)
- Added text truncation (max 12k characters) to prevent processing issues
- Enhanced Unicode character handling and normalization
- Updated `test_embedding_server()` and `auto_start_server()` functions to work with binary approach
- Removed dependency on embedding server startup scripts

**query.py**:
- **Aligned with binary architecture**: Updated to use direct binary execution approach
- Replaced `check_embedding_server()` with `check_embedding_system()` for binary validation
- Updated `generate_query_embedding()` to remove server URL dependency
- Modified main function to use binary-based embedding generation
- Consistent with ingest.py and gradio_browser.py architecture

**gradio_browser.py**:
- **Aligned with binary architecture**: Updated imports and function calls
- Replaced server-based embedding calls with direct binary approach
- Updated `search_and_highlight()` to use `test_embedding_server()` and `generate_embeddings()`
- Removed dependency on server URL parameters
- Now fully compatible with binary-based embedding system

#### New Files/Directories:

**chroma_db/**:
- ChromaDB database files for vector storage
- Contains embedding vectors and metadata for processed documents

**gradio_browser.py**:
- New Gradio-based web interface for document browsing and search
- Advanced multi-color highlighting with semantic relevance analysis
- Sophisticated CSS styling and visual interface

**llama.cpp/**:
- Complete llama.cpp repository clone
- Contains embedding model: `models/qwen3-embedding-0.6b-q8_0.gguf`
- Built binaries in `build/bin/` directory including `llama-embedding`

## Key Architecture Changes

### Embedding Generation
- **Before**: Used HTTP API server approach requiring separate server process
- **After**: Direct binary execution using `llama-embedding` with better reliability and performance

### Text Processing Improvements
- Enhanced UTF-8 handling throughout the pipeline
- Better text normalization and cleaning
- Improved error handling for problematic documents

## File Structure
```
llmrag/
├── ingest.py              # Document ingestion and processing
├── llm_wrapper.py         # LLM and embedding generation interface
├── query.py              # Search and query functionality
├── info.py               # Database information utility
├── gradio_browser.py     # Web UI for document browsing
├── chroma_db/            # Vector database storage
└── llama.cpp/            # llama.cpp repository with models and binaries
```

## Build and Test Commands
Based on project structure, likely commands:
- `python ingest.py` - Process and ingest documents
- `python query.py` - Query the knowledge base
- `python info.py` - Get database information
- `python gradio_browser.py` - Launch web interface

## Development Notes
- **All components now aligned**: ingest.py, query.py, and gradio_browser.py use consistent binary-based architecture
- The system uses direct binary execution for embeddings instead of server-based approach
- UTF-8 handling has been improved across document processing pipeline
- ChromaDB is used for vector storage and similarity search
- Qwen3 embedding model (0.6B parameters, Q8_0 quantized) for high-quality embeddings
- No server startup required - embeddings generated directly via `llama-embedding` binary

## Architecture Status
- ✅ **ingest.py**: Fully updated to binary approach
- ✅ **llm_wrapper.py**: Core binary execution implementation
- ✅ **query.py**: Aligned with binary architecture (no server dependencies)
- ✅ **gradio_browser.py**: Aligned with binary architecture (no server dependencies)
- ✅ **All tests passing**: Import, embedding generation, and functionality verified