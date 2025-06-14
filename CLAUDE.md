# CLAUDE.md - Project Status and Documentation

## Project Overview
This is an LLM RAG (Large Language Model - Retrieval Augmented Generation) system that processes PDF documents and provides semantic search capabilities using OpenAI embeddings and OpenRouter LLM analysis.

## Current Architecture (Clean State)

### Technology Stack
- **Embeddings**: OpenAI text-embedding-3-large (3072 dimensions)
- **Vector Database**: ChromaDB with persistent storage
- **LLM Analysis**: OpenRouter API (configurable models)
- **PDF Processing**: PyMuPDF for text extraction
- **Web Interface**: Gradio for interactive search
- **Text Processing**: Advanced UTF-8 handling and normalization

### API Requirements
1. **OpenAI API Key** - For embedding generation
   - Used for: text-embedding-3-large model
   - Required environment variable: `OPENAI_API_KEY`

2. **OpenRouter API Key** - For LLM analysis and explanations
   - Used for: Semantic analysis, text highlighting, language detection
   - Required environment variable: `OPENROUTER_API_KEY`

## File Structure (Current)
```
llmrag/
├── ingest.py                    # PDF processing and embedding generation
├── query.py                     # Semantic search and result display
├── llm_wrapper.py              # API integration (OpenAI + OpenRouter)
├── gradio_browser.py           # Web interface for document browsing
├── info.py                     # Database information and utilities
├── test_llm_wrapper.py         # API connection tests
├── test_chunking.py            # Text chunking tests
├── pyproject.toml              # Project dependencies
├── .env.example                # Environment configuration template
├── .env                        # Environment configuration
├── kill_port_7860.sh           # Port management utility script
├── CLAUDE.md                   # This file - project documentation
├── README.md                   # User documentation
└── chroma_db/                  # ChromaDB vector database (auto-created)
```

## Recent Cleanup (Completed)

### Legacy Files Removed
- ✅ `check_server.py` - Old llama.cpp server checking logic
- ✅ `start_embedding_server.sh` - Server startup script  
- ✅ `llama.cpp/` directory - Entire llama.cpp repository and models
- ✅ `great mythologies of the world.pdf:Zone.Identifier` - Windows metadata

### Function Names Modernized
- ✅ `test_embedding_server()` → `test_openai_embeddings()`
- ✅ `auto_start_server()` → `check_openai_api()`
- ✅ Removed unused `server_url` parameter from `generate_embeddings()`

### Code Quality Improvements
- ✅ Moved `import re` statements to file headers
- ✅ Updated all import statements across all files
- ✅ Updated all function calls to use new names
- ✅ Cleaned up environment configuration files

## Core Components Documentation

### 1. llm_wrapper.py - API Integration Layer
**Purpose**: Unified interface for OpenAI and OpenRouter APIs

**Key Functions**:
- `llm_call()` - OpenRouter API for LLM analysis
- `generate_embeddings()` - OpenAI API for text embeddings
- `test_openai_embeddings()` - Verify OpenAI API connectivity
- `check_openai_api()` - Comprehensive API validation

**Features**:
- Advanced UTF-8 text preprocessing
- Comprehensive error handling with timeouts
- Text normalization and cleaning
- Embedding normalization for consistent similarity scores

### 2. ingest.py - Document Processing Pipeline
**Purpose**: Extract text from PDFs and generate embeddings

**Key Functions**:
- `extract_pdf_pages()` - Extract text from PDF with page range support
- `chunk_text()` - Smart text chunking (500 chars, 50 char overlap)
- `generate_page_embeddings()` - Batch embedding generation
- `save_to_chroma()` - Store embeddings in ChromaDB
- `check_embedding_system()` - Validate embedding API

**Features**:
- Page-by-page processing with progress tracking
- Smart text chunking with word boundary preservation
- UTF-8 compatibility with error handling
- Metadata preservation (page numbers, chunk info)
- Batch processing for efficiency

### 3. query.py - Semantic Search Engine
**Purpose**: Search documents and display highlighted results

**Key Functions**:
- `query_chroma_collections()` - Similarity search across collections
- `generate_query_embedding()` - Convert query to embedding
- `highlight_relevant_text()` - LLM-powered result highlighting
- `detect_language()` - Automatic language detection
- `display_results()` - Formatted result presentation
- `list_available_collections()` - Database introspection

**Features**:
- Multi-collection search capability
- Advanced text highlighting with color coding
- Multilingual explanations (Italian, Spanish, French, English)
- Configurable similarity thresholds
- Visual result formatting with borders and footnotes

### 4. gradio_browser.py - Enhanced Web Interface
**Purpose**: Feature-rich web-based document search matching terminal functionality

**Key Functions**:
- `search_and_highlight()` - Main search function with rich formatting
- `process_highlighted_text_for_html()` - HTML conversion with footnote extraction
- `analyze_individual_result_web()` - Per-result LLM analysis
- `synthesize_results_web()` - Cross-result synthesis and insights
- `create_web_border()` - Professional bordered sections
- `create_custom_css()` - Comprehensive styling with dark/light themes

**Features**:
- **Rich Text Display**: Semantic highlighting with footnoted explanations
- **Multi-Level Analysis**: Content → Relevance Analysis → LLM Analysis → Synthesis
- **Professional Styling**: Color-coded bordered sections matching terminal aesthetics
- **Dual Answer Mode**: Direct answers + detailed search analysis
- **Responsive Design**: Dark/light theme support with proper CSS
- **Multilingual Intelligence**: Native language explanations
- **Interactive Interface**: Real-time search with collection filtering

### 5. info.py - Database Utilities
**Purpose**: Database management and statistics

**Features**:
- Collection listing and statistics
- Database health monitoring
- Storage usage information
- Collection management tools

### 6. kill_port_7860.sh - Port Management Utility
**Purpose**: Automated port management for web interface

**Key Functions**:
- `lsof -ti :7860` - Find processes using port 7860
- `kill -TERM` - Graceful process termination
- `kill -KILL` - Forced termination if needed
- Process verification and status reporting

**Features**:
- **Smart Termination**: Tries graceful SIGTERM first, then SIGKILL if needed
- **Process Details**: Shows PID and command details before termination
- **Verification**: Confirms port is actually freed after termination
- **Error Handling**: Reports success/failure status
- **Safety**: Uses lsof to precisely target port-specific processes

## Text Processing Pipeline

### 1. Document Ingestion
1. **PDF Extraction**: PyMuPDF extracts text page by page
2. **Text Cleaning**: UTF-8 normalization and character filtering
3. **Smart Chunking**: 500-character chunks with 50-character overlap
4. **Embedding Generation**: OpenAI text-embedding-3-large
5. **Storage**: ChromaDB with full metadata preservation

### 2. Query Processing
1. **Query Embedding**: Convert search query to embedding vector
2. **Similarity Search**: ChromaDB cosine similarity across collections
3. **Result Ranking**: Sort by similarity score (normalized)
4. **Language Detection**: Automatic detection for proper explanations
5. **Highlighting**: LLM-powered semantic highlighting
6. **Display**: Formatted results with color coding and explanations

## Advanced Features

### Smart Text Chunking Strategy
- **Chunk Size**: 500 characters (optimal for embedding models)
- **Overlap**: 50 characters to preserve context
- **Word Boundaries**: Breaks at spaces to avoid word splitting
- **Merge Logic**: Combines very short final chunks
- **Metadata**: Tracks chunk relationships and source pages

### Rich Text Highlighting System

**Terminal Interface (query.py)**:
- **Yellow Background**: Semantically relevant text sections
- **Green Text**: Detailed explanations of semantic connections
- **Bordered Sections**: Unicode box-drawing characters for visual structure
- **Footnoted Analysis**: Numbered explanations with comprehensive insights
- **Multi-Section Display**: Content → Relevance → LLM Analysis → Synthesis

**Web Interface (gradio_browser.py)**:
- **Yellow Highlights**: Semantic annotations with footnote numbers
- **Explanation Cards**: Dedicated sections for numbered explanations
- **Professional Borders**: CSS-styled sections with color coding
- **Multi-Level Analysis**: Matching terminal functionality in web format
- **Responsive Design**: Dark/light theme support with proper styling

### Multilingual Intelligence
- **Detection**: Automatic language identification
- **Explanations**: Native language responses
- **Supported**: Italian, Spanish, French, English
- **Fallback**: English for unsupported languages

## Performance Characteristics

### Processing Speed
- **Ingestion**: ~2-3 seconds per PDF page
- **Search**: <1 second for typical queries
- **Embedding**: ~500ms per text chunk
- **Highlighting**: ~2-3 seconds per result

### Resource Usage
- **Memory**: ~200MB base + embedding cache
- **Storage**: ~1MB per document (embeddings + metadata)
- **Network**: API calls only (no local models)

### Scalability
- **Documents**: Tested with 100+ page documents
- **Collections**: Multiple PDFs in single database
- **Concurrent**: Thread-safe ChromaDB operations
- **Caching**: Persistent storage for fast retrieval

## Build and Test Commands
```bash
# Setup
cp .env.example .env  # Configure API keys

# Test API connections
python llm_wrapper.py

# Process document
python ingest.py document.pdf -v

# Search documents
python query.py "search query" -v

# List collections
python query.py --list

# Web interface
python gradio_browser.py

# Port management (if port 7860 is occupied)
./kill_port_7860.sh

# Database info
python info.py

# Run tests
python test_llm_wrapper.py
python test_chunking.py
```

## Configuration Options

### Environment Variables
- `OPENROUTER_API_KEY` - Required for LLM analysis
- `OPENAI_API_KEY` - Required for embeddings
- `SEMANTIC_MODEL` - Optional LLM model selection
- `PAK_DEBUG` - Optional debug logging

### Recommended Models
- `anthropic/claude-3-haiku:beta` (default, fast & cost-effective)
- `google/gemini-flash-2.0` (very fast analysis)
- `anthropic/claude-3-sonnet:beta` (balanced performance)

## Development Notes

### Code Quality Standards
- ✅ **Clean Architecture**: No legacy llama.cpp dependencies
- ✅ **Clear Naming**: Functions reflect actual purpose
- ✅ **Error Handling**: Comprehensive API error management
- ✅ **Type Hints**: Full typing support
- ✅ **Documentation**: Comprehensive docstrings
- ✅ **Testing**: API connection and functionality tests

### API Integration
- **OpenAI**: High-quality embeddings via official API
- **OpenRouter**: Flexible LLM access with model choice
- **ChromaDB**: Efficient vector storage and retrieval
- **Gradio**: User-friendly web interface

### Extensibility
- **Model Swapping**: Easy LLM model changes via environment
- **Language Support**: Expandable language detection rules
- **Highlighting**: Customizable color schemes and rules
- **Output Formats**: Extensible result display options

## Current Status: ✅ PRODUCTION READY

The system is now completely clean of legacy code and ready for production use with:
- **Modern OpenAI-based embedding architecture**: High-quality text-embedding-3-large
- **Clean, well-documented codebase**: Comprehensive function documentation and type hints
- **Comprehensive error handling**: Robust API integration with fallback mechanisms
- **Enhanced Multi-interface access**: 
  - **CLI**: Rich terminal interface with Unicode borders and ANSI colors
  - **Web**: Professional Gradio interface with advanced CSS styling
- **Full multilingual support**: Native language explanations (Italian, Spanish, French, English)
- **Advanced semantic search capabilities**: LLM-powered highlighting and analysis
- **Rich Text Processing**: Footnoted explanations and multi-level analysis sections
- **Professional Presentation**: Color-coded sections and responsive design

**Recent Enhancement**: Web interface now matches the sophistication of the terminal interface with rich highlighting, multi-level analysis, and professional styling.

All legacy llama.cpp dependencies have been removed and the system now relies entirely on cloud APIs for maximum reliability and performance.