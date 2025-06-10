# LLM RAG System

A powerful Retrieval-Augmented Generation (RAG) system that processes PDF documents, generates semantic embeddings, and enables intelligent querying with highlighted results and relevance analysis.

## Features

- **PDF Text Extraction**: Extract text content from PDF documents page by page
- **Semantic Embeddings**: Generate high-quality embeddings using sentence-transformers
- **Intelligent Querying**: Search documents using semantic similarity
- **Multi-language Support**: Automatic language detection with native language explanations
- **Visual Results**: Highlighted relevant text with color-coded explanations
- **Flexible Models**: Configurable embedding and LLM models

## Installation

### Prerequisites

- Python 3.12+
- Poetry (recommended) or pip

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd llmrag
   ```

2. **Install dependencies**:
   ```bash
   # Using Poetry (recommended)
   poetry install
   poetry shell

   # Or using pip
   pip install -r requirements.txt
   ```

3. **Environment Configuration**:
   Create a `.env` file in the project root:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   SEMANTIC_MODEL=anthropic/claude-3-haiku:beta
   OPENROUTER_APP_URL=http://localhost
   OPENROUTER_APP_TITLE=LLM_RAG_System
   PAK_DEBUG=false
   ```

## Usage

### 1. Document Ingestion

Process a PDF document to extract text and generate embeddings:

```bash
./ingest.py path/to/your/document.pdf
```

**Options**:
- `-v, --verbose`: Enable verbose logging

**Output**:
- Text and embeddings stored in ChromaDB collection: `pdf_{pdf_name}`
- Database location: `./chroma_db/`

### 2. Querying Documents

Search processed documents using semantic queries:

```bash
./query.py document_name "your search query"
```

**Options**:
- `-k, --top-k N`: Number of top results to show (default: 5)
- `-s, --min-similarity FLOAT`: Minimum similarity threshold (default: 0.0)
- `-t, --show-text`: Show full text content with highlights
- `-v, --verbose`: Enable verbose logging

**Examples**:
```bash
# Basic search
./query.py pricetomorrow "pricing strategies"

# Show top 3 results with text content
./query.py pricetomorrow "market analysis" -k 3 -t

# Filter by similarity threshold
./query.py pricetomorrow "competitive advantage" -s 0.3 -t
```

### 3. Testing LLM Connection

Verify your LLM configuration:

```bash
python llm_wrapper.py
```

## Architecture

### Core Components

1. **ingest.py**: Document processing pipeline
   - PDF text extraction using PyMuPDF
   - Embedding generation via Qwen3 llama.cpp server
   - Storage in ChromaDB vector database

2. **query.py**: Semantic search engine
   - ChromaDB similarity search
   - Multi-language content analysis
   - Visual result highlighting with explanations

3. **llm_wrapper.py**: LLM integration layer
   - OpenRouter API integration for text analysis
   - Qwen3 embedding server communication
   - Auto-start server functionality

4. **check_server.py**: Server management utility
   - Check embedding server status
   - Auto-start server functionality

5. **start_embedding_server.sh**: Embedding server launcher
   - Starts Qwen3 model via llama.cpp
   - Configures embedding-specific parameters

### Embedding Models

**Primary**: Qwen3-Embedding-0.6B-Q8_0 (via llama.cpp server)
- High-quality multilingual embeddings
- Runs via dedicated embedding server on port 8080
- Auto-start functionality available

**Storage**: ChromaDB vector database
- Persistent storage in `./chroma_db/`
- Efficient similarity search
- Metadata preservation

### LLM Models

**Default**: anthropic/claude-3-haiku:beta
**Configurable**: Set via `SEMANTIC_MODEL` environment variable

## Output Features

### Visual Search Results

- **Color-coded highlights**: Yellow background for relevant text
- **Explanations**: Green text explaining semantic connections
- **Footnote system**: Numbered references for detailed analysis
- **Bordered layout**: Clean, organized result presentation

### Multi-language Support

Automatic language detection with native explanations:
- **Italian**: Spiegazioni in italiano naturale
- **Spanish**: Explicaciones en español natural
- **French**: Explications en français naturel
- **English**: Natural English explanations

### Similarity Scoring

- **Cosine similarity**: Precise semantic matching
- **Ranked results**: Best matches first
- **Threshold filtering**: Configurable minimum similarity
- **Statistical summary**: Performance metrics

## File Structure

```
llmrag/
├── ingest.py                    # PDF processing and embedding generation
├── query.py                     # Semantic search and result display
├── llm_wrapper.py              # LLM API integration and embedding server communication
├── check_server.py             # Embedding server status checker
├── start_embedding_server.sh   # Embedding server launcher script
├── pyproject.toml              # Project dependencies and metadata
├── llama.cpp/                  # llama.cpp submodule for embedding server
│   ├── models/                 # Embedding model files
│   └── build/                  # Compiled binaries
├── .gitignore                  # Git ignore patterns
└── README.md                   # This file

# Generated files (ignored by git):
├── .env                        # Environment configuration
└── chroma_db/                  # ChromaDB vector database storage
    ├── chroma.sqlite3          # Database file
    └── {collection_dirs}/      # Collection data
```

## Dependencies

### Core Libraries
- `pymupdf` (1.23.28+): PDF text extraction
- `chromadb` (0.4.22+): Vector database for embeddings
- `numpy` (1.24.0+): Numerical computations
- `requests` (2.32.3+): HTTP requests for LLM API and embedding server
- `python-dotenv` (1.1.0+): Environment management
- `scikit-learn` (1.3.0+): Additional ML utilities

### External Dependencies
- `llama.cpp`: Embedding server (included as submodule)
- `qwen3-embedding-0.6b-q8_0.gguf`: Embedding model file

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key (required) | - |
| `SEMANTIC_MODEL` | LLM model for analysis | `anthropic/claude-3-haiku:beta` |
| `OPENROUTER_APP_URL` | Application URL for API | `http://localhost` |
| `OPENROUTER_APP_TITLE` | Application title | `LLM_RAG_System` |
| `PAK_DEBUG` | Enable debug logging | `false` |

### Model Configuration

The system supports various embedding models through sentence-transformers:
- Qwen models for multilingual support
- BERT-based models for English
- Specialized domain models

## Performance

### Optimization Features
- **Batch processing**: Efficient embedding generation
- **Compressed storage**: NumPy `.npz` format for embeddings
- **Streaming support**: Large document handling
- **Caching**: Model loading optimization

### Benchmarks
- **Processing**: ~1-2 seconds per PDF page
- **Search**: <1 second for typical queries
- **Memory**: ~100MB base + model size

## Troubleshooting

### Common Issues

1. **Missing API Key**:
   ```
   Error: OPENROUTER_API_KEY not found
   Solution: Add API key to .env file
   ```

2. **Model Loading Failed**:
   ```
   Error: Model download failed
   Solution: Check internet connection, system will auto-fallback
   ```

3. **Collection Not Found**:
   ```
   Error: Collection not found: pdf_document_name
   Solution: Run ingest.py first to process documents
   ```

4. **Low Similarity Scores**:
   ```
   Issue: No relevant results
   Solution: Try broader queries or lower similarity threshold
   ```

### Debug Mode

Enable detailed logging:
```bash
export PAK_DEBUG=true
./ingest.py document.pdf -v
./query.py document "query" -v -t
```

## Development

### Adding New Features

1. **Custom Embedding Models**: Modify `load_embedding_model()` in both scripts
2. **Output Formats**: Extend `display_results()` in query.py
3. **Language Support**: Add language detection rules in `highlight_relevant_text()`

### Testing

```bash
# Test connections
python llm_wrapper.py
python check_server.py

# Process sample document
./ingest.py sample.pdf -v

# Test search functionality  
./query.py sample "test query" -t -v
```

### Setup Requirements

Before using the system, you need:

1. **Build llama.cpp**:
   ```bash
   cd llama.cpp
   mkdir build
   cd build
   cmake .. -DGGML_CUDA=ON  # For CUDA support, or use cpu-only
   make llama-server
   ```

2. **Download Qwen3 embedding model**:
   ```bash
   # Place the model file at:
   # llama.cpp/models/qwen3-embedding-0.6b-q8_0.gguf
   ```

3. **Start embedding server** (manual or automatic):
   ```bash
   # Manual start
   ./start_embedding_server.sh
   
   # Check status
   python check_server.py
   
   # Auto-start
   python check_server.py --start
   ```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

For issues and questions:
- Check the troubleshooting section
- Review error logs with debug mode enabled
- Verify environment configuration
- Test with smaller documents first