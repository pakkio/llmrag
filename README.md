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
   Copy the example environment file and configure it:
   ```bash
   cp .env.example .env
   # Edit .env with your actual API key and preferences
   ```
   
   Required configuration:
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
   - PDF text extraction using PyMuPDF (fitz)
   - Page-by-page text processing with progress tracking
   - Embedding generation via llama.cpp server API calls
   - Batch storage in ChromaDB with metadata (page numbers, document info)

2. **query.py**: Semantic search engine
   - ChromaDB cosine similarity search across collections
   - Multi-collection querying support
   - Language-aware content analysis with LLM integration
   - Visual result highlighting with color-coded explanations

3. **llm_wrapper.py**: Unified API wrapper
   - **LLM Integration**: OpenRouter API for semantic analysis and explanations
   - **Embedding Integration**: llama.cpp server communication via REST API
   - **Server Management**: Auto-start functionality with health checks
   - **Error Handling**: Comprehensive timeout and retry logic

4. **check_server.py**: Embedding server utility
   - Health check via embedding API test calls
   - Background server startup with progress monitoring
   - Port conflict detection and resolution guidance

5. **start_embedding_server.sh**: Server launcher script
   - Configures llama-server with embedding-specific parameters
   - Validates binary and model file existence
   - Sets optimal parameters: `--embedding --pooling last -ub 8192`

### Embedding Models

**Current Model**: Qwen3-Embedding-0.6B-Q8_0 (`qwen3-embedding-0.6b-q8_0.gguf`)
- **Architecture**: Small but efficient multilingual embedding model
- **Size**: ~600MB quantized to 8-bit (Q8_0)
- **Context**: Supports up to 8192 tokens with unlimited batch size (`-ub 8192`)
- **Pooling**: Last token pooling (`--pooling last`)
- **Special Token**: Requires `<|endoftext|>` suffix for optimal performance
- **Server**: Runs via llama.cpp server on port 8080
- **Auto-start**: Available via `check_server.py --start`

**Storage**: ChromaDB vector database
- Persistent storage in `./chroma_db/`
- Efficient cosine similarity search
- Metadata preservation with page numbers and document info

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
├── .env.example                # Environment configuration template
├── llama.cpp/                  # llama.cpp submodule for embedding server
│   ├── models/                 # Embedding model files
│   └── build/                  # Compiled binaries
├── .gitignore                  # Git ignore patterns
└── README.md                   # This file

# Generated files (ignored by git):
├── .env                        # Environment configuration (copy from .env.example)
└── chroma_db/                  # ChromaDB vector database storage
    ├── chroma.sqlite3          # Database file
    └── {collection_dirs}/      # Collection data
```

## Dependencies

### Core Libraries
- `pymupdf` (1.23.28+): PDF text extraction
- `chromadb` (0.4.22+): Vector database for embeddings  
- `numpy` (1.24.0+): Numerical computations and embedding operations
- `requests` (2.32.3+): HTTP requests for LLM API and embedding server
- `python-dotenv` (1.1.0+): Environment management
- `scikit-learn` (1.3.0+): Additional ML utilities and similarity calculations
- `tqdm` (4.66.0+): Progress bars for ingestion process
- `sentence-transformers` (2.2.2+): Fallback embedding models (optional)
- `torch` (2.1.0+): PyTorch backend for sentence-transformers

### External Dependencies
- **llama.cpp**: Embedding server (included as git submodule)
  - Provides `llama-server` binary for embedding generation
  - Built from source with support for various hardware accelerations
  - Located at `llama.cpp/build/bin/llama-server`
- **qwen3-embedding-0.6b-q8_0.gguf**: Primary embedding model
  - Pre-quantized 8-bit model (~600MB)
  - Located at `llama.cpp/models/qwen3-embedding-0.6b-q8_0.gguf`
  - Supports multilingual text embedding

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key (required) | - |
| `SEMANTIC_MODEL` | LLM model for analysis | `anthropic/claude-3-haiku:beta` |
| `OPENROUTER_APP_URL` | Application URL for API | `http://localhost` |
| `OPENROUTER_APP_TITLE` | Application title | `LLM_RAG_System` |
| `PAK_DEBUG` | Enable debug logging | `false` |

### Embedding Model Configuration

The system uses llama.cpp server for embeddings. To change models:

#### 1. Supported Model Types
- **Text Embedding Models**: Any GGUF model that supports `--embedding` flag
- **Popular Options**:
  - `qwen3-embedding-0.6b-q8_0.gguf` (current, multilingual)
  - `nomic-embed-text-v1.5.Q8_0.gguf` (English optimized)
  - `bge-large-en-v1.5.Q8_0.gguf` (Large English model)
  - `multilingual-e5-large.Q8_0.gguf` (Multilingual alternative)

#### 2. How to Change Embedding Models

**Step 1**: Download your desired GGUF embedding model
```bash
# Example: Download alternative model
wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf -P llama.cpp/models/
```

**Step 2**: Update the server startup script
```bash
# Edit start_embedding_server.sh
MODEL_PATH="$LLAMA_DIR/models/your-new-model.gguf"
```

**Step 3**: Adjust server parameters if needed
```bash
# For different context lengths or pooling methods:
exec ./bin/llama-server \
    -m "$MODEL_PATH" \
    --embedding \
    --pooling mean \          # or 'last', 'cls' depending on model
    -ub 4096 \               # adjust batch size for your model
    --port 8080
```

**Step 4**: Update code if different special tokens are needed
```python
# In llm_wrapper.py, modify the generate_embeddings function:
if not text.endswith('<|endoftext|>'):
    text = text + '<|endoftext|>'  # Change this for different models
```

#### 3. Model-Specific Considerations

| Model Family | Special Token | Pooling | Context Length | Notes |
|--------------|---------------|---------|----------------|--------|
| Qwen3 | `<|endoftext|>` | last | 8192 | Current default |
| Nomic-Embed | None required | mean | 8192 | English optimized |
| BGE | `[CLS]` | cls | 512 | BERT-based |
| E5 | None required | mean/last | 4096 | Multilingual |

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

2. **Embedding Server Not Running**:
   ```
   Error: Connection refused to embedding server
   Solution: Start server with ./start_embedding_server.sh or check_server.py --start
   ```

3. **llama-server Binary Missing**:
   ```
   Error: llama-server binary not found
   Solution: Build llama.cpp: cd llama.cpp/build && make llama-server
   ```

4. **Model File Missing**:
   ```
   Error: Model not found at llama.cpp/models/qwen3-embedding-0.6b-q8_0.gguf
   Solution: Download model from HuggingFace or check file path
   ```

5. **Collection Not Found**:
   ```
   Error: Collection not found: pdf_document_name
   Solution: Run ingest.py first to process documents
   ```

6. **Low Similarity Scores**:
   ```
   Issue: No relevant results
   Solution: Try broader queries or lower similarity threshold (-s 0.1)
   ```

7. **Server Port Already in Use**:
   ```
   Error: Port 8080 already in use
   Solution: Kill existing server: pkill -f llama-server, or change port in scripts
   ```

8. **CUDA/GPU Issues**:
   ```
   Error: CUDA initialization failed
   Solution: Rebuild llama.cpp without CUDA: cmake .. && make llama-server
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

1. **Build llama.cpp server**:
   ```bash
   cd llama.cpp
   
   # Create build directory
   mkdir -p build
   cd build
   
   # Configure build (choose one option):
   # For CPU-only:
   cmake ..
   
   # For CUDA support (if you have NVIDIA GPU):
   cmake .. -DGGML_CUDA=ON
   
   # For Metal support (macOS):
   cmake .. -DGGML_METAL=ON
   
   # Build the server
   make llama-server
   
   # Verify build
   ls -la bin/llama-server  # Should show the executable
   ```

2. **Download Qwen3 embedding model** (if not already present):
   ```bash
   cd llama.cpp/models
   
   # The model should already be present as qwen3-embedding-0.6b-q8_0.gguf
   # If missing, download from HuggingFace:
   wget https://huggingface.co/Qwen/Qwen3-0.6B-Embedding-GGUF/resolve/main/qwen3-embedding-0.6b-q8_0.gguf
   ```

3. **Verify server functionality**:
   ```bash
   # Check if server binary exists and model is present
   python check_server.py
   
   # Start server manually for testing
   ./start_embedding_server.sh
   
   # Or auto-start server
   python check_server.py --start
   ```

4. **Test the complete setup**:
   ```bash
   # Test LLM connection
   python llm_wrapper.py
   
   # Test with a sample document
   ./ingest.py sample.pdf -v
   ./query.py sample "test query" -t -v
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