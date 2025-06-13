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

# Theological/conceptual queries (works well with religious texts)
./query.py The_Bible_Book "What can we learn from David's relationship with God?"
```

### Example: Semantic Search in Action

The system excels at understanding complex, conceptual queries. Here's a real example:

**Query**: "What can we learn from David's relationship with God?"

**Results Retrieved**:
1. **David vs Goliath** (similarity: 0.50) - Shows David's faith enabling impossible victories
2. **David and Bathsheba** (similarity: 0.52) - Demonstrates even God's chosen can sin but character matters
3. **Saul's Fall from Favor** (similarity: 0.54) - Illustrates God's favor, mercy, and absolute judgment

**Generated Answer** (after LLM processing):
> *David's relationship with God teaches us that God chooses based on character not appearances, that unwavering faith enables impossible victories, that even the chosen can fall but repentance matters, and that divine mercy and judgment work together. His story shows the importance of trusting God completely while remaining humble despite success.*

The semantic search successfully connected a theological question to relevant biblical passages across different stories, enabling a comprehensive answer about spiritual lessons.

### 3. Testing LLM Connection

Verify your LLM configuration:

```bash
python llm_wrapper.py
```

## Architecture

### Core Components

1. **ingest.py**: Document processing pipeline
   - PDF text extraction using PyMuPDF (fitz)
   - Enhanced UTF-8 text processing with error handling
   - Page-by-page text processing with progress tracking
   - Embedding generation via direct llama-embedding binary execution
   - Batch storage in ChromaDB with metadata (page numbers, document info)

2. **query.py**: Semantic search engine
   - ChromaDB cosine similarity search across collections
   - Multi-collection querying support
   - Language-aware content analysis with LLM integration
   - Visual result highlighting with color-coded explanations

3. **llm_wrapper.py**: Unified API wrapper
   - **LLM Integration**: OpenRouter API for semantic analysis and explanations
   - **Embedding Integration**: Direct llama-embedding binary execution
   - **Text Processing**: Advanced UTF-8 handling and text normalization
   - **Error Handling**: Comprehensive timeout and retry logic with binary process management

4. **gradio_browser.py**: Web interface for document browsing
   - Gradio-based web UI for interactive document search
   - User-friendly interface for querying processed documents

### Embedding Models

**Current Model**: Qwen3-Embedding-0.6B-Q8_0 (`qwen3-embedding-0.6b-q8_0.gguf`)
- **Architecture**: Small but efficient multilingual embedding model
- **Size**: ~600MB quantized to 8-bit (Q8_0)
- **Context**: Supports up to 4096 tokens with batch size 1
- **Pooling**: Mean pooling
- **Execution**: Direct binary execution via llama-embedding
- **Text Processing**: Advanced UTF-8 cleaning and normalization
- **Performance**: 120-second timeout with comprehensive error handling

**Storage**: ChromaDB vector database
- Persistent storage in `./chroma_db/`
- Efficient cosine similarity search
- Metadata preservation with page numbers and document info
- Auto-generated collection names: `pdf_{document_name}`

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
├── llm_wrapper.py              # LLM API integration and direct embedding binary execution
├── gradio_browser.py           # Gradio web interface for document browsing
├── info.py                     # Database information and utility functions
├── pyproject.toml              # Project dependencies and metadata
├── .env.example                # Environment configuration template
├── llama.cpp/                  # llama.cpp repository for embedding generation
│   ├── models/                 # Embedding model files (qwen3-embedding-0.6b-q8_0.gguf)
│   └── build/bin/              # Compiled binaries (llama-embedding)
├── .gitignore                  # Git ignore patterns
└── README.md                   # This file

# Generated files (ignored by git):
├── .env                        # Environment configuration (copy from .env.example)
└── chroma_db/                  # ChromaDB vector database storage
    ├── chroma.sqlite3          # Database file
    └── {collection_dirs}/      # Collection data and embeddings
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
- **llama.cpp**: Embedding binary execution (included as git submodule)
  - Provides `llama-embedding` binary for direct embedding generation
  - Built from source with support for various hardware accelerations
  - Located at `llama.cpp/build/bin/llama-embedding`
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

**Step 2**: Update the model path in llm_wrapper.py
```python
# Edit llm_wrapper.py, modify model_path variable:
model_path = Path(__file__).parent / "llama.cpp/models/your-new-model.gguf"
```

**Step 3**: Adjust binary parameters if needed
```python
# In llm_wrapper.py, modify the cmd array:
cmd = [
    str(embedding_binary),
    "-m", str(model_path),
    "--pooling", "mean",      # or 'last', 'cls' depending on model
    "-c", "4096",            # adjust context length for your model
    "-b", "1",               # batch size
    "-p", cleaned_text
]
```

#### 3. Model-Specific Considerations

| Model Family | Special Processing | Pooling | Context Length | Notes |
|--------------|-------------------|---------|----------------|--------|
| Qwen3 | UTF-8 cleaning | mean | 4096 | Current default |
| Nomic-Embed | None required | mean | 8192 | English optimized |
| BGE | Text normalization | cls | 512 | BERT-based |
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

2. **Embedding Binary Missing**:
   ```
   Error: llama-embedding binary not found
   Solution: Build llama.cpp: cd llama.cpp/build && make llama-embedding
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

7. **Embedding Generation Timeout**:
   ```
   Error: Embedding generation timed out
   Solution: Reduce text length or increase timeout in llm_wrapper.py
   ```

8. **Text Processing Issues**:
   ```
   Error: Text is empty after cleaning
   Solution: Check document encoding or try different PDF extraction settings
   ```

9. **CUDA/GPU Issues**:
   ```
   Error: CUDA initialization failed
   Solution: Rebuild llama.cpp without CUDA: cmake .. && make llama-embedding
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

1. **Build llama.cpp embedding binary**:
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
   
   # Build the embedding binary
   make llama-embedding
   
   # Verify build
   ls -la bin/llama-embedding  # Should show the executable
   ```

2. **Download Qwen3 embedding model** (if not already present):
   ```bash
   cd llama.cpp/models
   
   # The model should already be present as qwen3-embedding-0.6b-q8_0.gguf
   # If missing, download from HuggingFace:
   wget https://huggingface.co/Qwen/Qwen3-0.6B-Embedding-GGUF/resolve/main/qwen3-embedding-0.6b-q8_0.gguf
   ```

3. **Verify embedding functionality**:
   ```bash
   # Test embedding generation directly
   python llm_wrapper.py
   
   # The test should show successful embedding generation
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