# LLM RAG System

A powerful Retrieval-Augmented Generation (RAG) system that processes PDF documents, generates semantic embeddings, and enables intelligent querying with highlighted results and relevance analysis.

## Features

- **PDF Text Extraction**: Extract text content from PDF documents with smart chunking
- **OpenAI Embeddings**: High-quality embeddings using text-embedding-3-large
- **Intelligent Querying**: Search documents using semantic similarity
- **Multi-language Support**: Automatic language detection with native language explanations
- **Visual Results**: Highlighted relevant text with color-coded explanations
- **Web Interface**: Gradio-based browser for interactive document search

## Installation

### Prerequisites

- Python 3.8+
- Poetry (recommended) or pip

### Quick Setup

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
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```
   
   Required configuration:
   ```env
   # OpenRouter API key for LLM analysis and explanations
   # Get your key from: https://openrouter.ai/keys
   OPENROUTER_API_KEY=your_openrouter_api_key_here

   # OpenAI API key for embeddings (text-embedding-3-large)
   # Get your key from: https://platform.openai.com/api-keys
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### 1. Document Ingestion

Process a PDF document to extract text and generate embeddings:

```bash
python ingest.py path/to/your/document.pdf
```

**Options**:
- `-v, --verbose`: Enable verbose logging
- `-p, --pages N`: Process only first N pages
- `--from-page N`: Start from page N
- `--to-page N`: End at page N

**Output**:
- Text and embeddings stored in ChromaDB collection: `pdf_{pdf_name}`
- Database location: `./chroma_db/`

### 2. Querying Documents

Search processed documents using semantic queries:

```bash
python query.py "your search query"
```

**Options**:
- `--pdf PDF_NAME`: Search specific PDF (default: all)
- `-k, --top-k N`: Number of results to show (default: 3)
- `-s, --min-similarity FLOAT`: Minimum similarity threshold (default: 0.0)
- `--no-text`: Hide text content
- `--list`: List available PDF collections
- `-v, --verbose`: Enable verbose logging

**Examples**:
```bash
# Basic search across all documents
python query.py "pricing strategies"

# Search specific document with 5 results
python query.py "market analysis" --pdf business_plan -k 5

# Filter by similarity threshold
python query.py "competitive advantage" -s 0.3

# List available documents
python query.py --list

# Complex theological query
python query.py "What can we learn from David's relationship with God?"
```

### 3. Web Interface

Launch the Gradio web interface:

```bash
python gradio_browser.py
```

Access at: http://localhost:7860

**Enhanced Features**:
- **Interactive Search Interface**: Real-time document search with collection filtering
- **Rich Text Highlighting**: Advanced semantic highlighting with footnoted explanations
- **Multi-Level Analysis**: Each result includes:
  - 📖 **Content**: Highlighted text with semantic annotations
  - 💡 **Relevance Analysis**: Numbered explanations for each highlight
  - 🧠 **LLM Analysis**: AI-powered relevance assessment
  - 🔬 **Comprehensive Synthesis**: Cross-result analysis and insights
- **Dual Answer Mode**: Direct answers + detailed search result analysis
- **Responsive Design**: Dark/light theme support with professional styling
- **Multilingual Support**: Native language explanations (Italian, Spanish, French, English)

### 4. Database Information

View database statistics:

```bash
python info.py
```

## Architecture

### Core Components

1. **ingest.py**: Document processing pipeline
   - PDF text extraction using PyMuPDF
   - Smart text chunking (500 chars with 50 char overlap)
   - OpenAI embedding generation
   - ChromaDB storage with metadata

2. **query.py**: Semantic search engine
   - ChromaDB similarity search
   - Multi-collection querying
   - LLM-powered result highlighting
   - Multilingual explanations

3. **llm_wrapper.py**: API integration layer
   - OpenRouter API for LLM analysis
   - OpenAI API for embeddings
   - Comprehensive error handling
   - Text preprocessing and normalization

4. **gradio_browser.py**: Enhanced web interface
   - Interactive document search with real-time results
   - Rich semantic highlighting with footnoted explanations
   - Multi-level analysis sections (Content, Relevance, LLM Analysis, Synthesis)
   - Professional CSS styling with dark/light theme support
   - Comprehensive result display matching terminal functionality

5. **info.py**: Database utilities
   - Collection statistics
   - Database management tools

### Technology Stack

- **Embeddings**: OpenAI text-embedding-3-large (3072 dimensions)
- **Vector Database**: ChromaDB with persistent storage
- **LLM**: Configurable via OpenRouter (default: Claude 3 Haiku)
- **PDF Processing**: PyMuPDF for text extraction
- **Web Interface**: Gradio for interactive search
- **Text Processing**: Advanced UTF-8 handling and normalization

## File Structure

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
├── .env                        # Environment configuration (create from .env.example)
└── chroma_db/                  # ChromaDB vector database (auto-created)
```

## Features in Detail

### Smart Text Chunking

- **Chunk Size**: 500 characters with 50 character overlap
- **Word Boundaries**: Preserves word integrity
- **Context Preservation**: Overlap maintains semantic continuity
- **Metadata**: Tracks chunk relationships and source pages

### Advanced Highlighting

The system provides sophisticated text highlighting with explanations:

**Terminal (query.py)**:
- **Yellow Background**: Semantically relevant text sections
- **Green Text**: Detailed explanations of relevance
- **Structured Sections**: Bordered content areas with emoji headers
- **Footnoted Explanations**: Numbered references with detailed analysis

**Web Interface (gradio_browser.py)**:
- **Yellow Highlights**: Semantically relevant text with footnote numbers
- **Explanation Cards**: Numbered explanations in dedicated sections
- **Multi-Level Analysis**: Content → Relevance → LLM Analysis → Synthesis
- **Professional Styling**: Color-coded sections with responsive design

### Multilingual Support

Automatic language detection with native explanations:
- **Italian**: Spiegazioni in italiano naturale
- **Spanish**: Explicaciones en español natural  
- **French**: Explications en français naturel
- **English**: Natural English explanations

### Similarity Scoring

- **Cosine Similarity**: Precise semantic matching
- **Normalized Embeddings**: Consistent similarity ranges
- **Threshold Filtering**: Configurable relevance cutoffs
- **Ranked Results**: Best matches first with confidence scores

## Testing

Test your setup:

```bash
# Test API connections
python llm_wrapper.py

# Test with sample document
python ingest.py sample.pdf -v
python query.py "test query" -v

# Run test suites
python test_llm_wrapper.py
python test_chunking.py
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM calls | Yes |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | Yes |
| `SEMANTIC_MODEL` | LLM model for analysis | No (defaults to Claude 3 Haiku) |

### Recommended Models

For `SEMANTIC_MODEL`:
- `anthropic/claude-3-haiku:beta` (default, fast & cost-effective)
- `google/gemini-flash-2.0` (very fast, good analysis)
- `anthropic/claude-3-sonnet:beta` (balanced performance)

## Performance

### Optimization Features
- **Batch Processing**: Efficient embedding generation
- **Persistent Storage**: ChromaDB for fast retrieval
- **Smart Chunking**: Optimal text segmentation
- **Caching**: Model loading optimization

### Benchmarks
- **Processing**: ~2-3 seconds per PDF page
- **Search**: <1 second for typical queries
- **Memory**: ~200MB base + embedding cache
- **Storage**: ~1MB per document (embeddings + metadata)

## Troubleshooting

### Common Issues

1. **Missing API Keys**:
   ```
   Error: OPENAI_API_KEY not found
   Solution: Add API keys to .env file
   ```

2. **Collection Not Found**:
   ```
   Error: No PDF collections found
   Solution: Run ingest.py first to process documents
   ```

3. **Low Similarity Scores**:
   ```
   Issue: No relevant results
   Solution: Try broader queries or lower threshold (-s 0.1)
   ```

4. **Empty Results**:
   ```
   Issue: Query returns no results
   Solution: Check if documents are properly ingested with --list
   ```

### Debug Mode

Enable detailed logging:
```bash
# Set debug in .env
PAK_DEBUG=true

# Or export temporarily
export PAK_DEBUG=true
python ingest.py document.pdf -v
```

## Development

### Adding New Features

1. **Custom Models**: Modify model selection in llm_wrapper.py
2. **Output Formats**: Extend display_results() in query.py
3. **Language Support**: Add language rules in highlight_relevant_text()
4. **Chunking Strategies**: Modify chunk_text() in ingest.py

### API Integration

The system uses two APIs:
- **OpenAI**: For high-quality text embeddings
- **OpenRouter**: For LLM analysis and explanations

Both APIs are abstracted through llm_wrapper.py for easy modification.

## Examples

### Complex Semantic Queries

The system excels at understanding conceptual queries:

**Query**: "What can we learn from David's relationship with God?"

**Results Retrieved**:
1. **David vs Goliath** - Faith enabling impossible victories
2. **David and Bathsheba** - Consequences of moral failure
3. **Saul's Relationship** - Contrast in divine favor

**Generated Analysis**: Multi-colored highlighting showing:
- PRIMARY: Direct references to David and God
- SECONDARY: Concepts of faith, trust, and divine relationship
- CONTEXT: Historical and theological background

### Business Document Analysis

**Query**: "competitive pricing strategies"

**Results**: Semantic matching finds relevant content even without exact phrase matches:
- Market positioning discussions
- Pricing model comparisons
- Competitive analysis frameworks

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

For issues and questions:
- Check the troubleshooting section
- Review error logs with debug mode enabled
- Verify API key configuration
- Test with smaller documents first