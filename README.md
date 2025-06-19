# LLM RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system that processes PDF documents and provides advanced search capabilities through hybrid semantic-keyword search, intelligent LLM reranking, adaptive query enhancement, and real-time confidence scoring.

## Features

### **🔍 Advanced Search System**
- **Hybrid Search**: Intelligent combination of semantic + keyword search (default: 60%/40%)
- **Semantic Search**: OpenAI embeddings using text-embedding-3-large for conceptual understanding
- **Keyword Search**: SQLite FTS5 with BM25-style ranking for exact term matching
- **Configurable Weights**: Customize semantic/keyword balance for optimal results
- **LLM Reranking**: Intelligent result reordering using Gemini 2.0 Flash Lite (~2s processing)
- **Adaptive Query Enhancement**: AI-powered query classification with automatic enhancement calibration
- **Confidence Scoring**: Real-time quality assessment with visual indicators (🟢 HIGH, 🟡 MEDIUM, 🔴 LOW)

### **🌐 Multilingual Intelligence**
- **Auto-Detection**: Intelligent language identification from document content
- **Manual Override**: Force specific language responses (Italian, Spanish, French, English)
- **Native Explanations**: Contextual analysis in the document's natural language

### **💻 Dual Interface Options**
| Feature | CLI (query.py) | Web (gradio_browser.py) |
|---------|----------------|------------------------|
| Search Methods | ✅ `--hybrid`, `--semantic`, `--bm25` | ✅ Interactive dropdown |
| Weight Control | ✅ `--semantic-weight`, `--keyword-weight` | ✅ Auto-normalizing sliders |
| LLM Reranking | ✅ `--rerank` flag | ✅ Reranking checkbox |
| Language Control | ✅ `--language italian` | ✅ Language dropdown |
| Real-time Feedback | ✅ Verbose logging | ✅ Live progress + logs |
| Visual Highlighting | ✅ ANSI colors | ✅ Rich HTML styling |

### **📊 Rich Result Analysis**
- **Smart Highlighting**: AI-powered semantic text highlighting with explanations
- **Multi-Level Analysis**: Content → Relevance → LLM Analysis → Synthesis
- **Dual Answer Mode**: Direct answers + detailed search breakdowns
- **Professional Styling**: Color-coded sections with responsive design

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
- Semantic embeddings stored in ChromaDB collection: `pdf_{pdf_name}`
- Keyword index stored in SQLite FTS5 database: `./hybrid_search.db`
- Vector database location: `./chroma_db/`

### 2. Querying Documents

Search processed documents using hybrid, semantic, or keyword search:

```bash
python query.py "your search query"
```

**Search Mode Options**:
- `--hybrid`: Hybrid search combining semantic + keyword (default, 60% semantic + 40% keyword)
- `--semantic`: Semantic search only (ChromaDB embeddings)
- `--bm25`: Keyword search only (SQLite FTS5 BM25)
- `--semantic-weight FLOAT`: Weight for semantic search in hybrid mode (default: 0.6)
- `--keyword-weight FLOAT`: Weight for keyword search in hybrid mode (default: 0.4)

**General Options**:
- `--pdf PDF_NAME`: Search specific PDF (default: all)
- `-k, --top-k N`: Number of results to show (default: 3)
- `-s, --min-similarity FLOAT`: Minimum similarity threshold (default: 0.0)
- `--language LANG`: Force response language (italian, spanish, french, english, default: auto-detect)
- `--enhancement MODE`: Enhancement mode (auto, minimal, full, maximum, off, default: auto)
- `--rerank`: Enable LLM reranking for improved result quality (~2s)
- `--no-enhancement`: Legacy flag to disable enhancement (equivalent to --enhancement=off)
- `--no-text`: Hide text content
- `--no-analysis`: Disable LLM analysis
- `--list`: List available PDF collections
- `-v, --verbose`: Enable verbose logging

**Examples**:
```bash
# Default hybrid search (60% semantic + 40% keyword)
python query.py "pricing strategies"

# Semantic search only (concepts and context)
python query.py "market analysis" --semantic

# Keyword search only (exact terms and phrases)
python query.py "machine learning" --bm25

# Custom hybrid weighting (80% semantic, 20% keyword)
python query.py "artificial intelligence" --semantic-weight 0.8 --keyword-weight 0.2

# Search specific document with 5 results
python query.py "competitive advantage" --pdf business_plan -k 5

# Filter by similarity threshold
python query.py "pricing models" -s 0.3

# List available documents
python query.py --list

# Keyword search for exact terminology
python query.py "REST API endpoint" --bm25

# LLM reranking for improved result quality (~2s processing time)
python query.py "machine learning algorithms" --rerank
python query.py "market analysis" --semantic --rerank
python query.py "competitive strategies" --hybrid --rerank

# Confidence scoring examples (visual quality assessment)
python query.py "Neptune distance from sun" --dual-answer  # → 97% HIGH confidence
python query.py "string theory cosmology" --dual-answer    # → 62% MEDIUM confidence  
python query.py "pasta recipe" --dual-answer               # → No results (outside domain)

# Semantic search for conceptual understanding
python query.py "What can we learn from David's relationship with God?" --semantic

# Force Italian language responses
python query.py "strategie di marketing" --language italian

# Force English responses for multilingual documents
python query.py "análisis de mercado" --language english

# Adaptive query enhancement (automatic classification and calibration)
python query.py "Quanto è grande il Sole?" --enhancement=auto      # Factual → minimal enhancement
python query.py "Cos'è una supernova?" --enhancement=auto          # Conceptual → full enhancement  
python query.py "Differenza tra pianeta e stella" --enhancement=auto # Comparative → maximum enhancement

# Manual enhancement control
python query.py "search query" --enhancement=minimal   # Factual optimization
python query.py "search query" --enhancement=full      # Balanced (previous default)
python query.py "search query" --enhancement=maximum   # Comparative optimization
python query.py "search query" --enhancement=off       # Disable enhancement

# Legacy query enhancement examples (automatic translation and expansion)
python query.py "Nettuno" --bm25  # → Enhanced to "Neptune planet eighth planet"
python query.py "strategie di marketing" --hybrid  # → Enhanced to include "marketing strategies business promotional"

# Disable query enhancement for exact term matching (legacy)
python query.py "machine learning" --bm25 --no-enhancement
```

### 3. Web Interface

Launch the Gradio web interface:

```bash
python gradio_browser.py
```

Access at: http://localhost:7860

**Port Management**: If port 7860 is occupied, use the utility script:
```bash
./kill_port_7860.sh
```

**Enhanced Web Interface Features**:

### **🔧 Search Configuration**
- **🔍 Search Method Selection**: Dropdown for Hybrid/Semantic/BM25 search modes
- **⚖️ Hybrid Weight Control**: Interactive sliders for semantic/keyword balance (auto-normalizing to 1.0)
- **🔄 LLM Reranking**: Checkbox to enable intelligent result reordering (~2s)
- **🌐 Language Selection**: Dropdown for Auto-detect/English/Italian/Spanish/French
- **📚 Collection Filtering**: Text input for specific document collections
- **🎯 Smart UI**: Sliders only visible when Hybrid mode is selected

### **📊 Advanced Search Features**
- **Interactive Search Interface**: Real-time document search with full CLI feature parity
- **Confidence Scoring**: Visual quality indicators (🟢 HIGH ≥80%, 🟡 MEDIUM 60-79%, 🔴 LOW <60%)
- **Rich Text Highlighting**: Advanced semantic highlighting with footnoted explanations
- **Multi-Level Analysis**: Each result includes:
  - 📖 **Content**: Highlighted text with semantic annotations
  - 💡 **Relevance Analysis**: Numbered explanations for each highlight
  - 🧠 **LLM Analysis**: AI-powered relevance assessment
  - 🔬 **Comprehensive Synthesis**: Cross-result analysis and insights
- **Dual Answer Mode**: Direct answers + detailed search result analysis with confidence assessment
- **Responsive Design**: Dark/light theme support with professional styling
- **Live Feedback**: Real-time logs showing search method, weights, and progress

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
   - **Search Method Control**: Full dropdown support for Hybrid/Semantic/BM25 modes
   - **Interactive Weight Tuning**: Real-time sliders for hybrid search balance (auto-normalizing)
   - **LLM Reranking Control**: Checkbox to enable intelligent result reordering
   - **Language Selection**: Dropdown for forced language responses
   - **Rich semantic highlighting**: Footnoted explanations with multi-level analysis
   - **Professional UI**: Smart controls with conditional visibility and live feedback
   - **Full CLI Parity**: All command-line features available in web interface

5. **info.py**: Database utilities
   - Collection statistics
   - Database management tools

### Technology Stack

- **Semantic Search**: OpenAI text-embedding-3-large (3072 dimensions) with ChromaDB
- **Keyword Search**: SQLite FTS5 with BM25 ranking and Porter stemming
- **Hybrid Search**: Intelligent combination with configurable weighting
- **LLM Analysis**: OpenAI GPT-4.1 Nano via OpenRouter (configurable)
- **LLM Reranking**: Google Gemini 2.0 Flash Lite for intelligent result ordering
- **PDF Processing**: PyMuPDF for text extraction
- **Web Interface**: Gradio for interactive search
- **Text Processing**: Advanced UTF-8 handling and normalization

## File Structure

```
llmrag/
├── ingest.py                    # PDF processing and dual-database ingestion
├── query.py                     # Hybrid/semantic/keyword search engine
├── sqlite_fts5.py              # SQLite FTS5 keyword search manager
├── llm_wrapper.py              # API integration (OpenAI + OpenRouter)
├── gradio_browser.py           # Web interface for document browsing
├── info.py                     # Database information and utilities
├── test_llm_wrapper.py         # API connection tests
├── test_chunking.py            # Text chunking tests
├── pyproject.toml              # Project dependencies
├── .env.example                # Environment configuration template
├── .env                        # Environment configuration (create from .env.example)
├── kill_port_7860.sh           # Utility to free port 7860 (kill processes)
├── chroma_db/                  # ChromaDB vector database (auto-created)
└── hybrid_search.db            # SQLite FTS5 keyword database (auto-created)
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

**Automatic Detection + Manual Override**:
- **Auto-Detection**: Intelligent language identification from document content
- **Manual Override**: Force specific language via `--language` (CLI) or dropdown (Web)

**Supported Languages**:
- **Italian**: Spiegazioni in italiano naturale
- **Spanish**: Explicaciones en español natural  
- **French**: Explications en français naturel
- **English**: Natural English explanations

**Usage Examples**:
```bash
# Auto-detect language from document
python query.py "strategia aziendale"  # → Italian responses

# Force specific language
python query.py "business strategy" --language italian  # → Italian responses
python query.py "estrategia empresarial" --language english  # → English responses
```

### Adaptive Query Enhancement

**Intelligent Query Classification + Automatic Enhancement Calibration**:
The system now automatically classifies queries and adapts enhancement levels for optimal results:

**Query Classification Types**:
1. **Factual Queries**: "quanto", "quando", "dove", "how big", "when did" → **Minimal Enhancement**
   - Example: "Quanto è grande il Sole?" → "How big Sun? size" 
   - Strategy: Translation + 1-2 direct synonyms only
   
2. **Conceptual Queries**: "cos'è", "come", "perché", "what is", "how does" → **Full Enhancement**
   - Example: "Cos'è una supernova?" → "What supernova? stellar explosion massive star core collapse"
   - Strategy: Balanced expansion with relevant terminology
   
3. **Comparative Queries**: "differenza", "confronto", "versus", "compare" → **Maximum Enhancement**
   - Example: "Differenza tra pianeta e stella" → "Difference planet star? celestial bodies stellar objects formation"
   - Strategy: Extensive synonyms + related concepts + domain terminology

**Enhancement Modes**:
- **`--enhancement=auto`** (default): Automatic classification and calibration
- **`--enhancement=minimal`**: Factual queries optimization  
- **`--enhancement=full`**: Balanced enhancement (previous default)
- **`--enhancement=maximum`**: Comparative queries optimization
- **`--enhancement=off`**: Disable enhancement completely

**Adaptive Process**:
1. **Query Classification**: AI-powered analysis of query type and intent
2. **Adaptive Calibration**: Enhancement level adjusted based on query characteristics
3. **Language Detection**: Identifies non-English queries
4. **Translation**: Converts to English for document matching
5. **Smart Expansion**: Term expansion calibrated to query type
6. **Context Addition**: Includes domain-specific terminology as appropriate
7. **Fallback Protection**: Uses original if enhancement fails

**Benefits**:
- **Intelligent Trade-offs**: Right-sized enhancement reduces noise for factual queries while maximizing recall for complex queries
- **Cross-Language Search**: Find English documents using non-English queries
- **Adaptive Efficiency**: Optimal enhancement level automatically selected
- **Better Precision**: LLM-guided term selection preserves relevance
- **Domain Awareness**: Adds field-specific terminology when appropriate
- **Backward Compatibility**: Legacy `--no-enhancement` flag continues to work

**Control Options**:
- **Default**: `--enhancement=auto` for intelligent adaptation
- **Manual Override**: Use specific modes when you know the query type
- **Legacy Support**: `--no-enhancement` still works for exact term matching

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
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM calls and reranking | Yes |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | Yes |
| `SEMANTIC_MODEL` | LLM model for analysis | No (defaults to GPT-4.1 Nano) |
| `RERANKING_MODEL` | Model for intelligent result reranking | No (defaults to Gemini 2.0 Flash Lite) |
| `EMBEDDING_MODEL` | Embedding model | No (defaults to text-embedding-3-large) |

### Recommended Models

For `SEMANTIC_MODEL`:
- `openai/gpt-4.1-nano` (current default, fast & cost-effective)
- `anthropic/claude-3-haiku:beta` (excellent alternative)
- `anthropic/claude-3-sonnet:beta` (balanced performance for complex analysis)

For `RERANKING_MODEL`:
- `google/gemini-2.0-flash-lite-001` (current default, optimized for reranking)
- `anthropic/claude-3-haiku:beta` (alternative for reranking)

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