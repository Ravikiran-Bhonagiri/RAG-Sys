# Production RAG System

A production-grade, multi-provider Retrieval Augmented Generation (RAG) system with comprehensive multi-modal capabilities.

## Overview

This system implements a sophisticated RAG pipeline that combines text and visual information retrieval with state-of-the-art language models. It supports multiple LLM providers and includes comprehensive evaluation frameworks.

### Key Features

*   **Multi-Provider Architecture**: Seamless integration with OpenAI, Anthropic Claude, Google Gemini, DeepSeek, and Ollama
*   **Multi-Modal Processing**: Handles both text and images from PDF documents
*   **Visual RAG**: Automated image captioning enables text-based retrieval of visual concepts
*   **Hybrid Retrieval**: Vector similarity search with metadata filtering
*   **Production-Ready**: Comprehensive testing, evaluation, and monitoring capabilities
*   **Cost Tracking**: Built-in token usage and cost estimation across providers

## Architecture

The system follows a modular pipeline architecture:

```
PDF Documents → Ingestion → Vector Storage → Retrieval → Generation → Answer
                    ↓            ↓              ↓           ↓
                 Text/Images  ChromaDB    Hybrid Search  Multi-LLM
```

For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Project Structure

```
production_rag/
├── config.py              # Central configuration
├── main.py               # CLI entry point
├── ingestion/            # Document processing
│   ├── extractors.py    # PDF text/image extraction
│   ├── processors.py    # Chunking and captioning
│   └── storage.py       # Vector database operations
├── retrieval/           # Search and ranking
│   └── search.py       # Hybrid retrieval implementation
├── generation/          # Answer generation
│   └── rag.py          # Multi-provider LLM clients
└── tests/              # Testing and evaluation
    ├── test_*.py       # Unit tests
    ├── benchmark_diverse_questions.py
    └── ragas_benchmark.py
```

## Installation

### Prerequisites

*   Python 3.10 or higher
*   pip package manager
*   Virtual environment (recommended)

### Setup Steps

1. **Create Virtual Environment**:
   ```bash
   python -m venv .venv
   ```

2. **Activate Environment**:
   ```bash
   # Windows
   .venv\Scripts\Activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   cd production_rag
   pip install -r requirements.txt
   ```

4. **Configure Environment**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   GOOGLE_API_KEY=your_key_here
   DEEPSEEK_API_KEY=your_key_here
   ```

## Usage

All commands should be run from the repository root directory.

### Document Ingestion

Process PDF documents into the vector database:

```bash
python -m production_rag.main ingest [OPTIONS]
```

**Options**:
*   `--reset`: Clear existing database before ingestion
*   `--pdf PATH`: Process specific PDF file

**Example**:
```bash
python -m production_rag.main ingest --reset
```

This will:
1. Extract text and images from PDFs in `basic_rag/data/`
2. Generate captions for images using the configured VLM
3. Chunk text into semantic units
4. Create embeddings and store in ChromaDB

### Querying

Ask questions to the RAG system:

```bash
python -m production_rag.main query "Your question here"
```

**Example**:
```bash
python -m production_rag.main query "What is the Mastra framework?"
```

The system will:
1. Retrieve relevant context from the vector store
2. Assemble context with metadata
3. Generate an answer using the configured LLM
4. Display the answer with source citations and cost metrics

## Configuration

### Provider Selection

Edit `config.py` to change the default provider:

```python
LLM_PROVIDER = "openai"  # Options: openai, anthropic, google, deepseek, ollama
```

Or set via environment variable:
```bash
export LLM_PROVIDER=deepseek
```

### Model Configuration

Each provider has configurable models in `config.py`:

```python
MODELS = {
    "openai": {
        "text": "gpt-4o-mini",
        "vision": "gpt-4o-mini",
        "embedding": "text-embedding-3-small"
    },
    # ... other providers
}
```

### Retrieval Parameters

Adjust retrieval behavior in `config.py`:

```python
TOP_K_RETRIEVAL = 10  # Initial retrieval count
TOP_K_FINAL = 5       # Final context size
CHUNK_SIZE = 1000     # Text chunk size
CHUNK_OVERLAP = 200   # Overlap between chunks
```

## Testing and Evaluation

See [TESTING.md](TESTING.md) for comprehensive testing documentation including:
*   Unit tests
*   Integration benchmarks
*   RAGAS evaluation framework
*   Performance metrics

## Documentation

*   [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and data flow
*   [TESTING.md](TESTING.md) - Testing and evaluation guide
*   [CONFIG.md](CONFIG.md) - Configuration reference

## Performance

Typical performance metrics:
*   **Ingestion**: ~2-5 seconds per PDF page (with image captioning)
*   **Query**: 1-3 seconds end-to-end (depending on provider)
*   **Cost**: $0.001-$0.01 per query (varies by provider and model)

## License

This project is for educational and research purposes.
