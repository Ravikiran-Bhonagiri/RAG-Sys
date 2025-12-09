# Configuration Reference

Complete reference for configuring the Production RAG system.

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_PROVIDER` | Default LLM provider | `openai`, `anthropic`, `google`, `deepseek`, `ollama` |

### API Keys

| Variable | Provider | Required When |
|----------|----------|---------------|
| `OPENAI_API_KEY` | OpenAI | Using OpenAI models or embeddings |
| `ANTHROPIC_API_KEY` | Anthropic | Using Claude models |
| `GOOGLE_API_KEY` | Google | Using Gemini models |
| `DEEPSEEK_API_KEY` | DeepSeek | Using DeepSeek models |

## Model Configuration

### Supported Models by Provider

#### OpenAI
```python
"openai": {
    "text": "gpt-4o-mini",      # Text generation
    "vision": "gpt-4o-mini",    # Image captioning
    "embedding": "text-embedding-3-small"
}
```

**Available Models**:
*   `gpt-4o` - Most capable, higher cost
*   `gpt-4o-mini` - Balanced performance and cost
*   `gpt-3.5-turbo` - Fastest, lowest cost

#### Anthropic
```python
"anthropic": {
    "text": "claude-3-5-sonnet-20240620",
    "vision": "claude-3-5-sonnet-20240620",
    "embedding": None  # Uses OpenAI embeddings
}
```

**Available Models**:
*   `claude-3-5-sonnet-20240620` - Most capable
*   `claude-3-haiku-20240307` - Fastest

#### Google Gemini
```python
"google": {
    "text": "gemini-2.0-flash",
    "vision": "gemini-2.0-flash",
    "embedding": "models/text-embedding-004"
}
```

**Available Models**:
*   `gemini-2.0-flash` - Latest, fast
*   `gemini-1.5-pro` - Most capable
*   `gemini-1.5-flash` - Balanced

#### DeepSeek
```python
"deepseek": {
    "text": "deepseek-chat",
    "vision": None,  # No vision support
    "embedding": None  # Uses OpenAI embeddings
}
```

#### Ollama (Local)
```python
"ollama": {
    "text": "minicpm-v",
    "vision": "minicpm-v",
    "embedding": "text-embedding-3-small"
}
```

## Ingestion Configuration

### Text Processing

```python
# Chunk size for text splitting
CHUNK_SIZE = 1000

# Overlap between chunks (preserves context)
CHUNK_OVERLAP = 200
```

**Recommendations**:
*   **Technical docs**: 800-1000 chars
*   **Narrative text**: 1200-1500 chars
*   **Code**: 500-800 chars

### Image Processing

```python
# Minimum image size to process (bytes)
MIN_IMAGE_SIZE_BYTES = 2000

# Image caption prompt
IMAGE_CAPTION_PROMPT = """Describe this image in detail, focusing on:
1. What the image shows
2. Any text visible in the image
3. The context or purpose of this visual element

Keep the description concise but informative."""
```

## Retrieval Configuration

### Search Parameters

```python
# Initial retrieval count
TOP_K_RETRIEVAL = 10

# Final context size after reranking
TOP_K_FINAL = 5
```

**Tuning Guide**:
*   Increase `TOP_K_RETRIEVAL` for better recall (slower)
*   Decrease `TOP_K_FINAL` to reduce context noise
*   Balance based on your document corpus size

## Generation Configuration

### System Prompt

```python
RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Always cite the source of your information when answering.
If you cannot find the answer in the context, say so clearly."""
```

**Customization**:
*   Modify tone and style
*   Add domain-specific instructions
*   Enforce output format requirements

## Storage Configuration

### Vector Database

```python
# ChromaDB storage directory
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
```

**Data Paths**:
```python
# Project root
PROJECT_ROOT = Path(__file__).parent

# Input data directory
DATA_DIR = PROJECT_ROOT.parent / "basic_rag" / "data"
```

## Cost Configuration

### Pricing (USD per 1M tokens)

```python
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
}
```

**Note**: Prices are approximate and subject to change. Check provider documentation for current rates.

## Advanced Configuration

### Custom Provider

To add a new LLM provider:

1. Add model configuration to `MODELS` dict in `config.py`
2. Implement `LLMClient` interface in `generation/rag.py`
3. Register in `LLMFactory.create_client()`

### Custom Embeddings

To use custom embedding models:

1. Modify `MODELS[provider]["embedding"]`
2. Update `VectorStore` initialization in `ingestion/storage.py`

## Configuration Best Practices

1. **Development**: Use `ollama` for local testing
2. **Production**: Use `openai` or `anthropic` for reliability
3. **Cost Optimization**: Use `deepseek` or `gpt-4o-mini`
4. **Quality**: Use `claude-3-5-sonnet` or `gpt-4o`
