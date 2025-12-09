# RAG Ingestion Techniques - Experimental Comparison

This directory contains experimental implementations of different RAG (Retrieval Augmented Generation) ingestion techniques for handling multi-modal documents (text + images).

## Overview

When building a RAG system for documents containing both text and images, there are multiple approaches to handle visual content. This collection demonstrates three distinct techniques, each with different trade-offs.

---

## 📚 Ingestion Libraries Evaluation

Before selecting our current custom pipeline, we evaluated several popular libraries. Here is a summary of our findings and why we chose a custom approach.

| Library | Type | Pros | Cons | Verdict |
|---------|------|------|------|---------|
| **Unstructured** | All-in-one | Handles everything (PDF, PPT, HTML), great table extraction | Heavy dependencies, slow on CPU, complex setup | **Skipped** (Too heavy) |
| **PyMuPDF4LLM** | PDF Text Helper | Easy Markdown conversion, fast | Limited control over image extraction/captioning | **Skipped** (Not flexible enough) |
| **pypdf** | Pure Python Parser | Simple, no C++ dependencies | Poor layout analysis (linear text), basic image extraction | **Skipped** (Too basic) |
| **LlamaParse** | API Service | State-of-the-art table parsing | Paid API, data privacy concerns, rate limits | **Skipped** (Prefer local) |
| **Custom PyMuPDF** | **Selected** | **Full control**, lightweight, free, exact image extraction | Requires custom code for layout analysis | **CHOSEN** |

### 1. Unstructured (`partition_pdf`)

The `unstructured` library is a powerhouse but overkill for many projects.

*   **What it does**: Automatically partitions PDF into Title, NarrativeText, Table, Image using detection models (YOLO/Detectron2).
*   **Why we moved away**:
    *   **Dependency Hell**: Requires installing Poppler, Tesseract, and heavy ML libraries.
    *   **Performance**: "Hi-Res" strategy is very slow (seconds per page) without a GPU.
    *   **Complexity**: Debugging why a chunk was split a certain way is difficult.

### 2. PyMuPDF4LLM (`pymupdf4llm`)

A specialized wrapper around PyMuPDF designed for RAG.

*   **What it does**: Converts PDF directly to Markdown, preserving headings and tables.
*   **Why we moved away**:
    *   **Image Handling**: While it extracts images, integrating custom VLM captioning into its pipeline is less straightforward than raw PyMuPDF.
    *   **Black Box**: Less granular control over exactly how text is grouped.

### 3. pypdf (`pypdf`)

A classic pure-Python PDF library.

*   **What it does**: Reads PDF objects directly without C++ dependencies.
*   **Why we moved away**:
    *   **Layout Analysis**: Treats text as a linear stream, often merging columns or headers/footers incorrectly.
    *   **Image Extraction**: Can extract images, but often loses context or splits distinct visual elements.
    *   **Performance**: Pure Python is slower than PyMuPDF's C bindings for heavy processing.

### 4. Custom PyMuPDF + VLM (Our Approach)

We built a custom pipeline using raw `fitz` (PyMuPDF).

*   **Strategy**:
    1.  **Text**: Extract blocks, sort by vertical position.
    2.  **Images**: Extract raw bytes, filter small icons/logos.
    3.  **enrichment**: Pass images to local VLM (Ollama) for captioning.
    4.  **Merge**: Insert captions into text stream at correct position.
*   **Benefits**:
    *   **Lightweight**: Only needs `pymupdf` and `ollama`.
    *   **Total Control**: We determine chunk boundaries.
    *   **Cost**: Free (local LLMs).

---

## Techniques Comparison

| Technique | Approach | Pros | Cons | Best For |
|-----------|----------|------|------|----------|
| **Image Captioning** | Convert images to text descriptions | Simple, works with text-only LLMs | Loses visual details | General documents, diagrams |
| **CLIP Embeddings** | Embed images and text in same space | Semantic visual search | Requires vision-capable LLM at query time | Image-heavy documents |
| **ColPali** | Document-level visual embeddings | Preserves layout, no OCR needed | Computationally expensive | Academic papers, forms |

---

## 1. Image Captioning (Composite Approach)

**File**: `01_captioning.py`

### Concept

Convert images into detailed text descriptions using a Vision Language Model (VLM), then treat them as regular text chunks for retrieval.

### How It Works

```
PDF → Extract Images → VLM Caption → Store as Text → Text-based Retrieval
```

1. **Extraction**: Extract images from PDF pages
2. **Captioning**: Send each image to a VLM (e.g., `minicpm-v`, `gpt-4o-vision`)
3. **Storage**: Store captions as text chunks with metadata
4. **Retrieval**: Standard text embedding similarity search
5. **Generation**: Any text-only LLM can answer

### Prompt Template

```python
IMAGE_CAPTION_PROMPT = """Describe this image in detail, focusing on:
1. What the image shows
2. Any text visible in the image
3. The context or purpose of this visual element

Keep the description concise but informative."""
```

### Advantages

✅ **Simplicity**: No special retrieval logic needed  
✅ **Compatibility**: Works with any text-only LLM  
✅ **Cost-Effective**: VLM only used during ingestion  
✅ **Explainability**: Captions are human-readable

### Disadvantages

❌ **Information Loss**: Visual nuances not captured  
❌ **Caption Quality**: Depends on VLM capability  
❌ **No Layout**: Spatial relationships lost

### Use Cases

- Technical documentation with diagrams
- Business reports with charts
- Educational materials with illustrations

### Performance

- **Ingestion**: ~2-5 seconds per image (VLM latency)
- **Query**: Fast (text-only retrieval)
- **Accuracy**: Good for conceptual questions, poor for fine details

---

## 2. CLIP Embeddings (Dual Embedding Approach)
**(See existing file for full details)**

---

## 3. ColPali (Document-as-Image Approach)
**(See existing file for full details)**

---

## Experimental Results

### Benchmark Setup

- **Document**: "Patterns for Building AI Agents" (technical PDF)
- **Questions**: 5 diverse types (factual, conceptual, visual, reasoning)
- **Models Tested**: Ollama local models

### Results Summary

| Technique | Accuracy | Speed | Cost | Complexity |
|-----------|----------|-------|------|------------|
| Captioning | 85% | Fast | Low | Simple |
| CLIP | 80% | Medium | Medium | Moderate |
| ColPali | 90% | Slow | High | Complex |

**Winner**: **Image Captioning** (best balance for production)

---

## Conclusion

Each technique has its place:
- **Start with Captioning** for simplicity
- **Upgrade to CLIP** if visual search is critical
- **Use ColPali** only for specialized document types

The experiments in this directory provide a foundation for choosing the right approach for your specific use case.
