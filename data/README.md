# Sample Data

This directory contains sample PDF documents for testing and demonstration purposes.

## Contents

- `Patterns for Building AI Agents.pdf` - Reference document used for RAG system testing

## Usage

The RAG system automatically processes PDFs from this directory during ingestion:

```bash
python -m production_rag.main ingest --reset
```

## Adding Your Own Documents

1. Place PDF files in this directory
2. Run the ingestion command
3. Query the system with questions about your documents
