# Testing and Evaluation Guide

Comprehensive guide for testing, benchmarking, and evaluating the Production RAG system.

## Overview

The testing suite includes three levels:
1. **Unit Tests** - Component-level validation
2. **Integration Benchmarks** - End-to-end functional testing
3. **RAGAS Evaluation** - Automated quality assessment

## 1. Unit Testing

### Running Unit Tests

Execute the full test suite:

```bash
pytest production_rag/tests/
```

Run specific test modules:

```bash
# Test ingestion components
pytest production_rag/tests/test_ingestion.py

# Test retrieval
pytest production_rag/tests/test_retrieval.py

# Test generation
pytest production_rag/tests/test_generation.py
```

### Test Coverage

#### Ingestion Tests (`test_ingestion.py`)

Tests the document processing pipeline:

*   **TextChunker**: Validates text splitting logic
    *   Chunk size constraints
    *   Overlap behavior
    *   Metadata preservation
*   **ImageCaptioner**: Validates VLM integration
    *   Caption generation
    *   Error handling
    *   Mock LLM responses

#### Storage Tests (`test_storage.py`)

Tests vector database operations:

*   **VectorStore**: ChromaDB integration
    *   Document addition
    *   Similarity search
    *   Metadata filtering

#### Retrieval Tests (`test_retrieval.py`)

Tests search and ranking:

*   **HybridRetriever**: Search pipeline
    *   Vector similarity
    *   Reranking logic
    *   Result formatting

#### Generation Tests (`test_generation.py`)

Tests answer generation:

*   **RAGGenerator**: LLM integration
    *   Context formatting
    *   Multi-provider support
    *   Citation handling

### Writing New Tests

Follow the existing patterns:

```python
import pytest
from unittest.mock import Mock, patch

def test_component_behavior():
    # Arrange
    component = YourComponent()
    
    # Act
    result = component.method()
    
    # Assert
    assert result == expected_value
```

## 2. Integration Benchmarks

### Diverse Questions Benchmark

Tests the complete RAG pipeline with varied question types.

#### Running the Benchmark

```bash
python -m production_rag.tests.benchmark_diverse_questions
```

#### Question Categories

The benchmark tests 5 question types:

1. **Factual**: Direct information retrieval
   *   Example: "What is the Mastra framework?"
   
2. **Conceptual**: Abstract concept explanation
   *   Example: "Explain the ReAct pattern"
   
3. **Component-based**: System architecture questions
   *   Example: "What are the core components of an Agent?"
   
4. **Visual**: Image-based questions
   *   Example: "Describe the architecture diagram"
   
5. **Reasoning**: Comparative analysis
   *   Example: "What is the difference between Workflow and Agent?"

#### Output

The benchmark generates:
*   Console output with real-time progress
*   `benchmark_report.md` with detailed results
*   Performance metrics (latency, token usage, cost)

#### Interpreting Results

**Success Criteria**:
*   All questions answered (no refusals)
*   Answers contain relevant information
*   Source citations present
*   Reasonable latency (<5s per query)

**Common Issues**:
*   **Refusals**: Indicates retrieval noise or insufficient context
*   **Hallucinations**: Check faithfulness metrics
*   **Missing citations**: Review prompt engineering

## 3. RAGAS Evaluation

### Overview

RAGAS (Retrieval Augmented Generation Assessment) provides automated quality scoring using an LLM judge.

### Setup

Install RAGAS dependencies:

```bash
pip install ragas datasets tabulate
```

Ensure API keys are configured in `.env`:
*   `DEEPSEEK_API_KEY` (recommended judge)
*   `OPENAI_API_KEY` (alternative judge)

### Running RAGAS

```bash
python -m production_rag.tests.ragas_benchmark
```

### Metrics Explained

#### Faithfulness (0.0 - 1.0)

**Measures**: Hallucination detection

**Definition**: Are the generated answers grounded in the retrieved context?

**Scoring**:
*   `1.0` - Perfect grounding, no hallucinations
*   `0.8+` - Acceptable for production
*   `<0.7` - Indicates hallucination issues

**Example**:
*   **Context**: "Mastra is a TypeScript framework"
*   **Good Answer**: "Mastra is built in TypeScript" (Score: 1.0)
*   **Bad Answer**: "Mastra is a Python framework" (Score: 0.0)

#### Answer Relevancy (0.0 - 1.0)

**Measures**: Question-answer alignment

**Definition**: Does the answer actually address the user's question?

**Scoring**:
*   `1.0` - Directly answers the question
*   `0.8+` - Acceptable relevance
*   `<0.5` - Off-topic or refusal

**Example**:
*   **Question**: "What is Mastra?"
*   **Good Answer**: "Mastra is a TypeScript framework..." (Score: 1.0)
*   **Bad Answer**: "I cannot answer this question" (Score: 0.0)

#### Context Precision (0.0 - 1.0)

**Measures**: Retrieval signal-to-noise ratio

**Definition**: What percentage of retrieved chunks are actually relevant?

**Scoring**:
*   `1.0` - All chunks relevant
*   `0.7+` - Acceptable noise level
*   `<0.5` - Too much irrelevant content

**Interpretation**:
*   Low scores indicate retrieval needs tuning
*   Consider increasing reranking strictness
*   May need better embedding model

#### Context Recall (0.0 - 1.0)

**Measures**: Retrieval completeness

**Definition**: Did we retrieve all necessary information to answer?

**Scoring**:
*   `1.0` - All required info retrieved
*   `0.7+` - Sufficient for most answers
*   `<0.5` - Missing critical information

**Interpretation**:
*   Low scores indicate insufficient retrieval
*   Consider increasing `TOP_K_RETRIEVAL`
*   May need better query expansion

### Configuring RAGAS

Edit `production_rag/tests/ragas_benchmark.py`:

```python
# Candidate model (system being tested)
CANDIDATE_PROVIDER = "openai"
CANDIDATE_MODEL = "gpt-4o-mini"

# Judge model (evaluator)
# Uses DeepSeek by default for cost efficiency
```

### Judge Model Selection

**DeepSeek** (Default):
*   **Pros**: Ultra-low cost, strict grading
*   **Cons**: May be overly harsh on edge cases
*   **Use**: Continuous integration, development

**GPT-4o**:
*   **Pros**: Nuanced evaluation, industry standard
*   **Cons**: Higher cost
*   **Use**: Final validation, production benchmarks

**Claude 3.5 Sonnet**:
*   **Pros**: Excellent reasoning, detailed feedback
*   **Cons**: Highest cost
*   **Use**: Deep analysis, research

### Output Files

RAGAS generates:
*   `ragas_results.csv` - Raw scores per question
*   Console output with summary statistics

### Interpreting RAGAS Results

**Excellent System** (Production-ready):
*   Faithfulness: >0.9
*   Answer Relevancy: >0.9
*   Context Precision: >0.7
*   Context Recall: >0.8

**Good System** (Acceptable):
*   Faithfulness: >0.8
*   Answer Relevancy: >0.8
*   Context Precision: >0.5
*   Context Recall: >0.7

**Needs Improvement**:
*   Any metric <0.5

## 4. Performance Testing

### Latency Benchmarks

Measure query latency:

```bash
time python -m production_rag.main query "test question"
```

**Target Metrics**:
*   Retrieval: <500ms
*   Generation: 1-3s (varies by provider)
*   Total: <5s

### Cost Analysis

Track costs across providers:

```python
# Costs are displayed after each query
# Format: $X.XXXX (Input: X tokens, Output: X tokens)
```

**Cost Optimization**:
*   Use `deepseek` for development ($0.0001/query)
*   Use `gpt-4o-mini` for production ($0.001/query)
*   Use `gpt-4o` only when quality is critical ($0.01/query)

## 5. Continuous Integration

### Automated Testing

Recommended CI pipeline:

```bash
# 1. Unit tests (fast)
pytest production_rag/tests/test_*.py

# 2. Integration benchmark (medium)
python -m production_rag.tests.benchmark_diverse_questions

# 3. RAGAS evaluation (slow, run nightly)
python -m production_rag.tests.ragas_benchmark
```

### Quality Gates

**Minimum Thresholds**:
*   Unit tests: 100% pass
*   Benchmark: 80% questions answered
*   RAGAS Faithfulness: >0.8
*   RAGAS Answer Relevancy: >0.7

## 6. Debugging Failed Tests

### Unit Test Failures

Check:
1. Mock configurations
2. API key availability
3. ChromaDB initialization

### Benchmark Failures

Check:
1. Vector store populated (`ingest` ran successfully)
2. API keys valid
3. Network connectivity

### RAGAS Low Scores

**Low Faithfulness**:
*   Review system prompt
*   Check for hallucination patterns
*   Validate context assembly

**Low Relevancy**:
*   Improve retrieval quality
*   Adjust `TOP_K_FINAL`
*   Enhance query processing

**Low Precision**:
*   Implement reranking
*   Filter irrelevant chunks
*   Improve embedding quality

**Low Recall**:
*   Increase `TOP_K_RETRIEVAL`
*   Improve chunking strategy
*   Enhance metadata

## Best Practices

1. **Run unit tests frequently** during development
2. **Run benchmarks before commits** to catch regressions
3. **Run RAGAS weekly** to track quality trends
4. **Document test failures** with reproduction steps
5. **Update ground truth** as system evolves
