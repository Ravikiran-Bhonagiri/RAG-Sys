"""
Tests for retrieval pipeline.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add production_rag to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from retrieval.search import HybridRetriever
from ingestion.storage import VectorStore

@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=VectorStore)
    return store

def test_hybrid_retriever_basic(mock_vector_store):
    # Setup mock return
    mock_vector_store.search.return_value = [
        {"text": "chunk1", "metadata": {"page_num": 1}, "distance": 0.5},
        {"text": "chunk2", "metadata": {"page_num": 2}, "distance": 0.1}
    ]
    
    retriever = HybridRetriever(mock_vector_store)
    results = retriever.retrieve("test query")
    
    assert len(results) == 2
    mock_vector_store.search.assert_called_once()

def test_reranker_sorts_correctly(mock_vector_store):
    retriever = HybridRetriever(mock_vector_store)
    
    # Unsorted input
    results = [
        {"text": "bad", "distance": 0.9},
        {"text": "good", "distance": 0.1},
        {"text": "ok", "distance": 0.5}
    ]
    
    reranked = retriever.rerank(results, "query", top_k=2)
    
    assert len(reranked) == 2
    assert reranked[0]['text'] == "good" # Smallest distance first
    assert reranked[1]['text'] == "ok"
