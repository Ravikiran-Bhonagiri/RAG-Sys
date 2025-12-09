"""
Tests for generation pipeline with multi-provider support.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add production_rag to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from generation.rag import RAGGenerator, LLMFactory, OllamaClient

from unittest.mock import patch, MagicMock

def test_format_context():
    # Setup mock factory to avoid real initialization
    # We patch the factory to return a mock client
    mock_client = MagicMock()
    with patch('generation.rag.LLMFactory.create_client', return_value=mock_client):
        generator = RAGGenerator()
        chunks = [
            {"text": "Hello world", "metadata": {"page_num": 1, "type": "text_chunk"}},
            {"text": "Image description", "metadata": {"page_num": 2, "type": "image_caption"}}
        ]
        
        context = generator.format_context(chunks)
        
        assert "[Source 1 - Page 1]" in context
        assert "Hello world" in context
        assert "[Source 2 - Page 2 [IMAGE]]" in context

def test_generate_answer_mock(monkeypatch):
    # Mock the factory to return a MockClient
    mock_client = MagicMock()
    mock_client.generate.return_value = "Generated Answer Provider"
    
    monkeypatch.setattr(LLMFactory, "create_client", lambda provider: mock_client)
    
    generator = RAGGenerator()
    answer = generator.generate_answer("query", "context")
    
    assert answer == "Generated Answer Provider"
    mock_client.generate.assert_called_once()

def test_answer_query_integration(monkeypatch):
    mock_client = MagicMock()
    mock_client.generate.return_value = "Generated Answer Integration"
    
    monkeypatch.setattr(LLMFactory, "create_client", lambda provider: mock_client)
    
    generator = RAGGenerator()
    chunks = [{"text": "ctx", "metadata": {"page_num": 1}}]
    
    response = generator.answer_query("test query", chunks)
    
    assert response['query'] == "test query"
    assert response['answer'] == "Generated Answer Integration"
    assert response['provider'] == "ollama" # Default from config
