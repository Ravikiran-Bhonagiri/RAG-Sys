"""
Tests for ingestion pipeline components.
"""
import pytest
from pathlib import Path
import sys

# Add production_rag to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from ingestion.processors import TextChunker, ImageCaptioner

class MockSplitter:
    def split_text(self, text):
        return ["chunk1", "chunk2"]

class MockOllama:
    def chat(self, model, messages):
        return {'message': {'content': 'This is a mocked caption'}}

@pytest.fixture
def chunker():
    return TextChunker()

def test_text_chunker_empty():
    chunker = TextChunker()
    pages = [{"text": "", "page_num": 1}]
    chunks = chunker.chunk_pages(pages)
    assert len(chunks) == 0

def test_text_chunker_basic():
    chunker = TextChunker(chunk_size=10, chunk_overlap=0)
    # Mocking behavior for simple split check or just rely on Langchain logic
    # Here we test our wrapper
    pages = [{"text": "Hello world. This is a test.", "page_num": 1}]
    chunks = chunker.chunk_pages(pages)
    assert len(chunks) > 0
    assert chunks[0]['page_num'] == 1
    assert chunks[0]['type'] == 'text_chunk'

def test_image_captioner_mock(monkeypatch):
    # Mock ollama
    import ingestion.processors
    monkeypatch.setattr(ingestion.processors, "ollama", MockOllama())
    
    captioner = ImageCaptioner(model_name="test-model")
    images = [{
        "image_bytes": b"fake",
        "page_num": 1,
        "image_id": "img1"
    }]
    
    captions = captioner.caption_images(images)
    
    assert len(captions) == 1
    assert captions[0]['text'] == 'This is a mocked caption'
    assert captions[0]['image_id'] == 'img1'
    assert captions[0]['type'] == 'image_caption'
