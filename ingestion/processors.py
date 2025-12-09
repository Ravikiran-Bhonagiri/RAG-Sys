"""
Processing module for chunking text and captioning images.
"""
import ollama
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

from production_rag.config import CHUNK_SIZE, CHUNK_OVERLAP, CAPTION_MODEL, IMAGE_CAPTION_PROMPT


class TextChunker:
    """Handles intelligent text chunking."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk text from pages into smaller segments.
        
        Args:
            pages: List of page dicts with 'text' and 'page_num'
            
        Returns:
            List of chunk dicts with text, page_num, and chunk_id
        """
        chunks = []
        
        for page in pages:
            page_text = page["text"]
            page_num = page["page_num"]
            
            if not page_text.strip():
                continue
            
            text_chunks = self.splitter.split_text(page_text)
            
            for i, chunk_text in enumerate(text_chunks):
                chunks.append({
                    "text": chunk_text,
                    "page_num": page_num,
                    "chunk_id": f"page{page_num}_chunk{i}",
                    "type": "text_chunk"
                })
        
        return chunks


class ImageCaptioner:
    """Generates captions for images using a VLM."""
    
    def __init__(self, model_name: str = CAPTION_MODEL):
        self.model_name = model_name
    
    def caption_image(self, image_bytes: bytes) -> str:
        """
        Generate a caption for a single image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Caption text
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': IMAGE_CAPTION_PROMPT,
                    'images': [image_bytes]
                }]
            )
            return response['message']['content']
        except Exception as e:
            print(f"⚠️ Warning: Failed to caption image: {e}")
            return "[Image: Caption generation failed]"
    
    def caption_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate captions for multiple images.
        
        Args:
            images: List of image dicts with 'image_bytes', 'page_num', 'image_id'
            
        Returns:
            List of caption dicts with caption text and metadata
        """
        captioned = []
        
        for img in images:
            print(f"  🖼️ Captioning {img['image_id']}...")
            caption = self.caption_image(img["image_bytes"])
            
            captioned.append({
                "text": caption,
                "page_num": img["page_num"],
                "image_id": img["image_id"],
                "chunk_id": f"{img['image_id']}_caption",
                "type": "image_caption"
            })
        
        return captioned
