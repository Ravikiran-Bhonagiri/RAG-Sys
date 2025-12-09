"""
Document extraction and parsing module.
Handles PDF loading and element extraction (text, images, tables).
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import io

from production_rag.config import MIN_IMAGE_SIZE_BYTES


class PDFExtractor:
    """Extracts text and images from PDF documents."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.doc = fitz.open(str(self.pdf_path))
        
    def extract_text_by_page(self) -> List[Dict[str, Any]]:
        """
        Extract text content from each page.
        
        Returns:
            List of dicts with page_num and text
        """
        pages = []
        for page_num, page in enumerate(self.doc, start=1):
            text = page.get_text()
            pages.append({
                "page_num": page_num,
                "text": text,
                "type": "text"
            })
        return pages
    
    def extract_images(self) -> List[Dict[str, Any]]:
        """
        Extract all images from the PDF.
        
        Returns:
            List of dicts with page_num, image_bytes, and image_id
        """
        images = []
        for page_num, page in enumerate(self.doc, start=1):
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Skip tiny images (likely decorative)
                if len(image_bytes) < MIN_IMAGE_SIZE_BYTES:
                    continue
                
                images.append({
                    "page_num": page_num,
                    "image_id": f"page{page_num}_img{img_index}",
                    "image_bytes": image_bytes,
                    "type": "image"
                })
        
        return images
    
    def extract_all(self) -> Dict[str, List]:
        """
        Extract both text and images.
        
        Returns:
            Dict with 'pages' (text per page) and 'images' (all images)
        """
        return {
            "pages": self.extract_text_by_page(),
            "images": self.extract_images(),
            "metadata": {
                "filename": self.pdf_path.name,
                "total_pages": len(self.doc)
            }
        }
    
    def close(self):
        """Close the PDF document."""
        if self.doc:
            self.doc.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
