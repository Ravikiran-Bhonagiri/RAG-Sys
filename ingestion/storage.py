"""
Vector database storage module using ChromaDB.
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from pathlib import Path

from production_rag.config import CHROMA_DB_DIR


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB."""
    
    def __init__(self, collection_name: str = "production_rag"):
        """Initialize ChromaDB client and collection."""
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Production RAG embeddings"}
        )
    
    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Add text chunks to the vector store.
        
        Args:
            chunks: List of chunk dicts with 'text', 'chunk_id', and metadata
        """
        if not chunks:
            return
        
        documents = []
        ids = []
        metadatas = []
        
        for chunk in chunks:
            documents.append(chunk["text"])
            ids.append(chunk["chunk_id"])
            
            # Store metadata
            metadata = {
                "page_num": chunk["page_num"],
                "type": chunk["type"]
            }
            # Add image_id if it's an image caption
            if "image_id" in chunk:
                metadata["image_id"] = chunk["image_id"]
            
            metadatas.append(metadata)
        
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        
        print(f"✅ Added {len(chunks)} chunks to vector store")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of results with text, metadata, and distance
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Format results
        formatted = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted.append({
                    "text": doc,
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if results['distances'] else None
                })
        
        return formatted
    
    def count(self) -> int:
        """Get total number of chunks in the collection."""
        return self.collection.count()
    
    def reset(self):
        """Delete and recreate the collection (for testing)."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"description": "Production RAG embeddings"}
        )
