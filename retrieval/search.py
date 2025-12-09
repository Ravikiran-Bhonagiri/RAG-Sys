"""
Search and retrieval module.
"""
from typing import List, Dict, Any

from production_rag.ingestion.storage import VectorStore
from production_rag.config import TOP_K_RETRIEVAL, TOP_K_FINAL


class HybridRetriever:
    """
    Handles hybrid search combining vector similarity with keyword matching.
    For simplicity, we use ChromaDB's built-in vector search.
    In production, you would combine this with BM25 keyword search.
    """
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            
        Returns:
            List of relevant chunks with text and metadata
        """
        # Vector search
        results = self.vector_store.search(query, top_k=top_k)
        
        return results
    
    def rerank(self, results: List[Dict[str, Any]], query: str, top_k: int = TOP_K_FINAL) -> List[Dict[str, Any]]:
        """
        Rerank results (simplified version).
        In production, use a cross-encoder model like Cohere Rerank.
        
        For now, we just return top_k results by distance.
        """
        # Sort by distance (lower is better)
        sorted_results = sorted(
            results,
            key=lambda x: x.get('distance', float('inf'))
        )
        
        return sorted_results[:top_k]
    
    def retrieve_and_rerank(self, query: str) -> List[Dict[str, Any]]:
        """
        Full retrieval pipeline: search -> rerank.
        
        Args:
            query: User query
            
        Returns:
            Top reranked results
        """
        # Step 1: Retrieve top_k candidates
        results = self.retrieve(query, top_k=TOP_K_RETRIEVAL)
        
        # Step 2: Rerank to get final top_k
        final_results = self.rerank(results, query, top_k=TOP_K_FINAL)
        
        return final_results
