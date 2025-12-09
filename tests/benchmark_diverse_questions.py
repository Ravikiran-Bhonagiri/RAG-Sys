"""
Diverse Questions Benchmark.
Runs 5 specific questions across OpenAI, Anthropic, Google, and DeepSeek
using the REAL Vector Store (ChromaDB) to test the full pipeline.
"""
import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Add production_rag to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from production_rag.generation.rag import RAGGenerator
from production_rag.retrieval.search import HybridRetriever
from production_rag.ingestion.storage import VectorStore
from production_rag.config import MODELS, CHROMA_DB_DIR

# Load Env
load_dotenv(Path(__file__).parents[2] / ".env")

PROVIDERS = ["openai", "anthropic", "google", "deepseek"]

QUESTIONS = [
    "What is the Mastra framework and what language is it built in?",
    "Explain the 'ReAct' pattern mentioned in the text.",
    "What are the core components of an Agent in this system?",
    "Describe the architecture diagram showing how agents connect to tools.",
    "What is the difference between a Workflow and an Agent according to the document?"
]

def run_benchmark():
    print(f"🚀 Starting Diverse Questions Benchmark")
    print(f"📂 Database: {CHROMA_DB_DIR}")
    
    # Initialize Retrieval (Common for all)
    try:
        store = VectorStore()
        retriever = HybridRetriever(store)
    except Exception as e:
        print(f"❌ Failed to load Vector DB: {e}")
        return

    for q_idx, question in enumerate(QUESTIONS, 1):
        print(f"\n{'='*80}")
        print(f"❓ Q{q_idx}: {question}")
        print(f"{'='*80}")
        
        # Retrieval Step (Shared)
        # We retrieve once per question to ensure consistency, 
        # or we could retrieve inside the loop if we want to test slight variances 
        # but retrieval is deterministic here so once is fine.
        start_retrieval = time.time()
        retrieved_chunks = retriever.retrieve(question, top_k=5)
        # Rerank
        reranked_chunks = retriever.rerank(retrieved_chunks, question, top_k=3)
        retrieval_time = time.time() - start_retrieval
        
        print(f"🔎 Retrieved {len(reranked_chunks)} chunks in {retrieval_time:.2f}s")
        sources = [f"Page {c['metadata'].get('page_num')}" for c in reranked_chunks]
        print(f"📄 Sources: {sources}")

        print(f"\n{'PROVIDER':<12} | {'LATENCY':<8} | {'COST':<8} | {'ANSWER'}")
        print("-" * 100)

        for provider in PROVIDERS:
            try:
                # Determine specific model
                model_name = MODELS[provider]["text"]
                
                # Init Generator
                # Note: We re-init per provider
                generator = RAGGenerator(provider_override=provider, model_override=model_name)
                
                if not generator.client:
                    print(f"{provider:<12} | {'SKIP':<8} | {'-':<8} | No Client")
                    continue
                
                # Answer
                # We use answer_query but pass our pre-retrieved chunks to use the specific ones
                # actually answer_query calls format_context internally, so that's perfect.
                response = generator.answer_query(question, reranked_chunks)
                
                # Display
                lat = f"{response['latency']}s"
                cost = f"${response['cost']:.6f}"
                ans = response['answer'].replace('\n', ' ')[:60] + "..."
                
                print(f"{provider:<12} | {lat:<8} | {cost:<8} | {ans}")
                
            except Exception as e:
                print(f"{provider:<12} | {'ERR':<8} | {'-':<8} | {str(e)[:40]}")

if __name__ == "__main__":
    run_benchmark()
