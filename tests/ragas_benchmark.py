"""
RAGAS Benchmark Script.
Uses the RAGAS framework to evaluate RAG performance metrics.
"""
import sys
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Add production_rag to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from production_rag.generation.rag import RAGGenerator
from production_rag.retrieval.search import HybridRetriever
from production_rag.ingestion.storage import VectorStore
from production_rag.config import MODELS

# RAGAS Imports
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# Load Env (For OpenAI Key used by RAGAS)
load_dotenv(Path(__file__).parents[2] / ".env")

# ==========================================
# Configuration
# ==========================================
CANDIDATE_PROVIDER = "openai" # Model to test
CANDIDATE_MODEL = "gpt-4o-mini" # Specific model

# Dataset
QUESTIONS = [
    {
        "question": "What is the Mastra framework and what language is it built in?",
        "ground_truth": "Mastra is an open-source TypeScript framework for building AI applications."
    },
    {
        "question": "Explain the 'ReAct' pattern mentioned in the text.",
        "ground_truth": "ReAct stands for 'Reasoning and Acting'. It is a pattern where an agent loops through reasoning (thinking about what to do) and acting (executing tool calls) to solve complex tasks."
    },
    {
        "question": "What are the core components of an Agent in this system?",
        "ground_truth": "The core components of an Agent are the Model (LLM), Tools (functions it can call), Memory (context history), and Workflows (structured paths)."
    },
    {
        "question": "Describe the architecture diagram showing how agents connect to tools.",
        "ground_truth": "The diagram shows an Agent block connected to a Tools block. The Agent receives input, consults its Memory, uses the Model to decide, and then invokes Tools to interact with the outside world."
    },
    {
        "question": "What is the difference between a Workflow and an Agent according to the document?",
        "ground_truth": "A Workflow is a deterministic, structured sequence of steps (like a flowchart). An Agent is probabilistic and autonomous; it decides its own steps based on the goal."
    }
]

def run_ragas_evaluation():
    print(f"🚀 Starting RAGAS Evaluation")
    print(f"📝 Candidate: {CANDIDATE_PROVIDER.upper()} ({CANDIDATE_MODEL})")
    
    # Init RAG System
    try:
        store = VectorStore()
        retriever = HybridRetriever(store)
        generator = RAGGenerator(
            provider_override=CANDIDATE_PROVIDER,
            model_override=CANDIDATE_MODEL
        )
    except Exception as e:
        print(f"❌ Failed to init RAG pipeline: {e}")
        return

    # Prepare Data for RAGAS
    questions = []
    answers = [] # Generated answer
    contexts_list = [] # List of list of strings
    ground_truths = [] # List of strings
    
    print("\ngenerating answers...")
    
    for item in QUESTIONS:
        q = item["question"]
        gt = item["ground_truth"]
        print(f"  Processing: {q[:40]}...")
        
        # Run Pipeline
        chunks = retriever.retrieve_and_rerank(q)
        response = generator.answer_query(q, chunks)
        
        # RAGAS expects 'contexts' as list of strings
        # Our chunks are dicts, need to extract 'text'
        ctx_strs = [c['text'] for c in chunks]
        
        questions.append(q)
        answers.append(response['answer'])
        contexts_list.append(ctx_strs)
        ground_truths.append(gt)

    # create HF Dataset
    data = {
        'question': questions,
        'answer': answers,
        'contexts': contexts_list,
        'ground_truth': ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    # Init RAGAS Metrics
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from production_rag.config import DEEPSEEK_API_KEY
        
        # Explicitly setup DeepSeek as Judge
        # DeepSeek is OpenAI Compatible
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
        
        # Use OpenAI Embeddings (standard) for metric calculation
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        print("\n🔍 Running RAGAS Metrics (Judge: DeepSeek)...")
        results = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ],
            llm=llm,
            embeddings=embeddings
        )
        
        print("\n📊 RAGAS Results:")
        df = results.to_pandas()
        print(df.to_markdown())
        
        # Save to file
        df.to_csv("ragas_results.csv", index=False)
        print("\nSaved results to ragas_results.csv")
        
    except Exception as e:
        print(f"❌ RAGAS Evaluation Failed: {e}")

if __name__ == "__main__":
    run_ragas_evaluation()
