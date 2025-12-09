"""
LLM-as-a-Judge Evaluation Script.
Evaluates the RAG system using a strong "Judge" LLM.
Metrics: Context Relevance, Faithfulness, Answer Correctness.
"""
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add production_rag to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from production_rag.generation.rag import RAGGenerator, LLMFactory
from production_rag.retrieval.search import HybridRetriever
from production_rag.ingestion.storage import VectorStore
from production_rag.config import MODELS

# Load Env
load_dotenv(Path(__file__).parents[2] / ".env")

# ==========================================
# Configuration
# ==========================================
CANDIDATE_PROVIDER = "openai" # The model being tested
JUDGE_PROVIDER = "openai"     # The model doing the grading
JUDGE_MODEL = MODELS[JUDGE_PROVIDER]["text"]

# Evaluation Dataset (Question + Ground Truth)
DATASET = [
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

EVAL_PROMPT = """
You are an expert AI evaluator. Your task is to grade the performance of a RAG system.

### Input Data
1. **User Question**: {question}
2. **Ground Truth**: {ground_truth}
3. **Retrieved Context**: {context}
4. **Generated Answer**: {answer}

### Evaluation Criteria
Evaluate on a scale of 0 to 10 for each metric:

1. **Context Relevance (0-10)**: Is the retrieved context actually relevant to the question? (10 = Perfectly relevant chunks, 0 = Completely irrelevant/noise).
2. **Faithfulness (0-10)**: Is the generated answer derived *only* from the context? Ignore whether it's factually true in the real world, focus on whether the context supports it. (10 = Fully supported by context, 0 = Hallucinated information not in context).
3. **Answer Correctness (0-10)**: Does the generated answer match the meaning of the Ground Truth? (10 = Perfect meaning match, 0 = Completely wrong).

### Output Format (JSON)
{{
  "context_relevance": {{
    "score": <int>,
    "reason": "<string>"
  }},
  "faithfulness": {{
    "score": <int>,
    "reason": "<string>"
  }},
  "correctness": {{
    "score": <int>,
    "reason": "<string>"
  }}
}}

Return ONLY valid JSON.
"""

def run_evaluation():
    print(f"⚖️ Starting LLM-as-a-Judge Evaluation")
    print(f"📝 Candidate: {CANDIDATE_PROVIDER.upper()}")
    print(f"👨‍⚖️ Judge: {JUDGE_PROVIDER.upper()} ({JUDGE_MODEL})")
    print("-" * 80)
    
    # Init Candidate
    try:
        store = VectorStore()
        retriever = HybridRetriever(store)
        candidate_model = MODELS[CANDIDATE_PROVIDER]["text"]
        candidate_gen = RAGGenerator(
            provider_override=CANDIDATE_PROVIDER,
            model_override=candidate_model
        )
        if not candidate_gen.client:
             raise RuntimeError("Candidate client failed to init")
    except Exception as e:
        print(f"❌ Failed to init Candidate pipeline: {e}")
        return

    # Init Judge
    try:
        judge_client = LLMFactory.create_client(JUDGE_PROVIDER)
    except Exception as e:
        print(f"❌ Failed to init Judge client: {e}")
        return

    total_scores = {"relevance": 0, "faithfulness": 0, "correctness": 0}
    
    for i, item in enumerate(DATASET, 1):
        q = item["question"]
        gt = item["ground_truth"]
        
        print(f"\nExample {i}: {q}")
        
        # 1. Run Candidate Pipeline
        # We manually retrieve to capture context for the Judge
        chunks = retriever.retrieve_and_rerank(q)
        context_str = candidate_gen.format_context(chunks)
        response = candidate_gen.answer_query(q, chunks)
        cand_ans = response["answer"]
        
        print(f"  🤖 Candidate Answer: {cand_ans[:60]}...")
        
        # 2. Run Judge
        prompt = EVAL_PROMPT.format(
            question=q,
            ground_truth=gt,
            context=context_str,
            answer=cand_ans
        )
        
        try:
            # Need strict JSON from Judge
            # Some providers have json mode, but we will prompt generic "return JSON" 
            # and strip fences if needed.
            judge_res = judge_client.generate(
                system_prompt="You are an evaluator. Output only JSON.",
                user_prompt=prompt,
                model=JUDGE_MODEL
            )
            
            # Clean generic markdown fences if present
            cleaned_res = judge_res.content.replace("```json", "").replace("```", "").strip()
            eval_data = json.loads(cleaned_res)
            
            # Display Score
            rel = eval_data["context_relevance"]["score"]
            fai = eval_data["faithfulness"]["score"]
            cor = eval_data["correctness"]["score"]
            
            print(f"  ⭐ Scores -> Relevance: {rel}/10 | Faithfulness: {fai}/10 | Correctness: {cor}/10")
            print(f"  💡 Reason (Corr): {eval_data['correctness']['reason']}")
            
            total_scores["relevance"] += rel
            total_scores["faithfulness"] += fai
            total_scores["correctness"] += cor
            
        except Exception as e:
            print(f"  ❌ Judge Failed: {e}")

    # Summary
    n = len(DATASET)
    print("\n" + "="*80)
    print("📊 FINAL EVALUATION REPORT")
    print("="*80)
    print(f"Avg Context Relevance: {total_scores['relevance']/n:.1f}/10")
    print(f"Avg Faithfulness:      {total_scores['faithfulness']/n:.1f}/10")
    print(f"Avg Correctness:       {total_scores['correctness']/n:.1f}/10")
    print("="*80)

if __name__ == "__main__":
    run_evaluation()
