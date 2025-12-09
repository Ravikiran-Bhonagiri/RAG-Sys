"""
RAG generation module.
Supports multi-provider generation using a Factory Pattern.
Includes Token Usage and Cost Calculation.
"""
import abc
import os
import time
import ollama
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from production_rag.config import LLM_MODEL, RAG_SYSTEM_PROMPT, LLM_PROVIDER, PRICING
from production_rag.config import OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, DEEPSEEK_API_KEY


@dataclass
class GenerationResult:
    content: str
    model_name: str
    usage: Dict[str, int]  # {'input': int, 'output': int}
    provider: str


# ==========================================
# Abstract Base Client
# ==========================================
class LLMClient(abc.ABC):
    """Abstract interface for LLM providers."""
    
    @abc.abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, model: str) -> GenerationResult:
        pass


# ==========================================
# Concrete Client Implementations
# ==========================================
class OllamaClient(LLMClient):
    def generate(self, system_prompt: str, user_prompt: str, model: str) -> GenerationResult:
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ]
            )
            # Ollama returns usage statistics in the response object usually
            # But the python client might wrap it. 
            # We will try to extract if available, else 0
            # eval_count = output tokens, prompt_eval_count = input tokens
            input_tokens = response.get('prompt_eval_count', 0)
            output_tokens = response.get('eval_count', 0)

            return GenerationResult(
                content=response['message']['content'],
                model_name=model,
                usage={"input": input_tokens, "output": output_tokens},
                provider="ollama"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama Error: {e}")


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, base_url: Optional[str] = None, provider_name="openai"):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.provider_name = provider_name
        except ImportError:
            raise ImportError("OpenAI SDK not installed.")

    def generate(self, system_prompt: str, user_prompt: str, model: str) -> GenerationResult:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            usage = response.usage
            return GenerationResult(
                content=response.choices[0].message.content,
                model_name=model,
                usage={
                    "input": usage.prompt_tokens,
                    "output": usage.completion_tokens
                },
                provider=self.provider_name
            )
        except Exception as e:
             raise RuntimeError(f"{self.provider_name.capitalize()} Error: {e}")


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str):
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Anthropic SDK not installed.")

    def generate(self, system_prompt: str, user_prompt: str, model: str) -> GenerationResult:
        try:
            message = self.client.messages.create(
                model=model,
                max_tokens=1024,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return GenerationResult(
                content=message.content[0].text,
                model_name=model,
                usage={
                    "input": message.usage.input_tokens,
                    "output": message.usage.output_tokens
                },
                provider="anthropic"
            )
        except Exception as e:
             raise RuntimeError(f"Anthropic Error: {e}")


class GoogleClient(LLMClient):
    def __init__(self, api_key: str):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.genai = genai
        except ImportError:
            raise ImportError("Google SDK not installed.")

    def generate(self, system_prompt: str, user_prompt: str, model: str) -> GenerationResult:
        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            model_instance = self.genai.GenerativeModel(model)
            response = model_instance.generate_content(full_prompt)
            
            # Extract usage metadata
            # Note: Gemini usage metadata access varies by version.
            # Usually usage_metadata is on the response object
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, 'usage_metadata'):
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
                
            return GenerationResult(
                content=response.text,
                model_name=model,
                usage={"input": input_tokens, "output": output_tokens},
                provider="google"
            )
        except Exception as e:
             raise RuntimeError(f"Google Error: {e}")


# ==========================================
# Factory
# ==========================================
class LLMFactory:
    """Factory to create LLM clients based on configuration."""
    
    @staticmethod
    def create_client(provider: str) -> LLMClient:
        provider = provider.lower()
        
        if provider == "ollama":
            return OllamaClient()
            
        elif provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not found in code.")
            return OpenAIClient(api_key=OPENAI_API_KEY)
            
        elif provider == "deepseek":
            if not DEEPSEEK_API_KEY:
                raise ValueError("DEEPSEEK_API_KEY not found in code.")
            return OpenAIClient(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com", provider_name="deepseek")
            
        elif provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not found in code.")
            return AnthropicClient(api_key=ANTHROPIC_API_KEY)
            
        elif provider == "google":
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY not found in code.")
            return GoogleClient(api_key=GOOGLE_API_KEY)
            
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")


# ==========================================
# Main RAG Generator
# ==========================================
class RAGGenerator:
    """Generates answers using retrieved context and configured LLM."""
    
    def __init__(self, provider_override=None, model_override=None):
        self.provider = provider_override or LLM_PROVIDER
        self.model_name = model_override or LLM_MODEL
        try:
            self.client = LLMFactory.create_client(self.provider)
            # print(f"🤖 Initialized RAG Graph with Provider: {self.provider.upper()} | Model: {self.model_name}")
        except ValueError as e:
            print(f"⚠️ Failed to init {self.provider}: {e}")
            self.client = None
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context string."""
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            page_num = chunk['metadata'].get('page_num', 'Unknown')
            chunk_type = chunk['metadata'].get('type', 'text')
            text = chunk['text']
            type_label = " [IMAGE]" if chunk_type == "image_caption" else ""
            context_parts.append(f"[Source {i} - Page {page_num}{type_label}]\n{text}\n")
        
        return "\n".join(context_parts)
    
    def calculate_cost(self, usage: Dict[str, int], model_name: str) -> float:
        """Calculate estimated cost based on config pricing."""
        prices = PRICING.get(model_name, PRICING.get("default", {"input":0, "output":0}))
        
        input_cost = (usage['input'] / 1_000_000) * prices['input']
        output_cost = (usage['output'] / 1_000_000) * prices['output']
        
        return round(input_cost + output_cost, 6)
    
    def generate_answer(self, query: str, context: str) -> GenerationResult:
        """Generate answer using abstract client."""
        if not self.client:
             raise RuntimeError(f"Client for {self.provider} not initialized.")

        prompt = f"""Context:
{context}

Question: {query}

Answer the question based on the context provided. Always cite which source(s) you used."""
        
        return self.client.generate(
            system_prompt=RAG_SYSTEM_PROMPT,
            user_prompt=prompt,
            model=self.model_name
        )
    
    def answer_query(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Full generation pipeline."""
        context = self.format_context(retrieved_chunks)
        start_time = time.time()
        
        result = self.generate_answer(query, context)
        
        latency = round(time.time() - start_time, 2)
        cost = self.calculate_cost(result.usage, result.model_name)
        
        return {
            "query": query,
            "answer": result.content,
            "context": context,
            "num_sources": len(retrieved_chunks),
            "provider": self.provider,
            "model": result.model_name,
            "latency": latency,
            "usage": result.usage,
            "cost": cost
        }
