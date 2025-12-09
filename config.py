"""
Configuration module for Production RAG system.
Supports multiple providers: ollama, openai, anthropic, google, deepseek.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# ==========================================
# LLM Provider Configuration
# ==========================================
# Options: "ollama", "openai", "anthropic", "google", "deepseek"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# Model Names
MODELS = {
    "ollama": {
        "text": "minicpm-v",
        "vision": "minicpm-v",
        "embedding": "text-embedding-3-small" 
    },
    "openai": {
        "text": "gpt-4o-mini",
        "vision": "gpt-4o-mini",
        "embedding": "text-embedding-3-small"
    },
    "anthropic": {
        "text": "claude-3-5-sonnet-20240620",
        "vision": "claude-3-5-sonnet-20240620",
        "embedding": None
    },
    "google": {
        "text": "gemini-2.0-flash",
        "vision": "gemini-2.0-flash",
        "embedding": "models/text-embedding-004"
    },
    "deepseek": {
        "text": "deepseek-chat", # Check exact model name in deepseek docs
        "vision": None,
        "embedding": None
    }
}

# Pricing (USD per 1 Million Tokens) - Updated as of Dec 2024
# Format: {model_name: {input: float, output: float}}
PRICING = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    
    # Anthropic
    "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    
    # Google (Gemini) - varying/free tiers exist, putting std pay-as-you-go
    "gemini-1.5-pro": {"input": 1.25, "output": 3.75}, # Approx < 128k context
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30}, 
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40}, # Estimated
    
    # DeepSeek
    "deepseek-chat": {"input": 0.14, "output": 0.28}, # Verification needed
    
    # Local (Free)
    "minicpm-v": {"input": 0.0, "output": 0.0},
    "llama3.2": {"input": 0.0, "output": 0.0},
    "gemma3": {"input": 0.0, "output": 0.0}
}
PRICING_DEFAULTS = {"input": 0.0, "output": 0.0}

# Active Models based on Provider
LLM_MODEL = MODELS[LLM_PROVIDER]["text"]
CAPTION_MODEL = MODELS[LLM_PROVIDER]["vision"] or MODELS["ollama"]["vision"]

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# ==========================================
# Application Settings
# ==========================================

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
TOP_K_RETRIEVAL = 10
TOP_K_FINAL = 5

# Image settings
MIN_IMAGE_SIZE_BYTES = 2000
IMAGE_CAPTION_PROMPT = """Describe this image in detail, focusing on:
1. What the image shows
2. Any text visible in the image
3. The context or purpose of this visual element

Keep the description concise but informative."""

# Generation settings
RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Always cite the source of your information when answering.
If you cannot find the answer in the context, say so clearly."""
