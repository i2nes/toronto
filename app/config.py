"""Application configuration with sensible defaults."""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
NOTES_DIR = BASE_DIR / "notes"
PROMPTS_DIR = BASE_DIR / "app" / "prompts"

# Ensure data directories exist
DATA_DIR.mkdir(exist_ok=True)
NOTES_DIR.mkdir(exist_ok=True)

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemma3:12b")  # Use already installed model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")

# RAG parameters (character-based to avoid tokenizer inconsistencies)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2400"))          # ≈600 tokens
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "320"))     # ≈80 tokens
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))         # 0=diverse, 1=relevant
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "1500"))

# Tool security (for Phase 5)
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "10"))
SEARCH_TIMEOUT = float(os.getenv("SEARCH_TIMEOUT", "10.0"))
SEARCH_ALLOWED_DOMAINS = None  # None = all, or set to list
SEARCH_BLOCKED_DOMAINS = []    # Add spam/malicious domains

# UI & assets
STATIC_VERSION = os.getenv("STATIC_VERSION", "1.0.0")

# Database
DB_PATH = DATA_DIR / "assistant.sqlite"
VECTOR_INDEX_PATH = DATA_DIR / "vectors.index"
METADATA_PATH = DATA_DIR / "metadata.json"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
