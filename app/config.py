"""
Central configuration. Everything is overridable via environment variables so the
same image can be deployed to Render / Fly / Railway / HF Spaces without code changes.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)

# --- LLM provider -----------------------------------------------------------
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "22"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1500"))

# --- Data ---------------------------------------------------------------
CATALOG_PATH = Path(os.environ.get("CATALOG_PATH", BASE_DIR / "data" / "catalog.json"))

# --- Retrieval ------------------------------------------------------------
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "25"))

# --- API behaviour ----------------------------------------------------------
MAX_RECOMMENDATIONS = 10
MIN_RECOMMENDATIONS = 1
MAX_CONVERSATION_TURNS = 8  # evaluator cap; we defend against runaway history too

# --- Safety -------------------------------------------------------------
REQUEST_HARD_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_HARD_TIMEOUT_SECONDS", "28"))