"""Central configuration for the FinGuard-RAG backend."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"

PAPER_METADATA: dict[str, dict[str, str]] = {
    "sharc.pdf": {
        "document_id": "SHARC",
        "title": "SHARC: SHAP-Based Interpretability for Regulatory Capital",
    },
    "genai_governance.pdf": {
        "document_id": "GAICF",
        "title": "Governing GenAI Across Financial Institutions (SR 26-2)",
    },
}

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DEVICE = os.getenv("FINGUARD_EMBEDDING_DEVICE", "cpu")
EMBEDDING_NORMALIZE = True
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# This is the exact reranker used successfully in notebooks/experiments.ipynb.
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_DEVICE = os.getenv("FINGUARD_RERANKER_DEVICE", "cpu")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

RETRIEVAL_K = 10
RERANK_TOP_N = 4

OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

ABSTENTION_MESSAGE = "That isn't covered in the provided documents."
