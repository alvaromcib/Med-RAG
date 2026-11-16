# src/config.py
"""Model settings. Everything runs locally: the LLM through Ollama and the embedding model on CPU."""
import os

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

LLM_MODEL = os.getenv("MEDRAG_LLM", "qwen2.5:7b")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def setup_models():
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=900.0, temperature=0,
                          context_window=8192)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.chunk_size = 512          # tokens per chunk
    Settings.chunk_overlap = 50
