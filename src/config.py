# src/config.py
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def setup_models():
    # 1. Configurar el LLM (El que genera la respuesta)
    Settings.llm = Ollama(model="llama3.2", request_timeout=300.0)
    
    # 2. Configurar el modelo de Embeddings (El que traduce a números)
    # BGE es uno de los mejores modelos open-source para esto
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    
    # 3. Tamaño del trozo de texto (Chunk)
    # 512 tokens es un estándar equilibrado para medicina
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50