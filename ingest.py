# ingest.py
from src.config import setup_models
from src.engine import MedRAG

if __name__ == "__main__":
    setup_models()
    MedRAG().build_index("./data")
    print("Index ready in ./storage/qdrant_db")
