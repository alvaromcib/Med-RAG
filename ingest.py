# ingest.py
from src.config import setup_models
from src.engine import MedRAG

def run():
    setup_models()
    rag = MedRAG()
    rag.build_index("./data")
    print("✅ Base de datos médica actualizada.")

if __name__ == "__main__":
    run()