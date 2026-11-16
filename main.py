# main.py
"""Interactive console: ask a question, get a streamed answer and the pages it comes from."""
import os

from src.config import setup_models
from src.engine import MedRAG


def main():
    if not os.path.exists("./storage/qdrant_db"):
        print("No index found. Run 'python ingest.py' first.")
        return
    setup_models()
    engine = MedRAG().get_query_engine()
    print("\n--- Neuroradiology assistant ready (type 'exit' to quit) ---")
    while True:
        question = input("\nQuestion: ")
        if question.lower() in ("exit", "quit", "q", "salir"):
            break
        response = engine.query(question)
        print("\nAnswer: ", end="", flush=True)
        response.print_response_stream()
        print("\n\nSources:")
        for node in response.source_nodes:
            print(f"- {node.metadata.get('file_name')} (p. {node.metadata.get('page_label')}) [score {node.score:.2f}]")


if __name__ == "__main__":
    main()
