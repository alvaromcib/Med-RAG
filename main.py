# main.py
import os
from src.config import setup_models
from src.engine import MedRAG

def main():
    setup_models()
    rag = MedRAG()
    
    # Comprobar si la base de datos existe
    if not os.path.exists("./storage/qdrant_db"):
        print("❌ No se encontró base de datos. Ejecuta primero 'python ingest.py'")
        return

    query_engine = rag.get_query_engine()
    
    print("\n--- 🧠 Asistente de Neuroradiología Listo ---")
    while True:
        question = input("\nPregunta (o 'salir'): ")
        if question.lower() in ['salir', 'exit', 'q']: break
        
        # 1. Obtenemos la respuesta (ahora es un objeto stream)
        response = query_engine.query(question)
        
        print("\nRespuesta: ", end="", flush=True)
        # 2. Imprimimos palabra por palabra según se generan
        response.print_response_stream()
        print("\n") # Salto de línea al terminar
        
        print("-" * 30)
        
        print("\n📚 Fuentes:")
        for node in response.source_nodes:
            name = node.metadata.get('file_name', 'Desconocido')
            page = node.metadata.get('page_label', 'N/A')
            print(f"- {name} (Pág. {page}) [Score: {node.score:.2f}]")

if __name__ == "__main__":
    main()