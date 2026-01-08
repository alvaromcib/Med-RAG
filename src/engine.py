# src/engine.py
import os
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

class MedRAG:
    def __init__(self, collection_name="med_rag_neuroradiology"):
        # Conexión a Qdrant
        self.client = QdrantClient(path="./storage/qdrant_db")
        self.vector_store = QdrantVectorStore(
            client=self.client, 
            collection_name=collection_name,
            enable_hybrid=True
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = None

    def build_index(self, data_path):
        """Procesa PDFs con LlamaParse y los guarda en Qdrant"""
        parser = LlamaParse(result_type="markdown", verbose=True, language="en")
        file_extractor = {".pdf": parser}
        
        print("🚀 Analizando PDFs con LlamaParse...")
        documents = SimpleDirectoryReader(
            input_dir=data_path, 
            recursive=True, 
            file_extractor=file_extractor
        ).load_data()

        print(f"📦 Creando índice para {len(documents)} fragmentos...")
        self.index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=self.storage_context,
            show_progress=True
        )
        return self.index

    def load_index(self):
        """Carga el índice ya existente desde el disco"""
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=self.storage_context
        )
        return self.index

    def get_query_engine(self):
        """Configura el motor de respuesta con streaming"""
        if not self.index:
            self.load_index()
            
        return self.index.as_query_engine(
            vector_store_query_mode="hybrid", 
            similarity_top_k=5,
            streaming=True
        )