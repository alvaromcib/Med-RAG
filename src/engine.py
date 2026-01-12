# src/engine.py
import os
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, PromptTemplate
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

    def get_query_engine(self):
        if not self.index:
            self.load_index()

        # Usamos similarity_top_k=10 aprovechando tus 32GB de RAM
        return self.index.as_query_engine(
            vector_store_query_mode="hybrid", 
            similarity_top_k=10, 
            streaming=True,
            response_mode="compact", # 'tree_sumarize'
            text_qa_template=self._get_medical_template()
        )

    def _get_medical_template(self):
        return PromptTemplate(
            "You are a specialized medical assistant in Neuroradiology and Neuro-oncology.\n"
            "Context information is below:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query in a professional and technical manner.\n"
            "If the answer is not in the context, say that you don't know.\n"
            "Query: {query_str}\n"
            "Answer: "
        )