# src/engine.py
"""Ingestion and query engine. PDFs are parsed locally with PyMuPDF (tables kept as Markdown), split into
512-token chunks that keep file, folder and page, and stored in a local Qdrant collection with dense
(bge-small) and sparse (BM25) vectors for hybrid search."""
from pathlib import Path

import pymupdf4llm
from llama_index.core import Document, PromptTemplate, StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

NOT_FOUND = "I don't know based on the provided documents."
TEMPLATE = PromptTemplate(
    "You are a specialised medical assistant in Neuroradiology and Neuro-oncology, supporting clinicians.\n"
    "Context information is below, each passage with its source:\n"
    "---------------------\n{context_str}\n---------------------\n"
    "Answer the query using ONLY the context, in 1-4 technical sentences. Give the specific figure, grade or "
    f"recommendation. If the context does not contain the answer, reply exactly: \"{NOT_FOUND}\"\n"
    "Query: {query_str}\nAnswer: ")


def load_pdfs(data_path):
    """One Document per page, with the metadata needed to cite it."""
    docs = []
    for pdf in sorted(Path(data_path).rglob("*.pdf")):
        for page in pymupdf4llm.to_markdown(str(pdf), page_chunks=True, show_progress=False):
            text = page["text"].strip()
            if len(text) < 50:
                continue
            docs.append(Document(text=text, metadata={
                "file_name": pdf.name, "category": pdf.parent.name, "page_label": str(page["metadata"].get("page_number", page["metadata"].get("page")))},
                excluded_embed_metadata_keys=["category"], excluded_llm_metadata_keys=["category"]))
    return docs


class MedRAG:
    def __init__(self, collection_name="med_rag_neuroradiology", path="./storage/qdrant_db"):
        self.client = QdrantClient(path=path)
        # Hybrid: dense bge-small vectors + sparse BM25 (fastembed "Qdrant/bm25", no neural net, low RAM).
        self.vector_store = QdrantVectorStore(client=self.client, collection_name=collection_name,
                                              enable_hybrid=True, fastembed_sparse_model="Qdrant/bm25",
                                              batch_size=16)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = None

    def build_index(self, data_path):
        docs = load_pdfs(data_path)
        print(f"Indexing {len(docs)} pages ...")
        self.index = VectorStoreIndex.from_documents(docs, storage_context=self.storage_context,
                                                     show_progress=True)
        return self.index

    def load_index(self):
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        return self.index

    def retriever(self, mode="hybrid", k=6):
        if not self.index:
            self.load_index()
        if self.client.count(self.vector_store.collection_name).count == 0:
            raise SystemExit("The index is empty: delete storage/ and run 'python ingest.py' again.")
        return self.index.as_retriever(vector_store_query_mode=mode, similarity_top_k=k,
                                       sparse_top_k=k * 2 if mode == "hybrid" else None)

    def get_query_engine(self, mode="hybrid", k=6, streaming=True):
        if not self.index:
            self.load_index()
        return self.index.as_query_engine(vector_store_query_mode=mode, similarity_top_k=k,
                                          sparse_top_k=k * 2 if mode == "hybrid" else None,
                                          streaming=streaming, response_mode="compact",
                                          text_qa_template=TEMPLATE)
