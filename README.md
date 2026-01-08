# Med-RAG: Neuroradiology Clinical Assistant

Medical RAG system for diagnostic support in Neuroradiology using LlamaIndex, Qdrant, and Ollama.

## Setup
1. **Models**: Install [Ollama](https://ollama.com/) and run `ollama pull llama3.2`.
2. **Environment**:
   - Create a `.env` file with `LLAMA_CLOUD_API_KEY=your_key`.
   - Install dependencies: `pip install -r requirements.txt`.
3. **Ingestion**:
   - Place your medical PDFs in `data/`.
   - Run `python ingest.py`.
4. **Chat**:
   - Run `python main.py`.

## Stack
- **Framework**: LlamaIndex
- **Vector DB**: Qdrant (Hybrid Search)
- **Parser**: LlamaParse (Markdown/Table extraction)
- **LLM**: Llama 3.2 (Local via Ollama)