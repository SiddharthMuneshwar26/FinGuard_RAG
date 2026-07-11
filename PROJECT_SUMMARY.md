# FinGuard-RAG Summary

Local RAG over SHARC and GAICF financial-AI papers.

```text
FinGuard-RAG/
├── data/                 # source PDFs
├── notebooks/            # experiments/evaluation
├── src/
│   ├── config.py         # paths, models, chunk/retrieval settings
│   ├── ingest.py         # page-level PDF loading + metadata
│   ├── chunking.py       # recursive 800/150 chunking + chunk IDs
│   ├── embeddings.py     # BGE embeddings + query prefix
│   ├── vectorstore.py    # FAISS persistence
│   ├── retrieval.py      # top-10 dense retrieval
│   ├── reranker.py       # CrossEncoder top-4 reranking
│   ├── prompts.py        # grounding and citation prompts
│   ├── llm.py            # local Ollama client
│   ├── rag_chain.py      # end-to-end class-based pipeline
│   ├── build_index.py    # index CLI
│   └── cli.py            # question-answering CLI
├── vectorstore/          # generated 136-vector FAISS index
├── requirements.txt
└── .gitignore
```

Pipeline: PDF → recursive chunks → `BAAI/bge-base-en-v1.5` → FAISS (10) → `cross-encoder/ms-marco-MiniLM-L-6-v2` (4) → Ollama `qwen3:8b` → grounded answer with page citations.

Metadata retained: title, filename, one-based page, document ID and stable chunk ID (`SHARC-p9-c1`). Backend imports, indexing, retrieval, reranking, Ollama generation and citation extraction pass. Streamlit is deferred.

```powershell
.\.venv\Scripts\python.exe -m src.build_index --force
.\.venv\Scripts\python.exe -m src.cli "What is SHAP?"
```
