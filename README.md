# 🛡️ FinGuard-RAG

A retrieval-augmented question-answering assistant over two 2026 research papers on
**AI governance and interpretability in bank risk management**:

- **GAICF** — *Governing Generative AI Across Financial Institutions: An SR 26-2-Compatible
  Framework for Generative AI Risk Control* (arXiv:2607.04103)
- **SHARC** — *SHAP-Based Interpretability in ML Risk Models for Regulatory Capital under
  ICAAP and CCAR* (arXiv:2607.05484)

The two papers are complementary — one governs generative-AI *outputs* in regulated
workflows, the other makes ML *risk-capital models* auditable — so together they form a
coherent niche: **trustworthy, auditable AI in financial regulation.**

## Pipeline

```
PDF ─▶ section-aware chunking (+metadata) ─▶ embeddings ─▶ Chroma vector store
                                                                │
question ─▶ dense search (top_k=20) ─▶ cross-encoder rerank (k=5) ─▶ grounded LLM ─▶ answer + citations
```

| Stage           | Choice                              | Why |
|-----------------|-------------------------------------|-----|
| Parsing         | PyMuPDF                             | fast, keeps page numbers |
| Chunking        | section-aware, word-windowed, overlap | metadata drives citations |
| Embeddings      | `bge-small-en-v1.5`                 | strong + laptop-CPU friendly |
| Vector store    | ChromaDB (persistent)              | zero-server, metadata filtering |
| Retrieval       | dense + cross-encoder rerank        | retrieve-then-rerank = real RAG |
| Generation      | Claude / OpenAI / extractive        | grounded, refuses when unsupported |
| Citations       | chunk-id → paper·section·page       | every claim is verifiable |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r src/requirements.txt

python scripts/download_papers.py     # fetch the two PDFs into data/
python -m src.ingest                  # build the vector store  (~1 min on CPU)
```

Set a key for synthesized answers (optional — omit to run in extractive mode):

```bash
export ANTHROPIC_API_KEY=...          # or: export LLM_PROVIDER=none
```

Run with local Ollama:

```bash
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=qwen3:8b
streamlit run src/app.py
```

On Windows PowerShell:

```powershell
$env:LLM_PROVIDER = 'ollama'
$env:OLLAMA_MODEL = 'qwen3:8b'
streamlit run src/app.py
```

If your local Ollama model uses a different alias, set `OLLAMA_MODEL` accordingly.

## Run

```bash
python -m src.retrieve                          # retrieval smoke test
python -m src.chatbot "What gap does SR 26-2 leave for generative AI?"
streamlit run src/app.py                        # chat UI
```

## Project layout

```
FinGuard-RAG/
├── data/                 # the two source PDFs
├── notebooks/
│   └── experiments.ipynb # chunk inspection + evaluation harness
├── scripts/
│   └── download_papers.py
└── src/
    ├── config.py         # every knob in one place
    ├── ingest.py         # parse → chunk → embed → store
    ├── retrieve.py       # dense search + rerank
    ├── chatbot.py        # grounded generation + citations
    ├── app.py            # Streamlit UI
    ├── vectorstore/      # persisted Chroma index (generated)
    └── requirements.txt
```

## Evaluation

`notebooks/experiments.ipynb` contains a starter harness: a small gold set of Q&A pairs,
retrieval metrics (recall@k, MRR), and a groundedness check. Reporting these — with an
honest "what failed and how I fixed it" note — is what turns a working demo into a
convincing portfolio piece.

## Possible extensions
- Hybrid search (add BM25/sparse; swap in `bge-m3` for dense+sparse in one model)
- Swap Chroma → Qdrant for native hybrid + a more production feel
- Expand the corpus (SR 26-2 text, SR 11-7, Basel/CCAR guidance) so retrieval scales
- Streaming responses + conversation memory in the UI
