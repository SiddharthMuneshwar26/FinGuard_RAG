<div align="center">

# 🛡️ FinGuard AI

### Local Retrieval-Augmented Generation for Financial AI Governance and Explainable Risk Models

[![Python](https://img.shields.io/badge/Python-3.13_tested-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Chat_UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_Inference-111111)](https://ollama.com/)
[![Qwen3](https://img.shields.io/badge/LLM-Qwen3_8B-6F42C1)](https://ollama.com/library/qwen3)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-BGE_Embeddings-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/BAAI/bge-base-en-v1.5)
[![FAISS](https://img.shields.io/badge/Vector_Search-FAISS-0467DF)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/Code_License-MIT-green.svg)](LICENSE)

**FinGuard AI** is an end-to-end, locally hosted RAG assistant that retrieves, reranks and explains research on financial AI governance, model-risk management, regulatory capital and explainable machine learning.

It combines dense retrieval, CrossEncoder reranking, grounded prompting, local LLM generation and page-level citations in a Streamlit interface.

[Features](#-key-features) •
[Architecture](#-system-architecture) •
[Installation](#-installation-and-setup) •
[Evaluation](#-retrieval-evaluation) •
[Usage](#-usage)

</div>

---

## 📌 Overview

Financial institutions increasingly use machine learning and generative AI in high-stakes workflows. These systems must be accurate, explainable, auditable and governed with appropriate controls.

FinGuard AI provides a focused research assistant over two complementary financial-AI papers:

1. **SHARC: SHAP-Based Interpretability in Machine Learning Risk Models for Regulatory Capital under ICAAP and CCAR**
2. **Governing Generative AI Across Financial Institutions: An SR 26-2-Compatible Framework for Generative AI Risk Control**

The assistant answers questions using retrieved evidence from the indexed documents rather than relying only on the language model's internal knowledge.

<p align="center">
  <img src="./assets/finguard-ui-overview.png"
       width="92%"
       alt="FinGuard AI Streamlit interface">
</p>

<p align="center">
  <em>FinGuard AI's local Streamlit interface and model configuration panel.</em>
</p>

### Example questions

```text
What is SHAP and SHARC?

Why is interpretability important for regulatory capital models?

What risks arise when banks deploy generative AI?

What controls should financial institutions apply to generative AI systems?
 
```
### Example grounded response

<p align="center">
  <img src="./assets/finguard-answer-with-citations.png"
       width="92%"
       alt="FinGuard AI answer with page-level citations">
</p>

<p align="center">
  <em>A grounded response generated from retrieved SHARC evidence with page and chunk citations.</em>
</p>

---

## ✨ Key Features

- **Fully local generation** with Ollama and Qwen3 8B
- **Dense semantic retrieval** using BAAI BGE embeddings
- **FAISS vector search** over document chunks
- **CrossEncoder reranking** to improve result ordering
- **Top-10 retrieval → top-4 final context** pipeline
- **Grounded prompting** that restricts answers to retrieved evidence
- **Page-level citations** with document title, page and stable chunk ID
- **Retrieved-context inspection** for transparency and debugging
- **Streamlit chat interface** with chat history and clear-chat controls
- **CLI interface** for fast backend testing
- **Metadata-aware ingestion** with titles, filenames, page numbers and document IDs
- **Retrieval evaluation** using Hit@k and Mean Reciprocal Rank on a curated gold set
- **Local-first design** with no mandatory paid LLM API

---

## 🧠 System Architecture

```mermaid
flowchart LR
    A[Financial-AI PDFs] --> B[PDF Ingestion]
    B --> C[Page Metadata]
    C --> D[Recursive Chunking]
    D --> E[BGE Embeddings]
    E --> F[(FAISS Index)]

    Q[User Question] --> G[Query Embedding]
    G --> F
    F --> H[Top-10 Candidates]
    H --> I[CrossEncoder Reranker]
    I --> J[Top-4 Evidence Chunks]
    J --> K[Grounded Prompt]
    Q --> K
    K --> L[Ollama + Qwen3 8B]
    L --> M[Answer + Page Citations]
    M --> N[Streamlit / CLI]
```

### Query-time flow

```text
Question
   ↓
BAAI/bge-base-en-v1.5 query embedding
   ↓
FAISS dense retrieval — top 10
   ↓
cross-encoder/ms-marco-MiniLM-L-6-v2 reranking
   ↓
Best 4 evidence chunks
   ↓
Grounded prompt
   ↓
Ollama running Qwen3 8B
   ↓
Answer with document and page citations
```

---

## 🛠️ Technology Stack

| Layer | Tool / Model | Purpose |
|---|---|---|
| Language | Python | Core application and pipeline |
| Experimentation | Jupyter Notebook | Chunking, retrieval and evaluation experiments |
| Document parsing | pypdf | Page-level PDF text extraction |
| Document schema | LangChain Core | Document objects and metadata |
| Text splitting | LangChain Text Splitters | Recursive chunking |
| Embeddings | `BAAI/bge-base-en-v1.5` | Dense semantic representations |
| Embedding integration | LangChain Hugging Face | Connects BGE embeddings to the pipeline |
| Vector database | FAISS CPU | Local similarity search and persistence |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Query-passage relevance scoring |
| Reranker library | Sentence Transformers | CrossEncoder inference |
| Local model runtime | Ollama | Local LLM serving |
| Generator | `qwen3:8b` | Grounded natural-language answers |
| User interface | Streamlit | Interactive research chat |
| HTTP health/error handling | Requests | Ollama connection checks |
| Version control | Git and GitHub | Source control and portfolio hosting |

---

## 🔍 Retrieval Design

### 1. Page-level ingestion

Each PDF is loaded page by page. The pipeline preserves metadata including:

```python
{
    "title": "...",
    "filename": "...",
    "document_id": "...",
    "page": 9,
    "chunk_id": "SHARC-p9-c1"
}
```

This metadata supports traceable citations in generated answers.

### Evidence transparency

Users can inspect the exact reranked passages supplied to Qwen3, including
the source paper, page number, chunk ID and reranker score.

<p align="center">
  <img src="./assets/finguard-retrieved-context.png"
       width="420"
       alt="FinGuard AI retrieved evidence viewer">
</p>

### 2. Recursive chunking

The current indexing configuration uses:

```text
Chunk size:    800 characters
Chunk overlap: 150 characters
```

The overlap helps preserve meaning when an explanation crosses a chunk boundary.

### 3. Dense retrieval

Both chunks and user questions are embedded using:

```text
BAAI/bge-base-en-v1.5
```

The vectors are stored in a persistent local FAISS index. With the current two-paper knowledge base and chunking configuration, the pipeline produces **136 document vectors**.

### 4. CrossEncoder reranking

FAISS retrieves the top 10 candidates. A CrossEncoder then jointly scores each question-passage pair:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Only the top 4 passages are sent to Qwen3. This provides broader initial recall while keeping the final context focused.

### 5. Grounded generation

The LLM is instructed to:

- answer only from the retrieved context;
- avoid unsupported claims;
- state when the documents do not contain enough information;
- cite the evidence used;
- distinguish proposals, findings and general practices;
- avoid presenting research explanations as investment advice.

---

## 📁 Project Structure

```text
FinGuard-RAG/
├── app.py                     # Streamlit chat interface
├── assets/                     # README screenshots and evaluation visuals
├── data/                      # Source research papers
├── notebooks/                 # Experiments and retrieval evaluation
├── scripts/                   # Supporting utility scripts
├── src/
│   ├── __init__.py
│   ├── config.py              # Paths, model names and retrieval settings
│   ├── ingest.py              # PDF extraction and metadata
│   ├── chunking.py            # Recursive splitting and chunk IDs
│   ├── embeddings.py          # BGE embedding configuration
│   ├── vectorstore.py         # FAISS creation, loading and persistence
│   ├── retrieval.py           # Top-k dense retrieval
│   ├── reranker.py            # CrossEncoder reranking
│   ├── prompts.py             # Grounding and citation instructions
│   ├── llm.py                 # Ollama client
│   ├── rag_chain.py           # End-to-end RAG orchestration
│   ├── build_index.py         # Index-building CLI
│   └── cli.py                 # Terminal question-answering interface
├── PROJECT_SUMMARY.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

Generated indexes, serialized chunks, Python caches and the virtual environment are intentionally excluded from Git.

---

## 🚀 Installation and Setup

### Prerequisites

- Windows, Linux or macOS
- Python 3.13 tested
- Git
- Ollama
- Enough memory to run Qwen3 8B locally

> The project works with CPU-based FAISS and PyTorch. Ollama can use supported GPU acceleration independently.

### 1. Clone the repository

```powershell
git clone https://github.com/SiddharthMuneshwar26/FinGuard_RAG.git
cd FinGuard_RAG
```

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
```

Packages can be installed directly through the environment interpreter without activating it:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

#### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install and prepare Ollama

Install Ollama from its official website, then pull the local model:

```powershell
ollama pull qwen3:8b
```

Test it:

```powershell
ollama run qwen3:8b
```

Exit the interactive Ollama session with:

```text
/bye
```

Keep the Ollama service running while using FinGuard AI.

### 4. Add the source papers

Place the licensed PDFs inside:

```text
data/
```

The default knowledge base expects the SHARC and financial GenAI-governance papers.

### 5. Build the FAISS index

```powershell
.\.venv\Scripts\python.exe -m src.build_index --force
```

This performs:

```text
PDF loading
→ metadata creation
→ recursive chunking
→ BGE embedding generation
→ FAISS index creation
→ local persistence
```

---

## 💬 Usage

### Streamlit interface

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit will display a local URL, normally:

```text
http://localhost:8501
```

The interface includes:

- conversational question answering;
- source citations;
- document page and chunk metadata;
- reranker scores;
- expandable retrieved context;
- model and retrieval configuration;
- chat-history clearing;
- actionable errors for missing indexes, Ollama connectivity and unavailable models.

### Command-line interface

Ask a single question:

```powershell
.\.venv\Scripts\python.exe -m src.cli "What is SHAP?"
```

Additional examples:

```powershell
.\.venv\Scripts\python.exe -m src.cli "Why is model interpretability important under ICAAP?"
```

```powershell
.\.venv\Scripts\python.exe -m src.cli "What controls should banks apply to generative AI?"
```

### Rebuild after changing documents or chunk settings

```powershell
.\.venv\Scripts\python.exe -m src.build_index --force
```

---

## 📊 Retrieval Evaluation

FinGuard includes notebook-based retrieval evaluation using a manually curated gold question set.

### Metrics

#### Hit@k

Hit@k checks whether at least one expected relevant chunk or document appears among the first `k` results.

```text
Hit@k = successful queries at k / total queries
```

#### Mean Reciprocal Rank

MRR rewards systems that place the first relevant result near the top:

```text
MRR = average(1 / rank of first relevant result)
```

### Evaluation configuration

| Component | Setting |
|---|---|
| Embedding model | `BAAI/bge-base-en-v1.5` |
| Initial retrieval | FAISS top 10 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Final context | Top 4 |
| Evaluation set | Manually curated gold questions |
| Metrics | Hit@k and MRR |

### Results

| Paper | Questions | Hit@5 | MRR |
|---|---:|---:|---:|
| GAICF | 5 | 100% | 1.000 |
| SHARC | 8 | 100% | 1.000 |
| **Overall** | **13** | **100%** | **1.000** |

The system retrieved a relevant result within the top five for every
evaluation question. An MRR of 1.000 indicates that the first relevant
result appeared at rank one for all 13 questions.

> **Evaluation scope:** These results were measured on a manually curated
> 13-question gold set across the current two-document knowledge base. They
> should not be interpreted as general-domain retrieval performance.
<p align="center">
  <img src="./assets/retrieval-evaluation.png"
       width="720"
       alt="Retrieval evaluation showing 100 percent Hit at 5 and 1.000 MRR">
</p>

### Gold evaluation set

The evaluation uses 13 manually curated domain questions: five targeting
the GAICF paper and eight targeting the SHARC paper.

<p align="center">
  <img src="./assets/gold-set-questions.png"
       width="820"
       alt="Thirteen manually curated gold-set questions used for retrieval evaluation">
</p>


### Qualitative reranker comparison

The following example compares the initial FAISS cosine-similarity results
with the final ranking produced by the CrossEncoder reranker.

<table>
  <tr>
    <th>FAISS retrieval only</th>
    <th>FAISS + CrossEncoder reranking</th>
  </tr>
  <tr>
    <td>
      <img src="./assets/retrieval-without-reranker.png"
           alt="Retrieval results without reranking"
           width="100%">
    </td>
    <td>
      <img src="./assets/retrieval-with-reranker.png"
           alt="Retrieval results after CrossEncoder reranking"
           width="100%">
    </td>
  </tr>
</table>

FAISS first retrieves candidates using embedding similarity. The
CrossEncoder then jointly evaluates each query–passage pair and assigns a
more precise relevance score before selecting the final evidence supplied
to the LLM.

This example illustrates query-specific rescoring and candidate reordering.
The quantitative Hit@5 and MRR results are reported separately above.

---

## 💡 Why These Design Choices?

### Why BGE embeddings?

BGE provides strong semantic retrieval for technical English text while remaining practical for local execution.

### Why FAISS?

FAISS is lightweight, fast, local and appropriate for a compact research-document collection. It avoids requiring an external vector-database service.

### Why retrieve 10 and rerank to 4?

Dense retrieval maximizes recall, while the CrossEncoder provides more precise query-passage relevance scoring. Passing only the best four chunks reduces irrelevant context and prompt size.

### Why Qwen3 8B?

Qwen3 8B provides capable local instruction following while remaining usable through consumer-oriented local inference tools such as Ollama.

### Why local inference?

Local inference provides:

- data privacy;
- no mandatory API key;
- reproducible model selection;
- offline experimentation after models are downloaded;
- direct experience with local LLM serving.

---

## 🧪 Reliability and Error Handling

The application handles common failure cases including:

- missing FAISS index;
- Ollama service not running;
- configured Ollama model not installed;
- empty model response;
- unexpected retrieval or generation errors.

Unsupported questions should produce an explicit insufficient-context response rather than a fabricated answer.

Example:

```text
Question:
What was Tesla's revenue in 2025?

Expected behavior:
The provided documents do not contain enough information to answer this.
```

---

## ⚠️ Limitations

- The knowledge base currently contains a small, specialized research collection.
- PDF extraction may not perfectly preserve complex tables, equations or multi-column layouts.
- Retrieval evaluation depends on the quality and coverage of the manually created gold set.
- Local Qwen3 8B inference speed depends on the user's CPU, GPU, memory and Ollama configuration.
- The assistant explains indexed research and is not a substitute for legal, regulatory, financial or investment advice.
- A local Ollama server cannot be accessed directly by a public cloud deployment unless Ollama is also hosted in that environment.

## 📄 License and Document Attribution

### Source documents

The research papers included in the knowledge base remain the intellectual property of their respective authors and are redistributed under their stated Creative Commons licences.


#### SHARC: SHAP-Based Interpretability in Machine Learning Risk Models for Regulatory Capital under ICAAP and CCAR

- **Author:** Ujjwala Vadrevu
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Original source:** [arXiv:2607.05484](https://arxiv.org/abs/2607.05484)
- **Changes:** The PDF itself is unmodified. FinGuard extracts, chunks and embeds its text for retrieval.

#### Governing Generative AI Across Financial Institutions: An SR 26-2-Compatible Framework for Generative AI Risk Control

- **Author(s):** Yiqing Wang
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Original source:** [arXiv:2607.04103](https://arxiv.org/abs/2607.04103)
- **Changes:** The PDF itself is unmodified. FinGuard extracts, chunks and embeds its text for retrieval.

Creative Commons attribution applies to the papers and does not automatically determine the licence of this repository's source code.

### Source code

The FinGuard AI source code is released under the [MIT License](LICENSE).

---

## 👤 Author

**Siddharth Muneshwar**

- GitHub: [@SiddharthMuneshwar26](https://github.com/SiddharthMuneshwar26)
- Repository: [FinGuard_RAG](https://github.com/SiddharthMuneshwar26/FinGuard_RAG)

---

## 🌟 Support

If this project helped you understand production-style RAG systems, consider starring the repository.

Contributions, suggestions and issue reports are welcome.

<div align="center">

**Built with local retrieval, reranking and grounded generation.**

🛡️ **FinGuard AI**

</div>
