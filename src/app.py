
from __future__ import annotations

from collections import Counter
from datetime import datetime
from statistics import mean
from typing import Any

import pandas as pd
import requests
import streamlit as st
from langchain_core.documents import Document

from src.config import (
    EMBEDDING_MODEL_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_NAME,
    RERANKER_MODEL_NAME,
    RERANK_TOP_N,
    RETRIEVAL_K,
)
from src.rag_chain import FinGuardRAG


APP_TITLE = "FinGuard AI"
SUGGESTED_QUESTIONS = [
    "What is SHAP and why is it useful for financial risk models?",
    "What risks arise when banks deploy generative AI?",
    "How does SHARC support regulatory capital model governance?",
    "What controls should financial institutions apply to generative AI?",
]

CUSTOM_CSS = """
<style>
.stApp {
    background:
        radial-gradient(circle at 8% 4%, rgba(45, 212, 191, .11), transparent 28%),
        radial-gradient(circle at 94% 0%, rgba(59, 130, 246, .12), transparent 28%),
        linear-gradient(180deg, #07111f 0%, #0a1423 46%, #08101c 100%);
}
[data-testid="stHeader"] {
    background: rgba(7, 17, 31, .72);
    backdrop-filter: blur(12px);
}
.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #081523 0%, #0b1a2a 100%);
    border-right: 1px solid rgba(137, 176, 214, .12);
}
.hero {
    position: relative;
    overflow: hidden;
    padding: 2rem 2.2rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(108, 214, 190, .22);
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(18, 51, 76, .97), rgba(11, 31, 51, .95));
    box-shadow: 0 22px 65px rgba(0, 0, 0, .28);
}
.hero::after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    border-radius: 50%;
    right: -70px;
    top: -120px;
    background: rgba(45, 212, 191, .12);
}
.eyebrow {
    color: #75ead5;
    font-size: .76rem;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
    margin-bottom: .55rem;
}
.hero-title {
    color: #f5fbff;
    font-size: clamp(2rem, 4vw, 3.45rem);
    line-height: 1.02;
    font-weight: 850;
    letter-spacing: -.045em;
    margin: 0;
}
.hero-copy {
    max-width: 780px;
    color: #b9cde0;
    font-size: 1rem;
    line-height: 1.65;
    margin-top: .85rem;
    margin-bottom: 1.15rem;
}
.badge-row { display: flex; flex-wrap: wrap; gap: .55rem; }
.badge {
    display: inline-flex;
    align-items: center;
    padding: .42rem .72rem;
    border: 1px solid rgba(169, 214, 236, .18);
    border-radius: 999px;
    color: #d7e8f5;
    background: rgba(5, 16, 28, .42);
    font-size: .76rem;
    font-weight: 650;
}
.panel-card {
    border: 1px solid rgba(145, 180, 212, .14);
    border-radius: 20px;
    padding: 1.1rem 1.15rem;
    background: linear-gradient(145deg, rgba(14, 32, 51, .92), rgba(9, 24, 40, .92));
    box-shadow: 0 14px 40px rgba(0, 0, 0, .16);
    margin-bottom: .8rem;
}
.panel-label {
    color: #7fe3d1;
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: .35rem;
}
.panel-title { color: #eef8ff; font-size: 1rem; font-weight: 760; margin-bottom: .35rem; }
.panel-copy { color: #9fb6c9; font-size: .86rem; line-height: 1.5; }
.pipeline { display: flex; align-items: center; flex-wrap: wrap; gap: .42rem; margin-top: .75rem; }
.pipeline-step {
    color: #dcecf7;
    border: 1px solid rgba(111, 208, 190, .18);
    background: rgba(13, 42, 59, .76);
    border-radius: 9px;
    padding: .34rem .55rem;
    font-size: .71rem;
    font-weight: 650;
}
.pipeline-arrow { color: #4ed9bd; opacity: .85; }
[data-testid="stMetric"] {
    border: 1px solid rgba(145, 180, 212, .14);
    border-radius: 17px;
    padding: .82rem .9rem;
    background: linear-gradient(145deg, rgba(14, 33, 53, .94), rgba(8, 23, 38, .94));
    box-shadow: 0 12px 28px rgba(0, 0, 0, .13);
}
[data-testid="stChatMessage"] {
    border: 1px solid rgba(145, 180, 212, .13);
    border-radius: 19px;
    padding: .42rem .6rem;
    background: rgba(10, 27, 44, .66);
    box-shadow: 0 10px 32px rgba(0, 0, 0, .11);
    margin-bottom: .72rem;
}
[data-testid="stChatInput"] { border-color: rgba(77, 218, 190, .42); }
.stButton > button {
    border-radius: 13px;
    border: 1px solid rgba(106, 196, 181, .22);
    background: linear-gradient(145deg, rgba(16, 45, 62, .96), rgba(11, 34, 52, .96));
    color: #e9f8ff;
    transition: all .18s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    border-color: rgba(85, 232, 201, .65);
    color: #fff;
    box-shadow: 0 10px 24px rgba(29, 191, 160, .12);
}
.stTabs [data-baseweb="tab-list"] { gap: .45rem; }
.stTabs [data-baseweb="tab"] {
    border-radius: 11px;
    padding-left: .8rem;
    padding-right: .8rem;
    background: rgba(13, 33, 51, .72);
}
[data-testid="stExpander"] {
    border: 1px solid rgba(145, 180, 212, .13);
    border-radius: 14px;
    background: rgba(10, 27, 44, .58);
}
.source-chip {
    display: inline-block;
    margin: .15rem .25rem .15rem 0;
    padding: .3rem .55rem;
    border-radius: 999px;
    color: #cae9f3;
    background: rgba(31, 116, 137, .18);
    border: 1px solid rgba(95, 197, 190, .18);
    font-size: .72rem;
}
.status-dot {
    display: inline-block;
    width: .58rem;
    height: .58rem;
    border-radius: 50%;
    margin-right: .42rem;
    background: #36d6a9;
    box-shadow: 0 0 0 5px rgba(54, 214, 169, .10);
}
.status-dot.offline {
    background: #f97373;
    box-shadow: 0 0 0 5px rgba(249, 115, 115, .10);
}
[data-testid="stDecoration"], footer { display: none; }
</style>
"""


@st.cache_resource(show_spinner=False)
def load_rag() -> FinGuardRAG:
    """Load the index and models once for the Streamlit process."""
    return FinGuardRAG()


@st.cache_data(ttl=15, show_spinner=False)
def check_ollama_health() -> tuple[bool, str]:
    """Check whether Ollama and the configured model are available."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=1.5)
        response.raise_for_status()
        names = {model.get("name", "") for model in response.json().get("models", [])}
        available = any(
            name == OLLAMA_MODEL_NAME or name.startswith(f"{OLLAMA_MODEL_NAME}:")
            for name in names
        )
        if available:
            return True, "Ollama and model ready"
        return False, f"Ollama online, but {OLLAMA_MODEL_NAME} is missing"
    except requests.RequestException:
        return False, "Ollama is offline"


def context_snapshot(documents: list[Document]) -> list[dict[str, Any]]:
    """Convert reranked documents into display-safe session data."""
    snapshots: list[dict[str, Any]] = []
    for document in documents:
        metadata = document.metadata
        snapshots.append(
            {
                "title": metadata.get("title"),
                "filename": metadata.get("filename", metadata.get("source", "Unknown")),
                "page": metadata.get("page_label", metadata.get("page", "?")),
                "chunk_id": metadata.get("chunk_id", "Unknown"),
                "rerank_score": metadata.get("rerank_score"),
                "retrieval_distance": metadata.get("retrieval_distance"),
                "text": document.page_content,
            }
        )
    return snapshots


def collect_session_metrics(history: list[dict[str, Any]]) -> dict[str, int | float]:
    """Aggregate session-level metrics from chat history."""
    sources = [source for turn in history for source in turn.get("sources", [])]
    contexts = [context for turn in history for context in turn.get("contexts", [])]
    scores = [
        float(context["rerank_score"])
        for context in contexts
        if context.get("rerank_score") is not None
    ]
    unique_sources = {
        source.get("filename") or source.get("file") or source.get("title") or "Unknown"
        for source in sources
    }
    return {
        "questions": len(history),
        "citations": len(sources),
        "evidence_chunks": len(contexts),
        "unique_sources": len(unique_sources),
        "average_score": mean(scores) if scores else 0.0,
    }


def transcript_as_markdown(history: list[dict[str, Any]]) -> str:
    """Build a downloadable Markdown transcript."""
    lines = [
        "# FinGuard AI Research Session",
        "",
        f"_Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
    ]
    for index, turn in enumerate(history, start=1):
        lines.extend(
            [
                f"## Question {index}",
                "",
                turn.get("question", ""),
                "",
                "### Answer",
                "",
                turn.get("answer", ""),
                "",
            ]
        )
        if turn.get("sources"):
            lines.extend(["### Sources", ""])
            for source in turn["sources"]:
                title = source.get("title") or source.get("filename") or source.get("file") or "Unknown"
                lines.append(
                    f"- {title}, page {source.get('page', '?')}, "
                    f"chunk `{source.get('chunk_id', 'Unknown')}`"
                )
            lines.append("")
    return "\n".join(lines)


def apply_theme() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown(
        f"""
        <section class="hero">
            <div class="eyebrow">Evidence-grounded local research</div>
            <h1 class="hero-title">{APP_TITLE}</h1>
            <p class="hero-copy">
                Explore explainable machine learning, financial AI governance,
                regulatory capital and model-risk controls through a private,
                local retrieval-augmented generation pipeline.
            </p>
            <div class="badge-row">
                <span class="badge">🧠 {OLLAMA_MODEL_NAME}</span>
                <span class="badge">🔎 FAISS + BGE</span>
                <span class="badge">🎯 CrossEncoder reranking</span>
                <span class="badge">📚 Page-level citations</span>
                <span class="badge">🔒 Local inference</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_card() -> None:
    st.markdown(
        """
        <div class="panel-card">
            <div class="panel-label">Active pipeline</div>
            <div class="panel-title">Evidence before generation</div>
            <div class="panel-copy">
                Every answer is assembled from retrieved and reranked passages,
                then generated locally with its supporting evidence attached.
            </div>
            <div class="pipeline">
                <span class="pipeline-step">Question</span><span class="pipeline-arrow">→</span>
                <span class="pipeline-step">BGE embedding</span><span class="pipeline-arrow">→</span>
                <span class="pipeline-step">FAISS top-k</span><span class="pipeline-arrow">→</span>
                <span class="pipeline-step">Reranker</span><span class="pipeline-arrow">→</span>
                <span class="pipeline-step">Qwen3</span><span class="pipeline-arrow">→</span>
                <span class="pipeline-step">Cited answer</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(history: list[dict[str, Any]]) -> None:
    metrics = collect_session_metrics(history)
    columns = st.columns(4)
    columns[0].metric("Questions", metrics["questions"])
    columns[1].metric("Citations", metrics["citations"])
    columns[2].metric("Evidence chunks", metrics["evidence_chunks"])
    columns[3].metric("Sources explored", metrics["unique_sources"])


def queue_question(question: str) -> None:
    st.session_state.queued_question = question


def render_suggested_questions() -> None:
    st.markdown("#### Start with a research question")
    columns = st.columns(2)
    for index, prompt in enumerate(SUGGESTED_QUESTIONS):
        columns[index % 2].button(
            prompt,
            key=f"suggested-question-{index}",
            use_container_width=True,
            on_click=queue_question,
            args=(prompt,),
        )


def display_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.caption("No valid inline citations were returned.")
        return
    tabs = st.tabs(
        [f"{index}. p.{source.get('page', '?')}" for index, source in enumerate(sources, start=1)]
    )
    for tab, source in zip(tabs, sources):
        with tab:
            title = source.get("title") or source.get("filename") or source.get("file") or "Unknown"
            st.markdown(f"**{title}**")
            st.caption(
                f"Page {source.get('page', '?')} · Chunk {source.get('chunk_id', 'Unknown')}"
            )
            score = source.get("rerank_score")
            if score is not None:
                st.metric("Reranker score", f"{float(score):.3f}")


def display_assistant_turn(turn: dict[str, Any]) -> None:
    st.markdown(turn["answer"])
    citation_tab, source_tab, evidence_tab = st.tabs(
        ["📌 Citation trail", "📚 Source cards", "🧩 Retrieved context"]
    )
    with citation_tab:
        sources = turn.get("sources", [])
        if not sources:
            st.caption("No citation trail was returned.")
        else:
            chips = []
            for source in sources:
                title = source.get("title") or source.get("filename") or source.get("file") or "Unknown"
                chips.append(
                    f'<span class="source-chip">{title} · p.{source.get("page", "?")}</span>'
                )
            st.markdown("".join(chips), unsafe_allow_html=True)
    with source_tab:
        display_sources(turn.get("sources", []))
    with evidence_tab:
        contexts = turn.get("contexts", [])
        if not contexts:
            st.caption("No retrieved context was saved.")
        for index, context in enumerate(contexts, start=1):
            label = (
                f"{index}. {context.get('filename', context.get('title', 'Source'))} "
                f"· p.{context.get('page', '?')}"
            )
            with st.expander(label):
                score = context.get("rerank_score")
                score_text = f"{float(score):.3f}" if score is not None else "n/a"
                st.caption(
                    f"Chunk {context.get('chunk_id', 'Unknown')} · Reranker score {score_text}"
                )
                st.write(context.get("text", ""))


def pipeline_error_message(exc: Exception) -> str:
    cause = exc.__cause__
    if isinstance(cause, requests.ConnectionError):
        return f"Ollama is not running at {OLLAMA_BASE_URL}. Start Ollama and try again."
    if isinstance(cause, requests.HTTPError) and cause.response is not None:
        if cause.response.status_code == 404:
            return (
                f"The Ollama model `{OLLAMA_MODEL_NAME}` is unavailable. "
                f"Run `ollama pull {OLLAMA_MODEL_NAME}`."
            )
    if isinstance(exc, FileNotFoundError):
        return (
            "The FAISS index is missing. Build it with "
            "`python -m src.build_index` before starting the app."
        )
    if isinstance(exc, RuntimeError) and "Ollama request failed" in str(exc):
        return f"Ollama could not serve `{OLLAMA_MODEL_NAME}`. Confirm the service and model."
    return "The RAG pipeline encountered an unexpected error. Check the terminal logs."


def clear_chat() -> None:
    st.session_state.chat_history = []
    st.session_state.pop("queued_question", None)


def configure_sidebar() -> None:
    online, health_message = check_ollama_health()
    with st.sidebar:
        st.markdown("## 🛡️ FinGuard")
        st.caption("Local evidence intelligence")
        dot_class = "status-dot" if online else "status-dot offline"
        st.markdown(
            f'<div class="panel-card"><div class="panel-label">System health</div>'
            f'<div class="panel-title"><span class="{dot_class}"></span>'
            f'{health_message}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("### Model stack")
        st.caption("Local generator")
        st.code(OLLAMA_MODEL_NAME, language=None)
        st.caption("Embedding model")
        st.code(EMBEDDING_MODEL_NAME, language=None)
        st.caption("Reranker")
        st.code(RERANKER_MODEL_NAME, language=None)
        first, second = st.columns(2)
        first.metric("Candidates", RETRIEVAL_K)
        second.metric("Final context", RERANK_TOP_N)
        st.divider()
        history = st.session_state.get("chat_history", [])
        st.download_button(
            "⬇️ Export research session",
            data=transcript_as_markdown(history),
            file_name="finguard_research_session.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=not history,
        )
        st.button("🗑️ Clear chat", use_container_width=True, on_click=clear_chat)
        st.caption("FinGuard explains indexed research. It does not provide investment advice.")


def render_empty_state() -> None:
    st.markdown("#### Research coverage")
    columns = st.columns(2)
    with columns[0]:
        st.markdown(
            """
            <div class="panel-card">
                <div class="panel-label">Research paper 01</div>
                <div class="panel-title">Explainable financial risk models</div>
                <div class="panel-copy">
                    SHAP, regulatory capital, ICAAP, CCAR and auditable
                    machine-learning risk decisions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with columns[1]:
        st.markdown(
            """
            <div class="panel-card">
                <div class="panel-label">Research paper 02</div>
                <div class="panel-title">Generative AI governance</div>
                <div class="panel-copy">
                    Controls, accountability, validation and risk governance
                    for GenAI inside financial institutions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    render_suggested_questions()


def render_relevance_chart(contexts: list[dict[str, Any]]) -> None:
    rows = []
    for index, context in enumerate(contexts, start=1):
        score = context.get("rerank_score")
        if score is None:
            continue
        rows.append(
            {
                "Evidence": (
                    f"{index}. {context.get('filename', context.get('title', 'Source'))} "
                    f"p.{context.get('page', '?')}"
                ),
                "Reranker score": float(score),
            }
        )
    if not rows:
        st.info("Ask a question to generate an evidence-relevance chart.")
        return
    frame = pd.DataFrame(rows).set_index("Evidence")
    st.bar_chart(frame, horizontal=True, height=270)


def render_source_coverage(history: list[dict[str, Any]]) -> None:
    labels = []
    for turn in history:
        for source in turn.get("sources", []):
            labels.append(
                source.get("filename") or source.get("file") or source.get("title") or "Unknown"
            )
    if not labels:
        st.info("Source coverage appears after the first cited answer.")
        return
    counts = Counter(labels)
    frame = pd.DataFrame(
        {"Document": list(counts.keys()), "Citations": list(counts.values())}
    ).set_index("Document")
    st.bar_chart(frame, height=250)


def render_insight_panel(history: list[dict[str, Any]]) -> None:
    st.markdown("### Research insights")
    turn = history[-1] if history else None
    with st.container(border=True):
        st.markdown("##### Latest evidence relevance")
        render_relevance_chart(turn.get("contexts", []) if turn else [])
    with st.container(border=True):
        st.markdown("##### Session source coverage")
        render_source_coverage(history)
    with st.container(border=True):
        st.markdown("##### How to read the evidence")
        st.caption(
            "FAISS retrieves broad candidates. The CrossEncoder reranker then scores "
            "query–passage pairs, and only the highest-ranked chunks are supplied to Qwen3."
        )


def process_question(rag: FinGuardRAG, question: str) -> None:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🛡️"):
        try:
            with st.status("Building an evidence-grounded answer...", expanded=True) as status:
                st.write("🔎 Searching the FAISS knowledge base")
                st.write("🎯 Reranking the strongest passages")
                st.write(f"🧠 Generating locally with {OLLAMA_MODEL_NAME}")
                result = rag.answer(question)
                status.update(
                    label="Answer grounded in retrieved evidence",
                    state="complete",
                    expanded=False,
                )
            answer = str(result.get("answer", "")).strip()
            if not answer:
                st.error("The pipeline returned an empty answer. Please try again.")
                return
            turn = {
                "question": question,
                "answer": answer,
                "sources": result.get("sources", []),
                "contexts": context_snapshot(result.get("documents", [])),
            }
            display_assistant_turn(turn)
            st.session_state.chat_history.append(turn)
        except Exception as exc:
            st.error(pipeline_error_message(exc))
            if not isinstance(exc, (FileNotFoundError, RuntimeError)):
                st.exception(exc)


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    render_hero()
    configure_sidebar()
    history = st.session_state.chat_history
    render_metrics(history)
    render_pipeline_card()

    try:
        rag = load_rag()
    except Exception as exc:
        st.error(pipeline_error_message(exc))
        st.exception(exc)
        return

    chat_column, insight_column = st.columns([1.9, 1], gap="large")
    with chat_column:
        st.markdown("### Research chat")
        if not history:
            render_empty_state()
        for turn in history:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(turn["question"])
            with st.chat_message("assistant", avatar="🛡️"):
                display_assistant_turn(turn)

        queued_question = st.session_state.pop("queued_question", None)
        typed_question = st.chat_input(
            "Ask about SHAP, model risk, regulatory capital or GenAI governance..."
        )
        question = queued_question or typed_question
        if question:
            process_question(rag, question)

    with insight_column:
        render_insight_panel(history)


if __name__ == "__main__":
    main()