"""Streamlit chat interface for the FinGuard-RAG backend."""

from __future__ import annotations

from typing import Any

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


@st.cache_resource(show_spinner=False)
def load_rag() -> FinGuardRAG:
    """Load the index and models once for the Streamlit process."""
    return FinGuardRAG()


def context_snapshot(documents: list[Document]) -> list[dict[str, Any]]:
    """Convert reranked documents into display-safe session data."""
    snapshots: list[dict[str, Any]] = []
    for document in documents:
        metadata = document.metadata
        snapshots.append(
            {
                "title": metadata.get("title"),
                "filename": metadata.get("filename", metadata.get("source", "Unknown")),
                "page": metadata.get("page", "?"),
                "chunk_id": metadata.get("chunk_id", "Unknown"),
                "rerank_score": metadata.get("rerank_score"),
                "text": document.page_content,
            }
        )
    return snapshots


def display_sources(sources: list[dict[str, Any]]) -> None:
    """Render page-level citations beneath an answer."""
    if not sources:
        st.caption("No valid inline citations were returned.")
        return
    st.markdown("**Sources**")
    for source in sources:
        title = source.get("title") or source.get("filename") or source.get("file", "Unknown")
        details = f"p.{source.get('page', '?')} · `{source.get('chunk_id', 'Unknown')}`"
        score = source.get("rerank_score")
        if score is not None:
            details += f" · score {float(score):.3f}"
        st.markdown(f"- **{title}** — {details}")


def display_context(contexts: list[dict[str, Any]]) -> None:
    """Render the reranked chunks supplied to the LLM."""
    with st.expander("Retrieved Context"):
        if not contexts:
            st.caption("No context was retrieved.")
            return
        for index, context in enumerate(contexts, start=1):
            title = context.get("title") or context.get("filename", "Unknown")
            score = context.get("rerank_score")
            heading = (
                f"{index}. {title} · p.{context.get('page', '?')} · "
                f"{context.get('chunk_id', 'Unknown')}"
            )
            if score is not None:
                heading += f" · score {float(score):.3f}"
            st.markdown(f"**{heading}**")
            st.write(context.get("text", ""))
            if index < len(contexts):
                st.divider()


def display_assistant_turn(turn: dict[str, Any]) -> None:
    """Render one saved assistant response."""
    st.markdown(turn["answer"])
    display_sources(turn.get("sources", []))
    display_context(turn.get("contexts", []))


def pipeline_error_message(exc: Exception) -> str:
    """Translate known backend failures into actionable UI messages."""
    cause = exc.__cause__
    if isinstance(cause, requests.ConnectionError):
        return f"Ollama is not running at {OLLAMA_BASE_URL}. Start Ollama and try again."
    if isinstance(cause, requests.HTTPError) and cause.response is not None:
        if cause.response.status_code == 404:
            return f"The Ollama model `{OLLAMA_MODEL_NAME}` is unavailable. Run `ollama pull {OLLAMA_MODEL_NAME}`."
    if isinstance(exc, FileNotFoundError):
        return "The FAISS index is missing. Build it with `python -m src.build_index` before starting the app."
    if isinstance(exc, RuntimeError) and "Ollama request failed" in str(exc):
        return f"Ollama could not serve `{OLLAMA_MODEL_NAME}`. Confirm the service and model are available."
    return "The RAG pipeline encountered an unexpected error. Check the terminal logs for details."


def configure_sidebar() -> None:
    """Show the active backend configuration and chat controls."""
    with st.sidebar:
        st.header("Configuration")
        st.markdown(f"**LLM:** `{OLLAMA_MODEL_NAME}`")
        st.markdown(f"**Embeddings:** `{EMBEDDING_MODEL_NAME}`")
        st.markdown(f"**Reranker:** `{RERANKER_MODEL_NAME}`")
        st.markdown(f"**Candidates:** {RETRIEVAL_K}")
        st.markdown(f"**Final contexts:** {RERANK_TOP_N}")
        st.divider()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


def main() -> None:
    """Run the FinGuard Streamlit application."""
    st.set_page_config(page_title="FinGuard AI", page_icon="🛡️", layout="centered")
    st.title("FinGuard AI")
    st.subheader("Local Financial AI Governance & Risk Research Assistant")
    configure_sidebar()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    try:
        rag = load_rag()
    except Exception as exc:
        st.error(pipeline_error_message(exc))
        st.exception(exc)
        return

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            display_assistant_turn(turn)

    question = st.chat_input("Ask about the indexed financial AI research...")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving evidence and generating a grounded answer..."):
                result = rag.answer(question)
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


if __name__ == "__main__":
    main()
