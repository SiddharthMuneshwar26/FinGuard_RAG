"""
app.py — Streamlit front end for FinGuard-RAG.

Run:  streamlit run src/app.py

Shows the grounded answer plus an expandable "Sources" panel so a reviewer can
verify every citation against the paper section and page it came from.
"""

import sys
from pathlib import Path

# Make `from src...` work when launched as `streamlit run src/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.chatbot import answer
from src import config

st.set_page_config(page_title="FinGuard-RAG", page_icon="🛡️", layout="centered")

st.title("🛡️ FinGuard-RAG")
st.caption("Grounded Q&A over two financial-AI governance & interpretability papers "
           "(GAICF / SR 26-2 and SHARC / ICAAP-CCAR).")

with st.sidebar:
    st.subheader("Config")
    st.write(f"**Provider:** {config.LLM_PROVIDER}")
    st.write(f"**Embedder:** {config.EMBED_MODEL.split('/')[-1]}")
    st.write(f"**Rerank:** {config.RERANK}  |  **top_k→k:** "
             f"{config.TOP_K}→{config.FINAL_K}")
    st.markdown("---")
    st.markdown("Example questions:")
    st.markdown(
        "- What governance gap does SR 26-2 create?\n"
        "- How does SHARC make GPR models auditable?\n"
        "- Which SVaR component dominates under stress?"
    )

if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask about the papers...")

# Replay history
for turn in st.session_state.history:
    st.chat_message("user").write(turn["q"])
    with st.chat_message("assistant"):
        st.markdown(turn["a"])
        if turn["sources"]:
            with st.expander("Sources"):
                for m in turn["sources"]:
                    st.markdown(
                        f"- **{m['paper_id']}** — {m['section']} · p.{m['page']} "
                        f"· `{m['chunk_id']}`"
                    )

if query:
    st.chat_message("user").write(query)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving + grounding..."):
            out = answer(query)
        st.markdown(out["answer"])
        if out["sources"]:
            with st.expander("Sources"):
                for m in out["sources"]:
                    st.markdown(
                        f"- **{m['paper_id']}** — {m['section']} · p.{m['page']} "
                        f"· `{m['chunk_id']}`"
                    )
    st.session_state.history.append(
        {"q": query, "a": out["answer"], "sources": out["sources"]}
    )
