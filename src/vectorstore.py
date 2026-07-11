"""FAISS persistence helpers."""
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import VECTORSTORE_DIR
from .embeddings import get_embedding_model


def create_vectorstore(
    chunks: list[Document],
    embeddings: Embeddings | None = None,
) -> FAISS:
    """Create a new FAISS vector store from chunks."""
    if not chunks:
        raise ValueError("No chunks were supplied to FAISS.")

    embeddings = embeddings or get_embedding_model()

    print(f"Creating FAISS index from {len(chunks)} chunks...")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    print("FAISS index created.")
    return vectorstore


def index_exists(index_dir: Path = VECTORSTORE_DIR) -> bool:
    """Return whether both persisted FAISS index files exist."""
    return (index_dir / "index.faiss").is_file() and (index_dir / "index.pkl").is_file()


def build_vectorstore(
    chunks: list[Document],
    path: Path = VECTORSTORE_DIR,
) -> FAISS:
    """Create and persist a FAISS store in one indexing operation."""
    vectorstore = create_vectorstore(chunks)
    save_vectorstore(vectorstore, path)
    return vectorstore


def save_vectorstore(
    vectorstore: FAISS,
    index_dir: Path = VECTORSTORE_DIR,
) -> None:
    """Save a FAISS index locally."""
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))

    print(f"Vector store saved to: {index_dir}")


def load_vectorstore(
    index_dir: Path = VECTORSTORE_DIR,
    embeddings: Embeddings | None = None,
) -> FAISS:
    """Load an existing local FAISS index."""
    if not index_exists(index_dir):
        raise FileNotFoundError(
            f"Vector store not found at {index_dir}. "
            "Run: python -m src.build_index"
        )

    embeddings = embeddings or get_embedding_model()

    # Only use this with an index created by you.
    return FAISS.load_local(
        folder_path=str(index_dir),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
