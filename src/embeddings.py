from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from .config import BGE_QUERY_PREFIX, EMBEDDING_DEVICE, EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
    device: str = EMBEDDING_DEVICE,
) -> HuggingFaceEmbeddings:
    """Create the Hugging Face embedding model."""
    print(f"Loading embedding model: {model_name}")
    print(f"Embedding device: {device}")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": device,
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 16,
        },
        show_progress=False,
    )


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the cached embedding model using central configuration."""
    return get_embedding_model()


def prepare_query(query: str) -> str:
    """Apply the BGE instruction prefix on queries only."""
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("Query cannot be empty")
    return BGE_QUERY_PREFIX + cleaned
