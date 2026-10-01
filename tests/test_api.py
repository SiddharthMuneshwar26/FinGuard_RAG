from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app


class FakeRAG:
    def answer(self, question: str, top_k: int = 4, generate: bool = True):
        return {
            "answer": "Test answer" if generate else None,
            "sources": [
                {
                    "title": "Test Document",
                    "filename": "test.pdf",
                    "file": "test.pdf",
                    "page": 1,
                    "chunk_id": "TEST-p1-c1",
                    "rerank_score": 1.5,
                }
            ],
            "documents": [
                type(
                    "FakeDocument",
                    (),
                    {
                        "page_content": "Test evidence chunk.",
                        "metadata": {
                            "chunk_id": "TEST-p1-c1",
                            "filename": "test.pdf",
                            "page": 1,
                            "retrieval_distance": 0.25,
                            "rerank_score": 1.5,
                            "rerank_position": 1,
                        },
                    },
                )()
            ],
            "latency_ms": {
                "retrieval": 1.0,
                "rerank": 2.0,
                "generation": 3.0 if generate else 0.0,
                "total": 6.0 if generate else 3.0,
            },
        }


def make_client():
    fake_rag = FakeRAG()

    with patch("api.main.FinGuardRAG", return_value=fake_rag):
        client = TestClient(app)
        client.__enter__()

    return client


def test_health():
    with patch("api.main.FinGuardRAG", return_value=FakeRAG()):
        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_retrieval_only():
    with patch("api.main.FinGuardRAG", return_value=FakeRAG()):
        with TestClient(app) as client:
            response = client.post(
                "/query",
                json={
                    "question": "What is SHAP?",
                    "top_k": 4,
                    "generate": False,
                },
            )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] is None
    assert len(data["citations"]) == 1
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["chunk_id"] == "TEST-p1-c1"
    assert data["latency_ms"]["generation"] == 0.0


def test_query_generation():
    with patch("api.main.FinGuardRAG", return_value=FakeRAG()):
        with TestClient(app) as client:
            response = client.post(
                "/query",
                json={
                    "question": "What is SHAP?",
                    "top_k": 4,
                    "generate": True,
                },
            )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "Test answer"
    assert len(data["citations"]) == 1
    assert len(data["chunks"]) == 1
    assert data["latency_ms"]["generation"] == 3.0


def test_query_rejects_invalid_top_k():
    with patch("api.main.FinGuardRAG", return_value=FakeRAG()):
        with TestClient(app) as client:
            response = client.post(
                "/query",
                json={
                    "question": "What is SHAP?",
                    "top_k": 0,
                    "generate": False,
                },
            )

    assert response.status_code == 422


def test_query_rejects_empty_question():
    with patch("api.main.FinGuardRAG", return_value=FakeRAG()):
        with TestClient(app) as client:
            response = client.post(
                "/query",
                json={
                    "question": "",
                    "top_k": 4,
                    "generate": False,
                },
            )

    assert response.status_code == 422
def test_query_returns_503_for_runtime_error():
    class FailingRAG:
        def answer(self, question: str, top_k: int = 4, generate: bool = True):
            raise RuntimeError("Ollama unavailable")

    with patch("api.main.FinGuardRAG", return_value=FailingRAG()):
        with TestClient(app) as client:
            response = client.post(
                "/query",
                json={
                    "question": "What is SHAP?",
                    "top_k": 4,
                    "generate": True,
                },
            )

    assert response.status_code == 503
    assert response.json()["detail"] == "Ollama unavailable"


def test_query_returns_500_for_unexpected_error():
    class BrokenRAG:
        def answer(self, question: str, top_k: int = 4, generate: bool = True):
            raise Exception("unexpected failure")

    with patch("api.main.FinGuardRAG", return_value=BrokenRAG()):
        with TestClient(app) as client:
            response = client.post(
                "/query",
                json={
                    "question": "What is SHAP?",
                    "top_k": 4,
                    "generate": True,
                },
            )

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"