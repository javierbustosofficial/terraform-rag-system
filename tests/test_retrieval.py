"""
Unit tests for the retrieval pipeline.

OpenAI calls (_rewrite_query, embed_chunks) are mocked so these tests run
offline and don't consume API credits. ChromaDB is hit for real.
"""

from unittest.mock import MagicMock, patch
import pytest

# A 1536-dim zero vector that matches text-embedding-3-small output size.
_ZERO_VEC = [0.0] * 1536


@pytest.fixture(autouse=True)
def _mock_openai_env(monkeypatch):
    """Prevent dotenv from clobbering the test environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# _keyword_score — pure function, no mocking needed
# ---------------------------------------------------------------------------

class TestKeywordScore:
    def _score(self, text, terms):
        from terraform_rag.retrieval.retriever import _keyword_score
        return _keyword_score(text, terms)

    def test_all_terms_present(self):
        assert self._score("aws s3 bucket resource", ["aws", "s3", "bucket"]) == pytest.approx(1.0)

    def test_no_terms_present(self):
        assert self._score("azure vm scale set", ["aws", "s3"]) == pytest.approx(0.0)

    def test_partial_match(self):
        score = self._score("aws provider config", ["aws", "s3", "bucket"])
        assert score == pytest.approx(1 / 3)

    def test_empty_terms_returns_zero(self):
        assert self._score("any text here", []) == pytest.approx(0.0)

    def test_case_insensitive(self):
        assert self._score("AWS S3 BUCKET", ["aws", "s3", "bucket"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# retrieve() — OpenAI mocked, real ChromaDB
# ---------------------------------------------------------------------------

SAMPLE_QUERY = "how do I configure an S3 bucket in Terraform?"


def _make_mock_embedding_response():
    item = MagicMock()
    item.embedding = _ZERO_VEC
    item.index = 0
    resp = MagicMock()
    resp.data = [item]
    return resp


@patch("terraform_rag.retrieval.retriever._rewrite_query", return_value=SAMPLE_QUERY)
@patch("openai.OpenAI")
def test_retrieve_returns_list(mock_openai_cls, mock_rewrite):
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _make_mock_embedding_response()
    mock_openai_cls.return_value = mock_client

    from terraform_rag.retrieval.retriever import retrieve
    results = retrieve(SAMPLE_QUERY, n_results=5)

    assert isinstance(results, list)
    assert len(results) <= 5


@patch("terraform_rag.retrieval.retriever._rewrite_query", return_value=SAMPLE_QUERY)
@patch("openai.OpenAI")
def test_retrieve_result_structure(mock_openai_cls, mock_rewrite):
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _make_mock_embedding_response()
    mock_openai_cls.return_value = mock_client

    from terraform_rag.retrieval.retriever import retrieve
    results = retrieve(SAMPLE_QUERY, n_results=3)

    for r in results:
        assert "text" in r
        assert "metadata" in r
        assert "score" in r
        assert isinstance(r["score"], float)
        assert isinstance(r["text"], str)
        assert len(r["text"]) > 0


@patch("terraform_rag.retrieval.retriever._rewrite_query", return_value=SAMPLE_QUERY)
@patch("openai.OpenAI")
def test_retrieve_respects_n_results(mock_openai_cls, mock_rewrite):
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _make_mock_embedding_response()
    mock_openai_cls.return_value = mock_client

    from terraform_rag.retrieval.retriever import retrieve
    for n in (1, 3, 5):
        results = retrieve(SAMPLE_QUERY, n_results=n)
        assert len(results) <= n


@patch("terraform_rag.retrieval.retriever._rewrite_query", return_value=SAMPLE_QUERY)
@patch("openai.OpenAI")
def test_retrieve_scores_are_sorted_descending(mock_openai_cls, mock_rewrite):
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _make_mock_embedding_response()
    mock_openai_cls.return_value = mock_client

    from terraform_rag.retrieval.retriever import retrieve
    results = retrieve(SAMPLE_QUERY, n_results=5)

    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


@patch("terraform_rag.retrieval.retriever._rewrite_query", return_value="aws provider block")
@patch("openai.OpenAI")
def test_retrieve_section_filter(mock_openai_cls, mock_rewrite):
    """Passing a section filter should not crash — results may be empty if no match."""
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _make_mock_embedding_response()
    mock_openai_cls.return_value = mock_client

    from terraform_rag.retrieval.retriever import retrieve
    results = retrieve("aws provider block", n_results=3, section="providers")

    assert isinstance(results, list)
