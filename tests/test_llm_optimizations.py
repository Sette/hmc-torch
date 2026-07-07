"""Tests for optimized LLM reranking utilities."""

from hmc.llm import ollama_agent
from hmc.pipeline.global_classifier.llm_train import _truncate_document_text


def test_rerank_document_labels_result_uses_cache(tmp_path, monkeypatch):
    calls = []

    def fake_call_ollama_api(**kwargs):
        calls.append(kwargs)
        return '{"selected_labels": ["cs.AI"]}'

    monkeypatch.setattr(ollama_agent, "_call_ollama_api", fake_call_ollama_api)

    kwargs = {
        "document_text": "A paper about neural networks.",
        "candidate_labels": [{"label": "cs.AI", "score": 0.73, "path": "cs > cs.AI"}],
        "model": "fake-model",
        "cache_dir": tmp_path,
    }

    first = ollama_agent.rerank_document_labels_result(**kwargs)
    second = ollama_agent.rerank_document_labels_result(**kwargs)

    assert first.payload == {"selected_labels": ["cs.AI"]}
    assert second.payload == {"selected_labels": ["cs.AI"]}
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(calls) == 1


def test_truncate_document_text_keeps_head_and_tail():
    text = "0123456789" * 10

    truncated, was_truncated = _truncate_document_text(text, max_chars=20)

    assert was_truncated is True
    assert truncated.startswith("01234567890123")
    assert truncated.endswith("456789")
    assert "[truncated]" in truncated


def test_truncate_document_text_can_be_disabled():
    text = "0123456789" * 10

    truncated, was_truncated = _truncate_document_text(text, max_chars=0)

    assert was_truncated is False
    assert truncated == text
