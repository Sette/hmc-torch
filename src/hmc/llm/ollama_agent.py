"""Ollama-backed LLM reranking utilities for hierarchical classification."""

from __future__ import annotations

import json
from typing import Any, Iterable
from urllib import error, request

_OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaAPIError(RuntimeError):
    """Ollama API failure."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"Ollama API error {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


def _normalize_ollama_chat_url(base_url: str | None) -> str:
    """Accept either an Ollama base URL or the complete /api/chat endpoint."""
    raw = (base_url or _OLLAMA_BASE_URL).strip().rstrip("/")
    if raw.endswith("/api/chat"):
        return raw
    if raw.endswith("/api"):
        return f"{raw}/chat"
    return f"{raw}/api/chat"


def _call_ollama_api(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    base_url: str | None = None,
    num_ctx: int = 4096,
    timeout: int = 120,
) -> str:
    options: dict[str, Any] = {
        "temperature": temperature,
        "num_predict": max_tokens,
    }
    if num_ctx > 0:
        options["num_ctx"] = num_ctx

    body = {
        "model": model,
        "messages": messages,
        "format": "json",
        "stream": False,
        "think": False,
        "options": options,
    }
    req = request.Request(
        _normalize_ollama_chat_url(base_url),
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "hmc-torch/0.0.8",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise OllamaAPIError(exc.code, detail) from exc
    except error.URLError as exc:
        chat_url = _normalize_ollama_chat_url(base_url)
        raise RuntimeError(f"Could not reach Ollama at {chat_url}: {exc.reason}") from exc

    message = payload.get("message", {})
    content = message.get("content")
    if not content:
        if message.get("thinking"):
            raise RuntimeError(
                "Ollama returned reasoning tokens but no JSON content. "
                "The request sets think=false and /no_think; if this persists, "
                "increase --llm_max_tokens or update the Ollama model/runtime. "
                f"Payload: {payload}"
            )
        raise RuntimeError(f"Ollama API returned no message content: {payload}")
    return content


def _json_from_text(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def build_candidate_context(
    *,
    document_text: str,
    candidate_labels: list[dict[str, Any]],
    hierarchy_path: list[str] | None = None,
) -> str:
    lines = [
        "You are reranking hierarchical labels for scientific document classification.",
        "Select only labels that are justified by the document and keep hierarchy consistency.",
        "Return only a valid JSON object with the key selected_labels.",
        "Do not use markdown, comments, or prose outside the JSON object.",
        "",
        f"DOCUMENT: {document_text}",
    ]
    if hierarchy_path:
        lines.append(f"HIERARCHY_PATH: {' > '.join(hierarchy_path)}")
    lines.append("CANDIDATES:")
    for item in candidate_labels:
        lines.append(
            f"- {item['label']} | score={item['score']:.4f} | path={item.get('path', '')}"
        )
    lines.append("Output format: {\"selected_labels\": [\"label1\", \"label2\"]}")
    return "\n".join(lines)


def rerank_document_labels(
    *,
    document_text: str,
    candidate_labels: list[dict[str, Any]],
    model: str,
    base_url: str | None = None,
    num_ctx: int = 4096,
    temperature: float = 0.0,
    max_tokens: int = 256,
    hierarchy_path: list[str] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """Ask Ollama to rerank candidate labels and return parsed JSON."""
    prompt = "/no_think\n" + build_candidate_context(
        document_text=document_text,
        candidate_labels=candidate_labels,
        hierarchy_path=hierarchy_path,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise label reranking agent. "
                "You must return only valid JSON. "
                "Do not include markdown, prose, or thinking tags."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    content = _call_ollama_api(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
        num_ctx=num_ctx,
        timeout=timeout,
    )
    return _json_from_text(content)


def parse_selected_labels(response: dict[str, Any]) -> list[str]:
    selected = response.get("selected_labels", [])
    if isinstance(selected, str):
        return [selected]
    if isinstance(selected, Iterable):
        return [str(item) for item in selected]
    return []
