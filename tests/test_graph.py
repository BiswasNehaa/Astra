"""Unit tests for graph.py's pure routing logic and node contracts.

generate_node/verify_node do `from vectorstore import search` and
`from llm import ask_ai` as local imports, so we stub those modules in
sys.modules via monkeypatch instead of hitting Chroma/Groq for real.
"""
import sys
import types

import pytest

from graph import decide_next_step, generate_node, verify_node


class TestDecideNextStep:
    def test_ends_when_supported(self):
        state = {"is_supported": True, "loop_count": 1}
        assert decide_next_step(state) == "end"

    def test_retries_when_unsupported_and_under_cap(self):
        state = {"is_supported": False, "loop_count": 1}
        assert decide_next_step(state) == "retry"

    def test_ends_when_unsupported_but_retry_cap_reached(self):
        state = {"is_supported": False, "loop_count": 2}
        assert decide_next_step(state) == "end"


def _stub_vectorstore(monkeypatch, documents, metadatas):
    fake = types.ModuleType("vectorstore")

    def fake_search(query, top_k=3):
        return {"documents": [documents], "metadatas": [metadatas]}

    fake.search = fake_search
    monkeypatch.setitem(sys.modules, "vectorstore", fake)


def _stub_llm(monkeypatch, response):
    fake = types.ModuleType("llm")
    fake.ask_ai = lambda *args, **kwargs: response
    monkeypatch.setitem(sys.modules, "llm", fake)


class TestGenerateNode:
    def test_sets_answer_context_and_increments_loop_count(self, monkeypatch):
        _stub_vectorstore(
            monkeypatch,
            documents=["chunk one", "chunk two"],
            metadatas=[
                {"title": "Paper A", "url": "http://a"},
                {"title": "Paper A", "url": "http://a"},
            ],
        )
        _stub_llm(monkeypatch, "a generated answer")

        state = {"question": "what is X?", "loop_count": 0}
        result = generate_node(state)

        assert result["answer"] == "a generated answer"
        assert result["context_chunks"] == ["chunk one", "chunk two"]
        assert result["loop_count"] == 1

    def test_deduplicates_sources_by_title(self, monkeypatch):
        _stub_vectorstore(
            monkeypatch,
            documents=["chunk one", "chunk two", "chunk three"],
            metadatas=[
                {"title": "Paper A", "url": "http://a"},
                {"title": "Paper A", "url": "http://a"},
                {"title": "Paper B", "url": "http://b"},
            ],
        )
        _stub_llm(monkeypatch, "a generated answer")

        result = generate_node({"question": "q", "loop_count": 0})

        assert result["sources"] == [
            {"title": "Paper A", "url": "http://a"},
            {"title": "Paper B", "url": "http://b"},
        ]


class TestVerifyNode:
    def test_marks_supported_on_a_clean_yes(self, monkeypatch):
        _stub_llm(monkeypatch, "yes")

        state = {"context_chunks": ["chunk"], "answer": "an answer", "loop_count": 1}
        result = verify_node(state)

        assert result["is_supported"] is True

    def test_marks_unsupported_on_a_clean_no(self, monkeypatch):
        _stub_llm(monkeypatch, "no")

        state = {"context_chunks": ["chunk"], "answer": "an answer", "loop_count": 1}
        result = verify_node(state)

        assert result["is_supported"] is False
