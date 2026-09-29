"""Unit tests for the pure text-processing helpers in ingestion.py.

ingestion.py does `from vectorstore import add_chunk` at module import time,
and vectorstore.py would in turn load a real sentence-transformers model.
We stub vectorstore out before importing ingestion so these tests stay fast
and don't need network access or a downloaded model.
"""
import sys
import types

if "vectorstore" not in sys.modules:
    fake_vectorstore = types.ModuleType("vectorstore")
    fake_vectorstore.add_chunk = lambda *args, **kwargs: None
    sys.modules["vectorstore"] = fake_vectorstore

if "arxiv" not in sys.modules:
    fake_arxiv = types.ModuleType("arxiv")
    fake_arxiv.Client = object
    fake_arxiv.Search = object
    fake_arxiv.SortCriterion = types.SimpleNamespace(SubmittedDate=1, Relevance=2)
    sys.modules["arxiv"] = fake_arxiv

from ingestion import chunk_text


def test_chunk_text_empty_string_returns_no_chunks():
    assert chunk_text("") == []


def test_chunk_text_shorter_than_chunk_size_returns_a_single_chunk():
    text = "one two three"
    assert chunk_text(text, chunk_size=100, overlap=20) == ["one two three"]


def test_chunk_text_splits_into_overlapping_windows():
    words = [f"w{i}" for i in range(25)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=10, overlap=3)

    assert chunks == [
        " ".join(words[0:10]),
        " ".join(words[7:17]),
        " ".join(words[14:24]),
        " ".join(words[21:25]),
    ]


def test_chunk_text_overlap_repeats_tail_of_previous_chunk():
    words = [f"w{i}" for i in range(20)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=10, overlap=3)

    assert chunks[0].split()[-3:] == chunks[1].split()[:3]
