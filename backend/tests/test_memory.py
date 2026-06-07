"""Unit and property-based tests for FAISSMemoryStore."""
import pytest
import tempfile
from hypothesis import given, settings, strategies as st
from src.memory import FAISSMemoryStore


@pytest.fixture
def memory(tmp_path):
    return FAISSMemoryStore(persist_dir=str(tmp_path))


def test_retrieve_empty_session_returns_empty(memory):
    results = memory.retrieve("no-such-session", "query", top_k=5)
    assert results == []


def test_store_and_retrieve_round_trip(memory):
    memory.store("s1", "LangGraph is a graph-based orchestration framework.", {})
    memory.store("s1", "FAISS is a vector similarity search library.", {})
    results = memory.retrieve("s1", "graph orchestration", top_k=5)
    assert len(results) >= 1
    assert any("LangGraph" in r for r in results)


def test_retrieve_respects_top_k(memory):
    for i in range(10):
        memory.store("s2", f"Document number {i} about machine learning.", {})
    results = memory.retrieve("s2", "machine learning", top_k=3)
    assert len(results) <= 3


def test_sessions_are_isolated(memory):
    memory.store("session-A", "Python is a programming language.", {})
    results = memory.retrieve("session-B", "Python programming", top_k=5)
    assert results == []


@given(st.lists(st.text(min_size=1, max_size=200), min_size=1, max_size=20))
@settings(max_examples=30, deadline=10000)
def test_retrieved_items_are_subset_of_stored(texts):
    """Property: retrieved items are always a subset of what was stored."""
    with tempfile.TemporaryDirectory() as tmp:
        mem = FAISSMemoryStore(persist_dir=tmp)
        session = "prop-test"
        for t in texts:
            mem.store(session, t, {})
        retrieved = mem.retrieve(session, texts[0], top_k=5)
        stored = mem._texts.get(session, [])
        assert all(r in stored for r in retrieved)
