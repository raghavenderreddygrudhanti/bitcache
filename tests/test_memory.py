import time
from unittest.mock import patch

import numpy as np
import pytest

from bitcache import AgentMemory


def _vec(seed, dim=64):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_save_and_retrieve():
    mem = AgentMemory(dim=64)
    vec = _vec(1)
    mid = mem.save_memory(vec, content="hello world", importance=0.8)

    results = mem.retrieve_memory(vec, k=1)
    assert len(results) == 1
    assert results[0]["id"] == mid
    assert results[0]["content"] == "hello world"
    assert results[0]["importance"] >= 0.79  # reinforced on retrieval


def test_reinforcement_on_retrieval():
    mem = AgentMemory(dim=64, reinforce_amount=0.1)
    vec = _vec(1)
    mid = mem.save_memory(vec, content="test", importance=0.5)

    mem.retrieve_memory(vec, k=1)
    state = mem._memory_state[mid]
    assert state["access_count"] == 1
    assert state["importance"] > 0.5


def test_manual_reinforce():
    mem = AgentMemory(dim=64)
    vec = _vec(1)
    mid = mem.save_memory(vec, content="test", importance=0.3)

    mem.reinforce_memory(mid, amount=0.5)
    assert mem._memory_state[mid]["importance"] == 0.8


def test_importance_capped_at_1():
    mem = AgentMemory(dim=64)
    vec = _vec(1)
    mid = mem.save_memory(vec, content="test", importance=0.9)

    mem.reinforce_memory(mid, amount=0.5)
    assert mem._memory_state[mid]["importance"] == 1.0


def test_forget_memory():
    mem = AgentMemory(dim=64)
    vec = _vec(1)
    mid = mem.save_memory(vec, content="forget me")

    assert mem.forget_memory(mid) is True
    assert len(mem) == 0
    assert mem.get_memory(mid) is None


def test_eviction_at_capacity():
    mem = AgentMemory(dim=64, capacity=5)
    vecs = [_vec(i) for i in range(7)]

    for i, v in enumerate(vecs):
        mem.save_memory(v, content=f"mem-{i}", importance=i * 0.1)

    # Should have evicted 2 lowest importance memories
    assert len(mem) == 5


def test_min_importance_filter():
    mem = AgentMemory(dim=64)
    vecs = [_vec(i) for i in range(5)]

    for i, v in enumerate(vecs):
        mem.save_memory(v, content=f"mem-{i}", importance=i * 0.2)

    # Only retrieve memories with importance >= 0.5
    results = mem.retrieve_memory(vecs[4], k=10, min_importance=0.5)
    for r in results:
        assert r["importance"] >= 0.5


def test_get_memory():
    mem = AgentMemory(dim=64)
    vec = _vec(1)
    mid = mem.save_memory(vec, content="stored", importance=0.7, metadata={"tag": "test"})

    info = mem.get_memory(mid)
    assert info is not None
    assert info["importance"] == 0.7
    assert info["access_count"] == 0


def test_get_stats():
    mem = AgentMemory(dim=64, capacity=100)
    vecs = [_vec(i) for i in range(10)]

    for i, v in enumerate(vecs):
        mem.save_memory(v, content=f"mem-{i}", importance=(i + 1) * 0.1)

    stats = mem.get_stats()
    assert stats["total"] == 10
    assert stats["capacity"] == 100
    assert stats["min_importance"] == pytest.approx(0.1, abs=0.01)
    assert stats["max_importance"] == pytest.approx(1.0, abs=0.01)


def test_decay_reduces_importance():
    mem = AgentMemory(dim=64, decay_rate=1.0)  # aggressive decay
    vec = _vec(1)
    mid = mem.save_memory(vec, content="old memory", importance=0.8)

    # Simulate 2 days passing
    mem._memory_state[mid]["last_accessed"] = time.time() - 2 * 86400

    mem._apply_decay()
    assert mem._memory_state[mid]["importance"] < 0.8


def test_empty_memory():
    mem = AgentMemory(dim=64)
    results = mem.retrieve_memory(_vec(1), k=5)
    assert results == []


def test_multiple_saves_same_content():
    mem = AgentMemory(dim=64)
    vec = _vec(1)

    id1 = mem.save_memory(vec, content="version 1", id="doc-1")
    id2 = mem.save_memory(vec, content="version 2", id="doc-1")

    assert id1 == id2
    assert len(mem) == 1
