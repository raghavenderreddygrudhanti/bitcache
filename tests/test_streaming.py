import numpy as np
import pytest

from bitcache import StreamingIndex


def test_insert_and_search():
    index = StreamingIndex(dim=64)
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(64).astype(np.float32)

    vid = index.insert(vec, id="doc-1", metadata={"source": "web"})
    assert vid == "doc-1"
    assert len(index) == 1

    scores, ids, metas = index.search(vec, k=1)
    assert ids[0] == "doc-1"
    assert metas[0]["source"] == "web"
    assert scores[0] > 0.99


def test_insert_batch():
    index = StreamingIndex(dim=64)
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((100, 64)).astype(np.float32)

    ids = index.insert_batch(vectors, ids=[f"doc-{i}" for i in range(100)])
    assert len(index) == 100
    assert ids[0] == "doc-0"


def test_delete():
    index = StreamingIndex(dim=64)
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((10, 64)).astype(np.float32)

    index.insert_batch(vectors, ids=[f"doc-{i}" for i in range(10)])
    assert len(index) == 10

    assert index.delete("doc-5") is True
    assert len(index) == 9
    assert index.delete("doc-5") is False  # already deleted

    # Search should not return deleted vector
    scores, ids, _ = index.search(vectors[5], k=10)
    assert "doc-5" not in ids


def test_update_vector():
    index = StreamingIndex(dim=64)
    rng = np.random.default_rng(42)
    vec1 = rng.standard_normal(64).astype(np.float32)
    vec2 = rng.standard_normal(64).astype(np.float32)

    index.insert(vec1, id="doc-1")
    assert index.update("doc-1", vector=vec2) is True

    # Search with vec2 should find doc-1
    scores, ids, _ = index.search(vec2, k=1)
    assert ids[0] == "doc-1"
    assert scores[0] > 0.99


def test_update_metadata():
    index = StreamingIndex(dim=64)
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(64).astype(np.float32)

    index.insert(vec, id="doc-1", metadata={"version": 1})
    index.update("doc-1", metadata={"version": 2})

    info = index.get("doc-1")
    assert info["metadata"]["version"] == 2


def test_metadata_filter():
    index = StreamingIndex(dim=64)
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((10, 64)).astype(np.float32)

    for i in range(10):
        tenant = "acme" if i < 5 else "globex"
        index.insert(vectors[i], id=f"doc-{i}", metadata={"tenant": tenant})

    # Filter by tenant
    scores, ids, metas = index.search(vectors[0], k=10, filter={"tenant": "acme"})
    assert len(ids) == 5
    assert all(m["tenant"] == "acme" for m in metas)


def test_slot_reuse_after_delete():
    index = StreamingIndex(dim=64)
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((5, 64)).astype(np.float32)

    index.insert_batch(vectors, ids=[f"doc-{i}" for i in range(5)])
    index.delete("doc-2")
    assert len(index) == 4

    # Insert new vector — should reuse slot
    new_vec = rng.standard_normal(64).astype(np.float32)
    index.insert(new_vec, id="doc-new")
    assert len(index) == 5


def test_auto_id_generation():
    index = StreamingIndex(dim=64)
    rng = np.random.default_rng(42)

    id1 = index.insert(rng.standard_normal(64).astype(np.float32))
    id2 = index.insert(rng.standard_normal(64).astype(np.float32))
    assert id1 != id2
    assert id1.startswith("vec_")


def test_save_load(tmp_path):
    index = StreamingIndex(dim=64)
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((50, 64)).astype(np.float32)

    index.insert_batch(vectors, ids=[f"doc-{i}" for i in range(50)])
    index.delete("doc-10")
    index.save(str(tmp_path / "idx"))

    loaded = StreamingIndex.load(str(tmp_path / "idx"))
    assert len(loaded) == 49
    assert loaded.get("doc-10") is None
    assert loaded.get("doc-0") is not None


def test_get_returns_none_for_missing():
    index = StreamingIndex(dim=64)
    assert index.get("nonexistent") is None


def test_insert_existing_id_updates():
    index = StreamingIndex(dim=64)
    rng = np.random.default_rng(42)
    vec1 = rng.standard_normal(64).astype(np.float32)
    vec2 = rng.standard_normal(64).astype(np.float32)

    index.insert(vec1, id="doc-1", metadata={"v": 1})
    index.insert(vec2, id="doc-1", metadata={"v": 2})

    assert len(index) == 1
    info = index.get("doc-1")
    assert info["metadata"]["v"] == 2
