import numpy as np
import pytest

from bitcache import VamanaIndex


def test_build_basic():
    dim = 64
    n = 500
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)

    index = VamanaIndex(dim=dim, R=16, L_build=30)
    index.build(vectors)
    assert len(index) == n


def test_search_self_retrieval():
    dim = 64
    n = 200
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)

    index = VamanaIndex(dim=dim, R=16, L_build=30)
    index.build(vectors)

    # Query with first vector — should find itself
    scores, indices = index.search(vectors[0], k=5, ef=30)
    assert indices[0] == 0


def test_search_recall():
    dim = 128
    n = 1000
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    index = VamanaIndex(dim=dim, R=32, L_build=50, alpha=1.2)
    index.build(vectors)

    # Ground truth: brute force
    query = vectors[0]
    true_scores = vectors @ query
    true_top10 = np.argsort(true_scores)[::-1][:10]

    # Graph search
    scores, indices = index.search(query, k=10, ef=50)

    # Check recall (should be significantly better than flat binary)
    hits = len(set(true_top10) & set(indices))
    recall = hits / 10
    assert recall >= 0.3  # graph should get at least 30% recall


def test_search_no_rerank():
    dim = 64
    n = 200
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)

    index = VamanaIndex(dim=dim, R=16, L_build=30)
    index.build(vectors)

    dists, indices = index.search(vectors[0], k=5, ef=30, rerank=False)
    assert len(indices) == 5
    assert dists[0] <= dists[-1]  # sorted ascending


def test_higher_ef_better_recall():
    dim = 128
    n = 1000
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    index = VamanaIndex(dim=dim, R=32, L_build=50)
    index.build(vectors)

    query = vectors[42]
    true_scores = vectors @ query
    true_top10 = set(np.argsort(true_scores)[::-1][:10])

    _, indices_low = index.search(query, k=10, ef=20)
    _, indices_high = index.search(query, k=10, ef=100)

    recall_low = len(true_top10 & set(indices_low)) / 10
    recall_high = len(true_top10 & set(indices_high)) / 10

    assert recall_high >= recall_low


def test_empty_index():
    index = VamanaIndex(dim=64)
    scores, indices = index.search(np.random.randn(64).astype(np.float32), k=5)
    assert len(scores) == 0
    assert len(indices) == 0
