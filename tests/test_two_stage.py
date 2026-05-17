import numpy as np
import pytest

from bitcache import TwoStageIndex


def _make_data(n, dim, seed=42):
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def _brute_force_top_k(query, database, k):
    scores = database @ query
    top_k = np.argsort(scores)[::-1][:k]
    return top_k


def test_basic_search():
    dim = 128
    vectors = _make_data(1000, dim)

    index = TwoStageIndex(dim=dim)
    index.add(vectors)

    scores, indices = index.search(vectors[0], k=5)
    assert len(scores) == 5
    assert indices[0] == 0  # self-match


def test_recall_vs_flat_binary():
    dim = 128
    n = 5000
    vectors = _make_data(n, dim)

    index = TwoStageIndex(dim=dim, rerank_factor=10)
    index.add(vectors)

    # Measure recall over 100 queries
    queries = vectors[:100]
    recalls = []
    for i in range(100):
        true_top10 = set(_brute_force_top_k(queries[i], vectors, 10))
        _, pred_indices = index.search(queries[i], k=10)
        hits = len(true_top10 & set(pred_indices))
        recalls.append(hits / 10)

    mean_recall = np.mean(recalls)
    # Two-stage should achieve much higher recall than flat binary (6%)
    assert mean_recall >= 0.5, f"recall too low: {mean_recall}"


def test_higher_rerank_factor_better_recall():
    dim = 128
    n = 5000
    vectors = _make_data(n, dim)

    index_low = TwoStageIndex(dim=dim, rerank_factor=5)
    index_low.add(vectors)

    index_high = TwoStageIndex(dim=dim, rerank_factor=50)
    index_high.add(vectors)

    query = vectors[42]
    true_top10 = set(_brute_force_top_k(query, vectors, 10))

    _, indices_low = index_low.search(query, k=10)
    _, indices_high = index_high.search(query, k=10)

    recall_low = len(true_top10 & set(indices_low)) / 10
    recall_high = len(true_top10 & set(indices_high)) / 10

    assert recall_high >= recall_low


def test_batch_search():
    dim = 64
    vectors = _make_data(500, dim)

    index = TwoStageIndex(dim=dim)
    index.add(vectors)

    queries = vectors[:10]
    scores, indices = index.search_batch(queries, k=5)
    assert scores.shape == (10, 5)
    assert indices.shape == (10, 5)

    # Each query should find itself
    for i in range(10):
        assert indices[i, 0] == i


def test_empty_index():
    index = TwoStageIndex(dim=128)
    scores, indices = index.search(np.random.randn(128).astype(np.float32), k=5)
    assert len(scores) == 0


def test_incremental_add():
    dim = 64
    index = TwoStageIndex(dim=dim)

    batch1 = _make_data(100, dim, seed=1)
    batch2 = _make_data(100, dim, seed=2)

    index.add(batch1)
    assert len(index) == 100

    index.add(batch2)
    assert len(index) == 200


def test_scores_are_inner_products():
    dim = 64
    vectors = _make_data(100, dim)

    index = TwoStageIndex(dim=dim, rerank_factor=100)
    index.add(vectors)

    query = vectors[0]
    scores, indices = index.search(query, k=1)

    # Self-match score should be ~1.0 (cosine similarity of identical unit vectors)
    assert scores[0] > 0.99
