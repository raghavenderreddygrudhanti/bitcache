import numpy as np
import pytest

from bitcache.float_routed import FloatRoutedIndex


def _make_clustered(n, dim, n_clusters=50, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    vectors = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        vectors[i] = centers[i % n_clusters] + rng.standard_normal(dim).astype(np.float32) * 0.3
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def _brute_force(query, database, k):
    scores = database @ query
    return np.argsort(scores)[::-1][:k]


def test_build():
    vectors = _make_clustered(1000, 128)
    index = FloatRoutedIndex(dim=128, n_partitions=10, n_probe=3)
    index.build(vectors)
    assert len(index) == 1000
    assert sum(index.partition_sizes) == 1000


def test_self_retrieval():
    vectors = _make_clustered(1000, 128)
    index = FloatRoutedIndex(dim=128, n_partitions=10, n_probe=5)
    index.build(vectors)
    scores, indices = index.search(vectors[0], k=5)
    assert indices[0] == 0


def test_recall():
    vectors = _make_clustered(5000, 128)
    index = FloatRoutedIndex(dim=128, n_partitions=20, n_probe=5, rerank_factor=50)
    index.build(vectors)

    recalls = []
    for i in range(50):
        true_top10 = set(_brute_force(vectors[i], vectors, 10))
        _, pred = index.search(vectors[i], k=10)
        recalls.append(len(true_top10 & set(pred)) / 10)

    assert np.mean(recalls) > 0.5


def test_higher_nprobe_better():
    vectors = _make_clustered(5000, 128)
    query = vectors[42]
    true_top10 = set(_brute_force(query, vectors, 10))

    idx_low = FloatRoutedIndex(dim=128, n_partitions=20, n_probe=2)
    idx_low.build(vectors)
    _, pred_low = idx_low.search(query, k=10)
    r_low = len(true_top10 & set(pred_low)) / 10

    idx_high = FloatRoutedIndex(dim=128, n_partitions=20, n_probe=10)
    idx_high.build(vectors)
    _, pred_high = idx_high.search(query, k=10)
    r_high = len(true_top10 & set(pred_high)) / 10

    assert r_high >= r_low


def test_batch_search():
    vectors = _make_clustered(2000, 64)
    index = FloatRoutedIndex(dim=64, n_partitions=10, n_probe=3)
    index.build(vectors)
    scores, indices = index.search_batch(vectors[:10], k=5)
    assert scores.shape == (10, 5)
    assert indices.shape == (10, 5)


def test_empty():
    index = FloatRoutedIndex(dim=128)
    scores, indices = index.search(np.random.randn(128).astype(np.float32), k=5)
    assert len(scores) == 0


def test_partition_balance():
    vectors = _make_clustered(10000, 128, n_clusters=32)
    index = FloatRoutedIndex(dim=128, n_partitions=32, n_probe=4)
    index.build(vectors)
    sizes = index.partition_sizes
    assert all(s > 0 for s in sizes)
    avg = 10000 / 32
    assert max(sizes) < avg * 4
