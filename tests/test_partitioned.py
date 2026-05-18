import numpy as np
import pytest

from bitcache.partitioned import PartitionedIndex


def _make_clustered_data(n, dim, n_clusters=50, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    vectors = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        vectors[i] = centers[i % n_clusters] + rng.standard_normal(dim).astype(np.float32) * 0.3
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def _brute_force_top_k(query, database, k):
    scores = database @ query
    return np.argsort(scores)[::-1][:k]


def test_build():
    vectors = _make_clustered_data(1000, 128)
    index = PartitionedIndex(dim=128, n_partitions=10, n_probe=3)
    index.build(vectors)
    assert len(index) == 1000
    assert len(index.partition_sizes) == 10
    assert sum(index.partition_sizes) == 1000


def test_search_self_retrieval():
    vectors = _make_clustered_data(1000, 128)
    index = PartitionedIndex(dim=128, n_partitions=10, n_probe=5)
    index.build(vectors)

    scores, indices = index.search(vectors[0], k=5)
    assert len(scores) == 5
    assert indices[0] == 0  # self-match


def test_recall_vs_exhaustive():
    n = 5000
    dim = 128
    vectors = _make_clustered_data(n, dim)
    queries = vectors[:50]

    # Partitioned
    index = PartitionedIndex(dim=dim, n_partitions=20, n_probe=5, rerank_factor=50)
    index.build(vectors)

    # Measure recall
    recalls = []
    for i in range(50):
        true_top10 = set(_brute_force_top_k(queries[i], vectors, 10))
        _, pred = index.search(queries[i], k=10)
        hits = len(true_top10 & set(pred))
        recalls.append(hits / 10)

    mean_recall = np.mean(recalls)
    # Partitioned should get reasonable recall (not perfect, but > 30%)
    assert mean_recall > 0.3, f"recall too low: {mean_recall}"


def test_higher_nprobe_better_recall():
    n = 5000
    dim = 128
    vectors = _make_clustered_data(n, dim)
    query = vectors[42]
    true_top10 = set(_brute_force_top_k(query, vectors, 10))

    index_low = PartitionedIndex(dim=dim, n_partitions=20, n_probe=2)
    index_low.build(vectors)
    _, pred_low = index_low.search(query, k=10)
    recall_low = len(true_top10 & set(pred_low)) / 10

    index_high = PartitionedIndex(dim=dim, n_partitions=20, n_probe=10)
    index_high.build(vectors)
    _, pred_high = index_high.search(query, k=10)
    recall_high = len(true_top10 & set(pred_high)) / 10

    assert recall_high >= recall_low


def test_speedup_over_exhaustive():
    n = 10000
    dim = 128
    vectors = _make_clustered_data(n, dim)

    index = PartitionedIndex(dim=dim, n_partitions=32, n_probe=4)
    index.build(vectors)

    # Should scan ~1/8 of vectors (4/32 partitions)
    expected_scan = n * 4 / 32
    actual_scan = index.vectors_scanned_per_query
    assert actual_scan < n  # must be less than full scan
    assert actual_scan == pytest.approx(expected_scan, rel=0.1)


def test_batch_search():
    vectors = _make_clustered_data(2000, 64)
    index = PartitionedIndex(dim=64, n_partitions=10, n_probe=3)
    index.build(vectors)

    queries = vectors[:10]
    scores, indices = index.search_batch(queries, k=5)
    assert scores.shape == (10, 5)
    assert indices.shape == (10, 5)


def test_empty_index():
    index = PartitionedIndex(dim=128)
    scores, indices = index.search(np.random.randn(128).astype(np.float32), k=5)
    assert len(scores) == 0


def test_partition_balance():
    vectors = _make_clustered_data(10000, 128, n_clusters=32)
    index = PartitionedIndex(dim=128, n_partitions=32, n_probe=4)
    index.build(vectors)

    sizes = index.partition_sizes
    # No partition should be empty
    assert all(s > 0 for s in sizes)
    # No partition should have more than 3x average
    avg = 10000 / 32
    assert max(sizes) < avg * 3
