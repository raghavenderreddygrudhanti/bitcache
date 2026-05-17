"""Benchmark: bitcache vs brute-force float32 search.

Measures recall, latency, and memory usage.
"""
import time

import numpy as np

from bitcache import BinaryIndex


def brute_force_search(query, database, k):
    scores = database @ query
    top_k = np.argpartition(scores, -k)[-k:]
    top_k_sorted = top_k[np.argsort(scores[top_k])[::-1]]
    return top_k_sorted


def recall_at_k(true_indices, predicted_indices, k):
    true_set = set(true_indices[:k])
    pred_set = set(predicted_indices[:k])
    return len(true_set & pred_set) / k


def main():
    dims = [384, 768, 1536]
    n_vectors = 100_000
    n_queries = 1000
    k = 10

    print(f"{'='*70}")
    print(f"bitcache benchmark: {n_vectors:,} vectors, {n_queries:,} queries, k={k}")
    print(f"{'='*70}")

    for dim in dims:
        print(f"\n--- dim={dim} ---")
        rng = np.random.default_rng(42)
        database = rng.standard_normal((n_vectors, dim)).astype(np.float32)
        database /= np.linalg.norm(database, axis=1, keepdims=True)
        queries = rng.standard_normal((n_queries, dim)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        # Build index
        t0 = time.time()
        index = BinaryIndex(dim=dim)
        index.add(database)
        build_time = time.time() - t0

        # Search
        t0 = time.time()
        bc_dists, bc_indices = index.search_batch(queries, k=k)
        search_time = time.time() - t0

        # Ground truth (brute force float32)
        t0 = time.time()
        gt_indices = []
        for q in queries:
            gt_indices.append(brute_force_search(q, database, k))
        gt_time = time.time() - t0
        gt_indices = np.array(gt_indices)

        # Recall
        recalls = []
        for i in range(n_queries):
            recalls.append(recall_at_k(gt_indices[i], bc_indices[i], k))
        mean_recall = np.mean(recalls)

        # Memory
        fp32_mb = (n_vectors * dim * 4) / (1024 * 1024)
        binary_mb = index.memory_usage_bytes / (1024 * 1024)

        print(f"  Build time:      {build_time:.2f}s")
        print(f"  Search time:     {search_time:.3f}s ({n_queries/search_time:.0f} QPS)")
        print(f"  Float32 search:  {gt_time:.3f}s ({n_queries/gt_time:.0f} QPS)")
        print(f"  Recall@{k}:       {mean_recall:.4f}")
        print(f"  Memory (fp32):   {fp32_mb:.1f} MB")
        print(f"  Memory (binary): {binary_mb:.1f} MB")
        print(f"  Compression:     {index.compression_ratio:.1f}x")


if __name__ == "__main__":
    main()
