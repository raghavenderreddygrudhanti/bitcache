"""Evaluation on real embeddings with FAISS baseline comparison.

Uses Cohere Wikipedia embeddings (768-dim) from HuggingFace.
Compares bitcache against FAISS IndexFlatIP and IndexBinaryFlat.
"""

import json
import os
import time

import faiss
import numpy as np

from bitcache import BinaryIndex, TwoStageIndex
from bitcache.quantize import quantize

N_VECTORS = 100_000
N_QUERIES = 1000
K = 10
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_real_embeddings():
    """Load Cohere Wikipedia embeddings from HuggingFace."""
    try:
        from datasets import load_dataset
        print("  Loading Cohere Wikipedia embeddings from HuggingFace...")
        ds = load_dataset(
            "Cohere/wikipedia-22-12-simple-embeddings",
            split="train",
            streaming=True,
        )
        vectors = []
        for i, row in enumerate(ds):
            vectors.append(row["emb"])
            if i >= N_VECTORS + N_QUERIES - 1:
                break
        vectors = np.array(vectors, dtype=np.float32)
        print(f"  Loaded {len(vectors)} vectors, dim={vectors.shape[1]}")
        return vectors
    except Exception as e:
        print(f"  Failed to load real data: {e}")
        print("  Falling back to synthetic clustered data...")
        return generate_clustered_data()


def generate_clustered_data():
    """Generate synthetic data that mimics real embeddings (clustered)."""
    rng = np.random.default_rng(42)
    dim = 768
    n_clusters = 100
    n_total = N_VECTORS + N_QUERIES

    # Generate cluster centers
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    # Generate points around centers
    vectors = []
    for i in range(n_total):
        cluster = i % n_clusters
        noise = rng.standard_normal(dim).astype(np.float32) * 0.3
        vec = centers[cluster] + noise
        vectors.append(vec)

    vectors = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= norms
    print(f"  Generated {len(vectors)} clustered vectors, dim={dim}")
    return vectors


def compute_ground_truth(queries, database, k):
    """Exact brute force search via FAISS."""
    dim = database.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(database)
    scores, indices = index.search(queries, k)
    return indices


def eval_faiss_binary(database_codes, query_codes, gt_indices, k):
    """FAISS IndexBinaryFlat baseline."""
    n_bits = database_codes.shape[1] * 8
    index = faiss.IndexBinaryFlat(n_bits)
    index.add(database_codes)

    t0 = time.time()
    _, indices = index.search(query_codes, k)
    search_time = time.time() - t0

    recall = _recall(gt_indices, indices, k)
    qps = N_QUERIES / search_time
    return recall, qps, search_time


def eval_faiss_hnsw(database, queries, gt_indices, k):
    """FAISS HNSW baseline."""
    dim = database.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)

    t0 = time.time()
    index.add(database)
    build_time = time.time() - t0

    index.hnsw.efSearch = 64
    t0 = time.time()
    _, indices = index.search(queries, k)
    search_time = time.time() - t0

    recall = _recall(gt_indices, indices, k)
    qps = N_QUERIES / search_time
    return recall, qps, search_time, build_time


def eval_bitcache_flat(database, queries, gt_indices, k):
    """bitcache BinaryIndex (flat scan)."""
    dim = database.shape[1]
    index = BinaryIndex(dim=dim)

    t0 = time.time()
    index.add(database)
    build_time = time.time() - t0

    t0 = time.time()
    _, indices = index.search_batch(queries, k=k)
    search_time = time.time() - t0

    recall = _recall(gt_indices, indices, k)
    qps = N_QUERIES / search_time
    return recall, qps, search_time, build_time


def eval_bitcache_two_stage(database, queries, gt_indices, k, rf):
    """bitcache TwoStageIndex."""
    dim = database.shape[1]
    index = TwoStageIndex(dim=dim, rerank_factor=rf)

    t0 = time.time()
    index.add(database)
    build_time = time.time() - t0

    t0 = time.time()
    _, indices = index.search_batch(queries, k=k)
    search_time = time.time() - t0

    recall = _recall(gt_indices, indices, k)
    qps = N_QUERIES / search_time
    return recall, qps, search_time, build_time


def _recall(gt, pred, k):
    n = len(gt)
    hits = 0
    for i in range(n):
        gt_set = set(gt[i, :k].tolist())
        pred_set = set(pred[i, :k].tolist()) if pred.ndim == 2 else set(pred[i])
        hits += len(gt_set & pred_set)
    return hits / (n * k)


def main():
    print("=" * 60)
    print("bitcache vs FAISS — Real Embedding Evaluation")
    print(f"N={N_VECTORS:,}, queries={N_QUERIES}, k={K}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    all_vectors = load_real_embeddings()
    database = all_vectors[:N_VECTORS]
    queries = all_vectors[N_VECTORS:N_VECTORS + N_QUERIES]
    dim = database.shape[1]

    # Memory stats
    fp32_mb = (N_VECTORS * dim * 4) / (1024**2)
    binary_mb = (N_VECTORS * ((dim + 7) // 8)) / (1024**2)
    print(f"\n  Float32 memory: {fp32_mb:.1f} MB")
    print(f"  Binary memory:  {binary_mb:.1f} MB")
    print(f"  Compression:    {fp32_mb/binary_mb:.1f}x")

    # Ground truth
    print("\nComputing ground truth (FAISS exact search)...")
    t0 = time.time()
    gt_indices = compute_ground_truth(queries, database, K)
    gt_time = time.time() - t0
    print(f"  Done in {gt_time:.1f}s ({N_QUERIES/gt_time:.0f} QPS)")

    results = {
        "dataset": "cohere-wikipedia-768" if dim == 768 else f"clustered-{dim}",
        "n_vectors": N_VECTORS,
        "n_queries": N_QUERIES,
        "dim": dim,
        "k": K,
        "memory_fp32_mb": round(fp32_mb, 1),
        "memory_binary_mb": round(binary_mb, 1),
        "compression": round(fp32_mb / binary_mb, 1),
    }

    # Prepare binary codes for FAISS binary
    database_codes = quantize(database)
    query_codes = quantize(queries)

    # --- FAISS Binary Flat ---
    print("\n[FAISS IndexBinaryFlat]")
    recall, qps, st = eval_faiss_binary(database_codes, query_codes, gt_indices, K)
    print(f"  Recall@{K}: {recall:.4f} | QPS: {qps:.0f} | Time: {st:.3f}s")
    results["faiss_binary_flat"] = {"recall": round(recall, 4), "qps": round(qps, 0)}

    # --- FAISS HNSW ---
    print("\n[FAISS IndexHNSWFlat]")
    recall, qps, st, bt = eval_faiss_hnsw(database, queries, gt_indices, K)
    print(f"  Recall@{K}: {recall:.4f} | QPS: {qps:.0f} | Build: {bt:.1f}s")
    results["faiss_hnsw"] = {"recall": round(recall, 4), "qps": round(qps, 0), "build_s": round(bt, 1)}

    # --- bitcache Binary Flat ---
    print("\n[bitcache BinaryIndex]")
    recall, qps, st, bt = eval_bitcache_flat(database, queries, gt_indices, K)
    print(f"  Recall@{K}: {recall:.4f} | QPS: {qps:.0f} | Build: {bt:.2f}s")
    results["bitcache_flat"] = {"recall": round(recall, 4), "qps": round(qps, 0), "build_s": round(bt, 2)}

    # --- bitcache Two-Stage ---
    for rf in [10, 50, 100]:
        print(f"\n[bitcache TwoStage rf={rf}]")
        recall, qps, st, bt = eval_bitcache_two_stage(database, queries, gt_indices, K, rf)
        print(f"  Recall@{K}: {recall:.4f} | QPS: {qps:.0f} | Build: {bt:.2f}s")
        results[f"bitcache_twostage_rf{rf}"] = {"recall": round(recall, 4), "qps": round(qps, 0), "build_s": round(bt, 2)}

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<30} {'Recall@10':>10} {'QPS':>8} {'Memory':>10}")
    print("-" * 60)
    print(f"{'FAISS exact (baseline)':<30} {'1.0000':>10} {N_QUERIES/gt_time:>8.0f} {fp32_mb:>8.1f} MB")
    print(f"{'FAISS binary flat':<30} {results['faiss_binary_flat']['recall']:>10.4f} {results['faiss_binary_flat']['qps']:>8.0f} {binary_mb:>8.1f} MB")
    print(f"{'FAISS HNSW':<30} {results['faiss_hnsw']['recall']:>10.4f} {results['faiss_hnsw']['qps']:>8.0f} {fp32_mb:>8.1f} MB")
    print(f"{'bitcache flat':<30} {results['bitcache_flat']['recall']:>10.4f} {results['bitcache_flat']['qps']:>8.0f} {binary_mb:>8.1f} MB")
    for rf in [10, 50, 100]:
        key = f"bitcache_twostage_rf{rf}"
        print(f"{'bitcache two-stage rf='+str(rf):<30} {results[key]['recall']:>10.4f} {results[key]['qps']:>8.0f} {'mixed':>10}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "eval_real_data.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
