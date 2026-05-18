"""Full evaluation suite for bitcache — 1M vectors.

Metrics:
1. Memory usage (bytes) — binary vs float32
2. Recall@k — how many true top-k are found
3. Search latency (QPS) — queries per second
4. Build time — index construction time
5. Two-stage recall improvement over flat binary
6. Streaming throughput — inserts/sec, deletes/sec
7. Memory decay behavior — importance over time

Run: python benchmarks/eval_full.py
"""

import json
import os
import time

import numpy as np

# Adjust these for your machine's RAM
N_VECTORS = 100_000
N_QUERIES = 1000
DIM = 768
K = 10
SEED = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def generate_data(n, dim, seed=SEED):
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= norms
    return vectors


def brute_force_top_k(queries, database, k):
    """Ground truth via float32 matmul."""
    # Process in chunks to avoid OOM
    chunk_size = 100
    all_indices = []
    for i in range(0, len(queries), chunk_size):
        chunk = queries[i:i+chunk_size]
        scores = chunk @ database.T
        top_k = np.argpartition(scores, -k, axis=1)[:, -k:]
        # Sort within top-k
        for j in range(len(chunk)):
            order = np.argsort(scores[j, top_k[j]])[::-1]
            top_k[j] = top_k[j][order]
        all_indices.append(top_k)
    return np.vstack(all_indices)


def eval_memory_usage():
    """Metric 1: Memory compression."""
    print("\n" + "="*60)
    print("METRIC 1: Memory Usage")
    print("="*60)

    fp32_bytes = N_VECTORS * DIM * 4
    binary_bytes = N_VECTORS * ((DIM + 7) // 8)

    fp32_mb = fp32_bytes / (1024**2)
    binary_mb = binary_bytes / (1024**2)
    ratio = fp32_bytes / binary_bytes

    print(f"  Vectors:     {N_VECTORS:,}")
    print(f"  Dimensions:  {DIM}")
    print(f"  Float32:     {fp32_mb:.1f} MB")
    print(f"  Binary:      {binary_mb:.1f} MB")
    print(f"  Compression: {ratio:.1f}x")

    return {
        "n_vectors": N_VECTORS,
        "dim": DIM,
        "fp32_mb": round(fp32_mb, 1),
        "binary_mb": round(binary_mb, 1),
        "compression_ratio": round(ratio, 1),
    }


def eval_recall_and_latency(database, queries, gt_indices):
    """Metric 2 & 3: Recall@k and search latency."""
    from bitcache import BinaryIndex, TwoStageIndex

    print("\n" + "="*60)
    print("METRIC 2 & 3: Recall and Latency")
    print("="*60)

    results = {}

    # --- Flat Binary ---
    print("\n  [BinaryIndex] Building...")
    t0 = time.time()
    bi = BinaryIndex(dim=DIM)
    bi.add(database)
    build_time = time.time() - t0
    print(f"  Build time: {build_time:.2f}s")

    print("  Searching...")
    t0 = time.time()
    _, bi_indices = bi.search_batch(queries, k=K)
    search_time = time.time() - t0
    qps = N_QUERIES / search_time

    recall = _compute_recall(gt_indices, bi_indices, K)
    print(f"  Recall@{K}: {recall:.4f}")
    print(f"  Latency: {search_time:.3f}s ({qps:.0f} QPS)")

    results["binary_flat"] = {
        "build_time_s": round(build_time, 2),
        "search_time_s": round(search_time, 3),
        "qps": round(qps, 0),
        "recall_at_k": round(recall, 4),
    }

    # --- Two-Stage (rerank_factor=10) ---
    for rf in [10, 50, 100]:
        print(f"\n  [TwoStageIndex rerank_factor={rf}] Building...")
        t0 = time.time()
        ts = TwoStageIndex(dim=DIM, rerank_factor=rf)
        ts.add(database)
        build_time = time.time() - t0
        print(f"  Build time: {build_time:.2f}s")

        print("  Searching...")
        t0 = time.time()
        _, ts_indices = ts.search_batch(queries, k=K)
        search_time = time.time() - t0
        qps = N_QUERIES / search_time

        recall = _compute_recall(gt_indices, ts_indices, K)
        print(f"  Recall@{K}: {recall:.4f}")
        print(f"  Latency: {search_time:.3f}s ({qps:.0f} QPS)")

        results[f"two_stage_rf{rf}"] = {
            "rerank_factor": rf,
            "build_time_s": round(build_time, 2),
            "search_time_s": round(search_time, 3),
            "qps": round(qps, 0),
            "recall_at_k": round(recall, 4),
        }

    return results


def eval_streaming_throughput():
    """Metric 6: Insert and delete throughput."""
    from bitcache import StreamingIndex

    print("\n" + "="*60)
    print("METRIC 6: Streaming Throughput")
    print("="*60)

    index = StreamingIndex(dim=DIM)
    rng = np.random.default_rng(SEED)

    # Insert throughput
    n_insert = 100_000
    vectors = rng.standard_normal((n_insert, DIM)).astype(np.float32)

    t0 = time.time()
    ids = index.insert_batch(vectors)
    insert_time = time.time() - t0
    insert_rate = n_insert / insert_time

    print(f"  Insert {n_insert:,} vectors: {insert_time:.2f}s ({insert_rate:.0f} vec/s)")

    # Delete throughput
    n_delete = 10_000
    delete_ids = ids[:n_delete]

    t0 = time.time()
    for did in delete_ids:
        index.delete(did)
    delete_time = time.time() - t0
    delete_rate = n_delete / delete_time

    print(f"  Delete {n_delete:,} vectors: {delete_time:.2f}s ({delete_rate:.0f} vec/s)")

    # Search after mutations
    query = vectors[n_delete + 1]
    t0 = time.time()
    for _ in range(100):
        index.search(query, k=10)
    search_time = time.time() - t0
    search_qps = 100 / search_time

    print(f"  Search after mutations: {search_qps:.0f} QPS")

    return {
        "n_insert": n_insert,
        "insert_time_s": round(insert_time, 2),
        "insert_rate_per_s": round(insert_rate, 0),
        "n_delete": n_delete,
        "delete_time_s": round(delete_time, 2),
        "delete_rate_per_s": round(delete_rate, 0),
        "search_qps_after_mutations": round(search_qps, 0),
    }


def eval_memory_decay():
    """Metric 7: Memory prioritization behavior."""
    from bitcache import AgentMemory

    print("\n" + "="*60)
    print("METRIC 7: Memory Decay Behavior")
    print("="*60)

    mem = AgentMemory(dim=DIM, capacity=1000, decay_rate=0.1, reinforce_amount=0.15)
    rng = np.random.default_rng(SEED)

    # Insert memories with varying importance
    for i in range(100):
        vec = rng.standard_normal(DIM).astype(np.float32)
        importance = rng.uniform(0.1, 1.0)
        mem.save_memory(vec, content=f"memory-{i}", importance=importance)

    stats_before = mem.get_stats()
    print(f"  Before decay: mean_importance={stats_before['mean_importance']:.3f}")

    # Simulate time passing (force decay)
    now = time.time()
    for state in mem._memory_state.values():
        state["last_accessed"] = now - 7 * 86400  # 7 days ago

    mem._apply_decay()
    stats_after = mem.get_stats()
    print(f"  After 7-day decay: mean_importance={stats_after['mean_importance']:.3f}")

    # Retrieve some memories (reinforces them)
    query = rng.standard_normal(DIM).astype(np.float32)
    results = mem.retrieve_memory(query, k=5)
    reinforced_ids = [r["id"] for r in results]

    stats_final = mem.get_stats()
    print(f"  After retrieval (5 reinforced): mean_importance={stats_final['mean_importance']:.3f}")
    print(f"  Total accesses: {stats_final['total_accesses']}")

    return {
        "n_memories": 100,
        "mean_importance_initial": round(stats_before["mean_importance"], 3),
        "mean_importance_after_7d_decay": round(stats_after["mean_importance"], 3),
        "mean_importance_after_reinforce": round(stats_final["mean_importance"], 3),
        "decay_rate": 0.1,
        "reinforce_amount": 0.15,
    }


def _compute_recall(gt_indices, pred_indices, k):
    """Compute mean recall@k."""
    n = len(gt_indices)
    hits = 0
    for i in range(n):
        gt_set = set(gt_indices[i, :k])
        pred_set = set(pred_indices[i, :k]) if pred_indices.ndim == 2 else set(pred_indices[i])
        hits += len(gt_set & pred_set)
    return hits / (n * k)


def main():
    print(f"bitcache Evaluation Suite")
    print(f"N={N_VECTORS:,}, dim={DIM}, queries={N_QUERIES}, k={K}")
    print(f"{'='*60}")

    all_results = {}

    # Metric 1: Memory
    all_results["memory"] = eval_memory_usage()

    # Generate data
    print("\nGenerating data...")
    t0 = time.time()
    database = generate_data(N_VECTORS, DIM)
    queries = generate_data(N_QUERIES, DIM, seed=99)
    print(f"  Data generated in {time.time()-t0:.1f}s")

    # Ground truth
    print("\nComputing ground truth (brute force float32)...")
    t0 = time.time()
    gt_indices = brute_force_top_k(queries, database, K)
    gt_time = time.time() - t0
    print(f"  Ground truth computed in {gt_time:.1f}s ({N_QUERIES/gt_time:.0f} QPS)")
    all_results["ground_truth_qps"] = round(N_QUERIES / gt_time, 0)

    # Metric 2 & 3: Recall and Latency
    all_results["search"] = eval_recall_and_latency(database, queries, gt_indices)

    # Free memory before streaming test
    del database, queries, gt_indices

    # Metric 6: Streaming
    all_results["streaming"] = eval_streaming_throughput()

    # Metric 7: Memory decay
    all_results["memory_decay"] = eval_memory_decay()

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "eval_full.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*60}")
    print("ALL RESULTS SAVED")
    print(f"{'='*60}")
    print(f"File: {out_path}")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
