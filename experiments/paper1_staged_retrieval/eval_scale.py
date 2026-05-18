"""Scale benchmark: find where exhaustive binary scan stops being practical.

Tests 50K, 500K, 5M vectors at rf=100 and rf=500.
Measures recall, latency (avg, p50, p95, p99), QPS, build time, memory.
"""

import csv
import json
import os
import time
import gc

import faiss
import matplotlib.pyplot as plt
import numpy as np

from bitcache import TwoStageIndex
from bitcache.quantize import quantize

DIM = 768
N_QUERIES = 100
K = 10
SEED = 42
DATASET_SIZES = [50_000, 500_000, 5_000_000]
RF_VALUES = [100, 500]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def generate_data(n, seed=SEED):
    rng = np.random.default_rng(seed)
    n_clusters = 200
    centers = rng.standard_normal((n_clusters, DIM)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    vectors = np.zeros((n, DIM), dtype=np.float32)
    for i in range(n):
        vectors[i] = centers[i % n_clusters] + rng.standard_normal(DIM).astype(np.float32) * 0.3
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= norms
    return vectors


def ground_truth_sample(queries, database, k):
    """Compute ground truth on a sample for large datasets."""
    dim = database.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(database)
    _, indices = index.search(queries, k)
    return indices


def recall_at_k(gt, pred, k):
    hits = 0
    for i in range(len(gt)):
        gt_set = set(gt[i, :k].tolist())
        pred_set = set(pred[i, :k].tolist())
        hits += len(gt_set & pred_set)
    return hits / (len(gt) * k)


def main():
    print("=" * 70)
    print("SCALE BENCHMARK — Where does exhaustive binary scan break?")
    print(f"Sizes: {[f'{s:,}' for s in DATASET_SIZES]}")
    print(f"rf values: {RF_VALUES}, queries={N_QUERIES}, dim={DIM}, k={K}")
    print("=" * 70)

    results = []

    for n in DATASET_SIZES:
        print(f"\n{'='*70}")
        print(f"DATASET SIZE: {n:,} vectors")
        print(f"{'='*70}")

        # Generate data
        print(f"  Generating {n:,} vectors...")
        t0 = time.time()
        database = generate_data(n)
        gen_time = time.time() - t0
        print(f"  Generated in {gen_time:.1f}s")

        # Queries from different seed
        queries = generate_data(N_QUERIES, seed=99)[:N_QUERIES]

        # Ground truth
        print(f"  Computing ground truth...")
        t0 = time.time()
        gt = ground_truth_sample(queries, database, K)
        gt_time = time.time() - t0
        print(f"  Ground truth in {gt_time:.1f}s")

        for rf in RF_VALUES:
            print(f"\n  --- rf={rf} ---")

            # Build
            t0 = time.time()
            index = TwoStageIndex(dim=DIM, rerank_factor=rf)
            index.add(database)
            build_time = time.time() - t0
            print(f"  Build time: {build_time:.2f}s")

            # Memory
            binary_mb = index._codes.nbytes / (1024**2) if index._codes is not None else 0
            float_mb = index._vectors.nbytes / (1024**2) if index._vectors is not None else 0
            total_mb = binary_mb + float_mb

            # Search with latency tracking
            latencies = []
            all_preds = []

            for i in range(N_QUERIES):
                t0 = time.perf_counter()
                _, pred = index.search(queries[i], k=K)
                elapsed = (time.perf_counter() - t0) * 1000  # ms
                latencies.append(elapsed)
                all_preds.append(pred)

            all_preds = np.array(all_preds)
            latencies = np.array(latencies)

            # Metrics
            r = recall_at_k(gt, all_preds, K)
            avg_ms = float(np.mean(latencies))
            p50_ms = float(np.percentile(latencies, 50))
            p95_ms = float(np.percentile(latencies, 95))
            p99_ms = float(np.percentile(latencies, 99))
            qps = 1000.0 / avg_ms
            n_candidates = min(K * rf, n)
            float_ops = n_candidates * DIM

            print(f"  Recall@10:  {r:.4f}")
            print(f"  Avg latency: {avg_ms:.1f}ms | p50={p50_ms:.1f}ms | p95={p95_ms:.1f}ms | p99={p99_ms:.1f}ms")
            print(f"  QPS: {qps:.1f}")
            print(f"  Memory: binary={binary_mb:.1f}MB, float={float_mb:.1f}MB, total={total_mb:.1f}MB")

            results.append({
                "n_vectors": n,
                "rf": rf,
                "recall_at_10": round(r, 4),
                "avg_latency_ms": round(avg_ms, 1),
                "p50_latency_ms": round(p50_ms, 1),
                "p95_latency_ms": round(p95_ms, 1),
                "p99_latency_ms": round(p99_ms, 1),
                "qps": round(qps, 1),
                "build_time_s": round(build_time, 2),
                "memory_binary_mb": round(binary_mb, 1),
                "memory_float_mb": round(float_mb, 1),
                "memory_total_mb": round(total_mb, 1),
                "candidates_reranked": n_candidates,
                "float_ops": float_ops,
            })

            # Free memory
            del index
            gc.collect()

        del database
        gc.collect()

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    json_path = os.path.join(RESULTS_DIR, "eval_scale.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(RESULTS_DIR, "eval_scale.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Plots
    for rf in RF_VALUES:
        subset = [r for r in results if r["rf"] == rf]
        sizes = [r["n_vectors"] for r in subset]
        lats = [r["avg_latency_ms"] for r in subset]
        recalls = [r["recall_at_10"] for r in subset]
        qps_vals = [r["qps"] for r in subset]

        # Latency vs scale
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sizes, lats, "o-", linewidth=2, markersize=8)
        ax.set_xscale("log")
        ax.set_xlabel("Dataset Size")
        ax.set_ylabel("Average Latency (ms)")
        ax.set_title(f"Latency vs Scale (rf={rf})")
        ax.grid(True, alpha=0.3)
        for i, (s, l) in enumerate(zip(sizes, lats)):
            ax.annotate(f"{l:.0f}ms", (s, l), fontsize=9, textcoords="offset points", xytext=(5, 5))
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"latency_vs_scale_rf{rf}.png"), dpi=150)

        # QPS vs scale
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sizes, qps_vals, "s-", linewidth=2, markersize=8, color="#FF9800")
        ax.set_xscale("log")
        ax.set_xlabel("Dataset Size")
        ax.set_ylabel("QPS")
        ax.set_title(f"Throughput vs Scale (rf={rf})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"qps_vs_scale_rf{rf}.png"), dpi=150)

    # Combined recall vs scale
    fig, ax = plt.subplots(figsize=(8, 5))
    for rf in RF_VALUES:
        subset = [r for r in results if r["rf"] == rf]
        sizes = [r["n_vectors"] for r in subset]
        recalls = [r["recall_at_10"] for r in subset]
        ax.plot(sizes, recalls, "o-", linewidth=2, markersize=8, label=f"rf={rf}")
    ax.set_xscale("log")
    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("Recall@10")
    ax.set_title("Recall vs Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "recall_vs_scale.png"), dpi=150)

    # Summary table
    print(f"\n\n{'='*70}")
    print("SCALE TEST SUMMARY")
    print(f"{'='*70}")
    print(f"{'Size':>10} {'rf':>4} {'Recall':>8} {'Latency':>10} {'p95':>8} {'p99':>8} {'QPS':>7} {'Memory':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['n_vectors']:>10,} {r['rf']:>4} {r['recall_at_10']:>8.4f} {r['avg_latency_ms']:>8.1f}ms {r['p95_latency_ms']:>6.1f}ms {r['p99_latency_ms']:>6.1f}ms {r['qps']:>7.1f} {r['memory_total_mb']:>6.1f}MB")

    print(f"\nResults saved to {json_path}")
    print(f"Charts saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
