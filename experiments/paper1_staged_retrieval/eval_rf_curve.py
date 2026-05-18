"""Recall-vs-RF curve benchmark.

Measures recall, latency (avg, p50, p95), QPS, and rerank cost
across multiple rerank factors. Produces the central tradeoff graph.
"""

import csv
import json
import os
import time

import faiss
import matplotlib.pyplot as plt
import numpy as np

from bitcache import TwoStageIndex

N_VECTORS = 50_000
N_QUERIES = 100
DIM = 768
K = 10
SEED = 42
RF_VALUES = [10, 25, 50, 100, 200, 500, 1000]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def generate_data():
    rng = np.random.default_rng(SEED)
    n_clusters = 100
    centers = rng.standard_normal((n_clusters, DIM)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    n_total = N_VECTORS + N_QUERIES
    vectors = np.zeros((n_total, DIM), dtype=np.float32)
    for i in range(n_total):
        vectors[i] = centers[i % n_clusters] + rng.standard_normal(DIM).astype(np.float32) * 0.3
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= norms
    return vectors[:N_VECTORS], vectors[N_VECTORS:]


def ground_truth(queries, database):
    index = faiss.IndexFlatIP(DIM)
    index.add(database)
    _, indices = index.search(queries, K)
    return indices


def recall_at_k(gt, pred, k):
    hits = 0
    for i in range(len(gt)):
        hits += len(set(gt[i, :k].tolist()) & set(pred[i, :k].tolist()))
    return hits / (len(gt) * k)


def main():
    print(f"{'='*70}")
    print(f"Recall-vs-RF Curve Benchmark")
    print(f"N={N_VECTORS:,}, queries={N_QUERIES}, dim={DIM}, k={K}")
    print(f"{'='*70}")

    database, queries = generate_data()
    gt = ground_truth(queries, database)

    results = []

    print(f"\n{'rf':>6} {'Recall@10':>10} {'Avg(ms)':>8} {'p50(ms)':>8} {'p95(ms)':>8} {'QPS':>7} {'Candidates':>11} {'FloatOps':>10}")
    print("-" * 70)

    for rf in RF_VALUES:
        index = TwoStageIndex(dim=DIM, rerank_factor=rf)
        index.add(database)

        latencies = []
        all_indices = []

        for i in range(N_QUERIES):
            t0 = time.perf_counter()
            _, pred = index.search(queries[i], k=K)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed * 1000)  # ms
            all_indices.append(pred)

        all_indices = np.array(all_indices)
        r = recall_at_k(gt, all_indices, K)

        latencies = np.array(latencies)
        avg_ms = float(np.mean(latencies))
        p50_ms = float(np.percentile(latencies, 50))
        p95_ms = float(np.percentile(latencies, 95))
        qps = 1000.0 / avg_ms

        n_candidates = min(K * rf, N_VECTORS)
        float_ops = n_candidates * DIM  # one dot product per candidate

        print(f"{rf:>6} {r:>10.4f} {avg_ms:>8.2f} {p50_ms:>8.2f} {p95_ms:>8.2f} {qps:>7.1f} {n_candidates:>11,} {float_ops:>10,}")

        results.append({
            "rf": rf,
            "recall_at_10": round(r, 4),
            "avg_latency_ms": round(avg_ms, 2),
            "p50_latency_ms": round(p50_ms, 2),
            "p95_latency_ms": round(p95_ms, 2),
            "qps": round(qps, 1),
            "candidates_reranked": n_candidates,
            "float_ops": float_ops,
        })

    # Save JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "eval_rf_curve.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "eval_rf_curve.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Plot 1: Recall vs rf
    rfs = [r["rf"] for r in results]
    recalls = [r["recall_at_10"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rfs, recalls, "o-", color="#4CAF50", linewidth=2, markersize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Rerank Factor (rf)")
    ax.set_ylabel("Recall@10")
    ax.set_title(f"Recall@10 vs Rerank Factor\n{N_VECTORS:,} vectors, dim={DIM}, k={K}")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.5, label="90% target")
    ax.set_ylim(0, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "recall_vs_rf.png"), dpi=150)

    # Plot 2: Recall vs latency
    lats = [r["avg_latency_ms"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lats, recalls, "s-", color="#2196F3", linewidth=2, markersize=8)
    for i, rf in enumerate(rfs):
        ax.annotate(f"rf={rf}", (lats[i], recalls[i]), fontsize=8, textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Average Latency (ms)")
    ax.set_ylabel("Recall@10")
    ax.set_title("Recall@10 vs Latency — Tradeoff Curve")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "recall_vs_latency.png"), dpi=150)

    # Plot 3: QPS vs rf
    qps_vals = [r["qps"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rfs, qps_vals, "D-", color="#FF9800", linewidth=2, markersize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Rerank Factor (rf)")
    ax.set_ylabel("Queries Per Second")
    ax.set_title("Throughput vs Rerank Factor")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "qps_vs_rf.png"), dpi=150)

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: results/recall_vs_rf.png")
    print(f"Saved: results/recall_vs_latency.png")
    print(f"Saved: results/qps_vs_rf.png")


if __name__ == "__main__":
    main()
