"""Benchmark on realistic embeddings + memory-vs-recall Pareto chart.

Uses sklearn make_classification data (60 natural clusters, 768 dims)
which mimics real text embedding distributions.
"""

import csv
import json
import os
import time

import faiss
import matplotlib.pyplot as plt
import numpy as np

from bitcache import TwoStageIndex
from bitcache.quantize import quantize

N_VECTORS = 50_000
N_QUERIES = 100
DIM = 768
K = 10
RF_VALUES = [10, 25, 50, 100, 200, 500, 1000]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_realistic_data():
    data = np.load("benchmarks/data/realistic_768.npy")
    database = data[:N_VECTORS - N_QUERIES]
    queries = data[N_VECTORS - N_QUERIES:N_VECTORS]
    return database, queries


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
    print("=" * 70)
    print("REALISTIC EMBEDDINGS BENCHMARK")
    print(f"N={N_VECTORS-N_QUERIES:,} database, {N_QUERIES} queries, dim={DIM}, k={K}")
    print("Data: sklearn make_classification (60 clusters, 200 informative dims)")
    print("=" * 70)

    database, queries = load_realistic_data()
    gt = ground_truth(queries, database)

    # === PART 1: Recall-vs-RF on realistic data ===
    print(f"\n{'rf':>6} {'Recall@10':>10} {'Avg(ms)':>8} {'p50(ms)':>8} {'p95(ms)':>8} {'QPS':>7}")
    print("-" * 55)

    rf_results = []
    for rf in RF_VALUES:
        index = TwoStageIndex(dim=DIM, rerank_factor=rf)
        index.add(database)

        latencies = []
        all_preds = []
        for i in range(N_QUERIES):
            t0 = time.perf_counter()
            _, pred = index.search(queries[i], k=K)
            latencies.append((time.perf_counter() - t0) * 1000)
            all_preds.append(pred)

        all_preds = np.array(all_preds)
        latencies = np.array(latencies)
        r = recall_at_k(gt, all_preds, K)
        avg = float(np.mean(latencies))
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        qps = 1000.0 / avg

        print(f"{rf:>6} {r:>10.4f} {avg:>8.2f} {p50:>8.2f} {p95:>8.2f} {qps:>7.1f}")
        rf_results.append({"rf": rf, "recall": round(r, 4), "avg_ms": round(avg, 2),
                          "p50_ms": round(p50, 2), "p95_ms": round(p95, 2), "qps": round(qps, 1)})

    # === PART 2: Baselines for Pareto chart ===
    print("\n\nBASELINES:")
    baselines = {}

    # FAISS Flat
    t0 = time.time()
    idx = faiss.IndexFlatIP(DIM); idx.add(database)
    _, pred = idx.search(queries, K)
    t = time.time() - t0
    baselines["FAISS Flat"] = {"recall": 1.0, "memory_mb": round(database.nbytes/1024**2, 1), "qps": round(N_QUERIES/t)}
    print(f"  FAISS Flat: recall=1.0, memory={baselines['FAISS Flat']['memory_mb']}MB")

    # FAISS HNSW
    idx = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
    idx.add(database); idx.hnsw.efSearch = 64
    t0 = time.time(); _, pred = idx.search(queries, K); t = time.time() - t0
    r = recall_at_k(gt, pred, K)
    mem = database.nbytes/1024**2 + 50  # graph overhead estimate
    baselines["FAISS HNSW"] = {"recall": round(r, 4), "memory_mb": round(mem, 1), "qps": round(N_QUERIES/t)}
    print(f"  FAISS HNSW: recall={r:.4f}, memory≈{mem:.0f}MB")

    # FAISS Binary
    codes_db = quantize(database); codes_q = quantize(queries)
    idx = faiss.IndexBinaryFlat(codes_db.shape[1]*8); idx.add(codes_db)
    t0 = time.time(); _, pred = idx.search(codes_q, K); t = time.time() - t0
    r = recall_at_k(gt, pred, K)
    baselines["FAISS Binary"] = {"recall": round(r, 4), "memory_mb": round(codes_db.nbytes/1024**2, 1), "qps": round(N_QUERIES/t)}
    print(f"  FAISS Binary: recall={r:.4f}, memory={baselines['FAISS Binary']['memory_mb']}MB")

    # FAISS IVF
    quantizer = faiss.IndexFlatIP(DIM)
    idx = faiss.IndexIVFFlat(quantizer, DIM, 100, faiss.METRIC_INNER_PRODUCT)
    idx.train(database); idx.add(database); idx.nprobe = 10
    t0 = time.time(); _, pred = idx.search(queries, K); t = time.time() - t0
    r = recall_at_k(gt, pred, K)
    baselines["FAISS IVF"] = {"recall": round(r, 4), "memory_mb": round(database.nbytes/1024**2, 1), "qps": round(N_QUERIES/t)}
    print(f"  FAISS IVF: recall={r:.4f}, memory={baselines['FAISS IVF']['memory_mb']}MB")

    # bitcache points for Pareto
    binary_mb = codes_db.nbytes / 1024**2
    float_mb = database.nbytes / 1024**2

    # === PART 3: Generate charts ===
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Chart: Recall vs RF (realistic data)
    fig, ax = plt.subplots(figsize=(8, 5))
    rfs = [r["rf"] for r in rf_results]
    recalls = [r["recall"] for r in rf_results]
    ax.plot(rfs, recalls, "o-", color="#4CAF50", linewidth=2.5, markersize=9, label="bitcache (realistic data)")
    ax.set_xscale("log")
    ax.set_xlabel("Rerank Factor (rf)", fontsize=11)
    ax.set_ylabel("Recall@10", fontsize=11)
    ax.set_title("Recall@10 vs Rerank Factor — Realistic Embeddings\n50K vectors, dim=768, 60 natural clusters", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.5, label="90% target")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "recall_vs_rf_realistic.png"), dpi=150)

    # Chart: Memory vs Recall Pareto
    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot baselines
    for name, data in baselines.items():
        ax.scatter(data["memory_mb"], data["recall"], s=100, zorder=5, label=name)
        ax.annotate(name, (data["memory_mb"], data["recall"]), fontsize=8,
                   textcoords="offset points", xytext=(5, 5))

    # Plot bitcache at different rf values
    for r in rf_results:
        # bitcache uses binary (4.6MB) + float for rerank candidates (variable)
        # But total stored = binary + float
        total_mem = binary_mb + float_mb  # stores both
        ax.scatter(total_mem, r["recall"], s=80, marker="s", color="#4CAF50", zorder=5)

    # Add bitcache binary-only point
    ax.scatter(binary_mb, rf_results[-1]["recall"], s=120, marker="D", color="#4CAF50",
              zorder=6, label=f"bitcache (rf=1000, recall={rf_results[-1]['recall']:.3f})")
    ax.annotate(f"bitcache\n(binary={binary_mb:.1f}MB)", (binary_mb, rf_results[-1]["recall"]),
               fontsize=8, textcoords="offset points", xytext=(-60, -20))

    ax.set_xlabel("Memory (MB)", fontsize=11)
    ax.set_ylabel("Recall@10", fontsize=11)
    ax.set_title("Memory vs Recall — Pareto Frontier", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "memory_vs_recall_pareto.png"), dpi=150)

    # Save JSON
    all_results = {
        "dataset": "sklearn_make_classification_60clusters_768d",
        "n_vectors": N_VECTORS - N_QUERIES,
        "rf_curve": rf_results,
        "baselines": baselines,
    }
    with open(os.path.join(RESULTS_DIR, "eval_realistic.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved: results/recall_vs_rf_realistic.png")
    print(f"Saved: results/memory_vs_recall_pareto.png")
    print(f"Saved: results/eval_realistic.json")


if __name__ == "__main__":
    main()
