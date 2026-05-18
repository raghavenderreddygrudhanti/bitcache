"""Comparison against multiple vector databases.

Compares bitcache against:
- FAISS (IndexFlatIP, IndexBinaryFlat, IndexHNSWFlat, IndexIVFFlat)
- Annoy (Spotify's approximate nearest neighbor)
- ChromaDB (in-memory mode)
- Qdrant (in-memory mode)

All on the same 100K clustered dataset, dim=768, k=10.
"""

import json
import os
import time
import tempfile
import shutil

import numpy as np
import faiss

from bitcache import BinaryIndex, TwoStageIndex
from bitcache.quantize import quantize

N_VECTORS = 100_000
N_QUERIES = 100  # reduced for slower DBs
K = 10
DIM = 768
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def generate_clustered_data(n, dim, n_clusters=100, seed=SEED):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    vectors = []
    for i in range(n):
        cluster = i % n_clusters
        noise = rng.standard_normal(dim).astype(np.float32) * 0.3
        vec = centers[cluster] + noise
        vectors.append(vec)
    vectors = np.array(vectors, dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def ground_truth(queries, database, k):
    index = faiss.IndexFlatIP(database.shape[1])
    index.add(database)
    _, indices = index.search(queries, k)
    return indices


def recall_at_k(gt, pred, k):
    n = len(gt)
    hits = 0
    for i in range(n):
        hits += len(set(gt[i, :k].tolist()) & set(pred[i][:k].tolist()))
    return hits / (n * k)


def bench_faiss_flat(database, queries, gt, k):
    dim = database.shape[1]
    index = faiss.IndexFlatIP(dim)
    t0 = time.time()
    index.add(database)
    build = time.time() - t0
    t0 = time.time()
    _, indices = index.search(queries, k)
    search = time.time() - t0
    return {"recall": 1.0, "qps": round(len(queries)/search), "build_s": round(build, 2), "memory": "full"}


def bench_faiss_binary(database, queries, gt, k):
    codes_db = quantize(database)
    codes_q = quantize(queries)
    n_bits = codes_db.shape[1] * 8
    index = faiss.IndexBinaryFlat(n_bits)
    index.add(codes_db)
    t0 = time.time()
    _, indices = index.search(codes_q, k)
    search = time.time() - t0
    r = recall_at_k(gt, indices, k)
    return {"recall": round(r, 4), "qps": round(len(queries)/search), "build_s": 0, "memory": "binary"}


def bench_faiss_ivf(database, queries, gt, k):
    dim = database.shape[1]
    nlist = 100
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    t0 = time.time()
    index.train(database)
    index.add(database)
    build = time.time() - t0
    index.nprobe = 10
    t0 = time.time()
    _, indices = index.search(queries, k)
    search = time.time() - t0
    r = recall_at_k(gt, indices, k)
    return {"recall": round(r, 4), "qps": round(len(queries)/search), "build_s": round(build, 2), "memory": "full"}


def bench_faiss_hnsw(database, queries, gt, k):
    dim = database.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    t0 = time.time()
    index.add(database)
    build = time.time() - t0
    index.hnsw.efSearch = 64
    t0 = time.time()
    _, indices = index.search(queries, k)
    search = time.time() - t0
    r = recall_at_k(gt, indices, k)
    return {"recall": round(r, 4), "qps": round(len(queries)/search), "build_s": round(build, 1), "memory": "full+graph"}


def bench_annoy(database, queries, gt, k):
    from annoy import AnnoyIndex
    dim = database.shape[1]
    index = AnnoyIndex(dim, 'dot')
    t0 = time.time()
    for i in range(len(database)):
        index.add_item(i, database[i])
    index.build(10)
    build = time.time() - t0
    t0 = time.time()
    all_indices = []
    for q in queries:
        ids = index.get_nns_by_vector(q.tolist(), k)
        all_indices.append(ids)
    search = time.time() - t0
    all_indices = np.array(all_indices)
    r = recall_at_k(gt, all_indices, k)
    return {"recall": round(r, 4), "qps": round(len(queries)/search), "build_s": round(build, 1), "memory": "mmap"}


def bench_chromadb(database, queries, gt, k):
    import chromadb
    client = chromadb.Client()
    collection = client.create_collection("bench", metadata={"hnsw:space": "ip"})
    t0 = time.time()
    # ChromaDB needs string IDs and batches of max 41666
    batch_size = 40000
    for start in range(0, len(database), batch_size):
        end = min(start + batch_size, len(database))
        ids = [str(i) for i in range(start, end)]
        embeddings = database[start:end].tolist()
        collection.add(ids=ids, embeddings=embeddings)
    build = time.time() - t0
    t0 = time.time()
    all_indices = []
    for q in queries:
        result = collection.query(query_embeddings=[q.tolist()], n_results=k)
        ids = [int(x) for x in result["ids"][0]]
        all_indices.append(ids)
    search = time.time() - t0
    all_indices = np.array(all_indices)
    r = recall_at_k(gt, all_indices, k)
    client.delete_collection("bench")
    return {"recall": round(r, 4), "qps": round(len(queries)/search), "build_s": round(build, 1), "memory": "full+graph"}


def bench_bitcache_flat(database, queries, gt, k):
    dim = database.shape[1]
    index = BinaryIndex(dim=dim)
    t0 = time.time()
    index.add(database)
    build = time.time() - t0
    t0 = time.time()
    _, indices = index.search_batch(queries, k=k)
    search = time.time() - t0
    r = recall_at_k(gt, indices, k)
    return {"recall": round(r, 4), "qps": round(len(queries)/search), "build_s": round(build, 2), "memory": "binary"}


def bench_bitcache_twostage(database, queries, gt, k, rf=100):
    dim = database.shape[1]
    index = TwoStageIndex(dim=dim, rerank_factor=rf)
    t0 = time.time()
    index.add(database)
    build = time.time() - t0
    t0 = time.time()
    _, indices = index.search_batch(queries, k=k)
    search = time.time() - t0
    r = recall_at_k(gt, indices, k)
    return {"recall": round(r, 4), "qps": round(len(queries)/search), "build_s": round(build, 2), "memory": "binary+float"}


def main():
    print("=" * 70)
    print("Vector Database Comparison")
    print(f"N={N_VECTORS:,}, queries={N_QUERIES}, dim={DIM}, k={K}")
    print("=" * 70)

    print("\nGenerating clustered data...")
    all_vecs = generate_clustered_data(N_VECTORS + N_QUERIES, DIM)
    database = all_vecs[:N_VECTORS]
    queries = all_vecs[N_VECTORS:N_VECTORS + N_QUERIES]

    print("Computing ground truth...")
    gt = ground_truth(queries, database, K)

    results = {}
    benchmarks = [
        ("FAISS Flat (exact)", bench_faiss_flat),
        ("FAISS Binary Flat", bench_faiss_binary),
        ("FAISS IVF (nprobe=10)", bench_faiss_ivf),
        ("FAISS HNSW (ef=64)", bench_faiss_hnsw),
        ("Annoy (n_trees=10)", bench_annoy),
        ("ChromaDB (in-memory)", bench_chromadb),
        ("bitcache flat", bench_bitcache_flat),
        ("bitcache two-stage rf=100", bench_bitcache_twostage),
    ]

    print(f"\n{'Method':<30} {'Recall@10':>10} {'QPS':>8} {'Build':>8} {'Memory':>12}")
    print("-" * 70)

    for name, fn in benchmarks:
        try:
            r = fn(database, queries, gt, K)
            results[name] = r
            print(f"{name:<30} {r['recall']:>10.4f} {r['qps']:>8} {r['build_s']:>7}s {r['memory']:>12}")
        except Exception as e:
            print(f"{name:<30} {'ERROR':>10} — {str(e)[:40]}")
            results[name] = {"error": str(e)}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "eval_vectordbs.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
