"""Comprehensive vector database comparison — 14 methods.

Compares bitcache against:
- FAISS: Flat, Binary, IVF, HNSW, PQ
- hnswlib
- Annoy
- USearch
- PyNNDescent
- Voyager
- scikit-learn NearestNeighbors (BallTree, KDTree)
- nmslib

All on 100K clustered vectors, dim=768, k=10.
"""

import json
import os
import time

import numpy as np
import faiss

from bitcache import BinaryIndex, TwoStageIndex
from bitcache.quantize import quantize

N_VECTORS = 100_000
N_QUERIES = 100
K = 10
DIM = 768
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def generate_data():
    rng = np.random.default_rng(SEED)
    n_clusters = 100
    n_total = N_VECTORS + N_QUERIES
    centers = rng.standard_normal((n_clusters, DIM)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    vectors = []
    for i in range(n_total):
        noise = rng.standard_normal(DIM).astype(np.float32) * 0.3
        vec = centers[i % n_clusters] + noise
        vectors.append(vec)
    vectors = np.array(vectors, dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors[:N_VECTORS], vectors[N_VECTORS:]


def ground_truth(queries, database):
    index = faiss.IndexFlatIP(DIM)
    index.add(database)
    _, indices = index.search(queries, K)
    return indices


def recall(gt, pred):
    n = len(gt)
    hits = 0
    for i in range(n):
        hits += len(set(gt[i, :K].tolist()) & set(list(pred[i])[:K]))
    return round(hits / (n * K), 4)


def run_bench(name, fn, database, queries, gt):
    try:
        t0 = time.time()
        r = fn(database, queries, gt)
        total = time.time() - t0
        print(f"  {name:<35} recall={r['recall']:.4f}  qps={r['qps']:>6}  build={r['build']:.1f}s")
        return r
    except Exception as e:
        err = str(e)[:60]
        print(f"  {name:<35} ERROR: {err}")
        return {"recall": 0, "qps": 0, "build": 0, "error": err}


# --- Benchmark functions ---

def bench_faiss_flat(db, q, gt):
    idx = faiss.IndexFlatIP(DIM)
    t0 = time.time(); idx.add(db); build = time.time() - t0
    t0 = time.time(); _, pred = idx.search(q, K); search = time.time() - t0
    return {"recall": 1.0, "qps": round(len(q)/search), "build": build}


def bench_faiss_binary(db, q, gt):
    codes_db = quantize(db); codes_q = quantize(q)
    idx = faiss.IndexBinaryFlat(codes_db.shape[1]*8)
    idx.add(codes_db)
    t0 = time.time(); _, pred = idx.search(codes_q, K); search = time.time() - t0
    return {"recall": recall(gt, pred), "qps": round(len(q)/search), "build": 0}


def bench_faiss_ivf(db, q, gt):
    quantizer = faiss.IndexFlatIP(DIM)
    idx = faiss.IndexIVFFlat(quantizer, DIM, 100, faiss.METRIC_INNER_PRODUCT)
    t0 = time.time(); idx.train(db); idx.add(db); build = time.time() - t0
    idx.nprobe = 10
    t0 = time.time(); _, pred = idx.search(q, K); search = time.time() - t0
    return {"recall": recall(gt, pred), "qps": round(len(q)/search), "build": build}


def bench_faiss_hnsw(db, q, gt):
    idx = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
    t0 = time.time(); idx.add(db); build = time.time() - t0
    idx.hnsw.efSearch = 64
    t0 = time.time(); _, pred = idx.search(q, K); search = time.time() - t0
    return {"recall": recall(gt, pred), "qps": round(len(q)/search), "build": build}


def bench_faiss_pq(db, q, gt):
    m = 48  # subquantizers
    idx = faiss.IndexPQ(DIM, m, 8, faiss.METRIC_INNER_PRODUCT)
    t0 = time.time(); idx.train(db); idx.add(db); build = time.time() - t0
    t0 = time.time(); _, pred = idx.search(q, K); search = time.time() - t0
    return {"recall": recall(gt, pred), "qps": round(len(q)/search), "build": build}


def bench_hnswlib(db, q, gt):
    import hnswlib
    idx = hnswlib.Index(space='ip', dim=DIM)
    idx.init_index(max_elements=N_VECTORS, ef_construction=200, M=32)
    t0 = time.time(); idx.add_items(db); build = time.time() - t0
    idx.set_ef(64)
    t0 = time.time(); pred, _ = idx.knn_query(q, k=K); search = time.time() - t0
    return {"recall": recall(gt, pred), "qps": round(len(q)/search), "build": build}


def bench_annoy(db, q, gt):
    from annoy import AnnoyIndex
    idx = AnnoyIndex(DIM, 'dot')
    t0 = time.time()
    for i in range(len(db)): idx.add_item(i, db[i])
    idx.build(10)
    build = time.time() - t0
    t0 = time.time()
    pred = [idx.get_nns_by_vector(q[i].tolist(), K) for i in range(len(q))]
    search = time.time() - t0
    return {"recall": recall(gt, np.array(pred)), "qps": round(len(q)/search), "build": build}


def bench_usearch(db, q, gt):
    from usearch.index import Index
    idx = Index(ndim=DIM, metric='ip', dtype='f32')
    t0 = time.time(); idx.add(np.arange(len(db)), db); build = time.time() - t0
    t0 = time.time()
    results = idx.search(q, K)
    search = time.time() - t0
    pred = results.keys
    return {"recall": recall(gt, pred), "qps": round(len(q)/search), "build": build}


def bench_pynndescent(db, q, gt):
    from pynndescent import NNDescent
    t0 = time.time()
    idx = NNDescent(db, metric='dot', n_neighbors=30)
    idx.prepare()
    build = time.time() - t0
    t0 = time.time()
    pred, _ = idx.query(q, k=K)
    search = time.time() - t0
    return {"recall": recall(gt, pred), "qps": round(len(q)/search), "build": build}


def bench_voyager(db, q, gt):
    from voyager import Index, Space
    idx = Index(Space.InnerProduct, num_dimensions=DIM)
    t0 = time.time()
    idx.add_items(db)
    build = time.time() - t0
    t0 = time.time()
    pred, _ = idx.query(q, k=K)
    search = time.time() - t0
    return {"recall": recall(gt, pred), "qps": round(len(q)/search), "build": build}


def bench_sklearn_ball(db, q, gt):
    from sklearn.neighbors import NearestNeighbors
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=K, algorithm='ball_tree', metric='euclidean')
    nn.fit(db)
    build = time.time() - t0
    t0 = time.time()
    _, pred = nn.kneighbors(q)
    search = time.time() - t0
    return {"recall": recall(gt, pred), "qps": round(len(q)/search), "build": build}


def bench_nmslib(db, q, gt):
    import nmslib
    idx = nmslib.init(method='hnsw', space='negdotprod')
    t0 = time.time()
    idx.addDataPointBatch(db)
    idx.createIndex({'M': 32, 'efConstruction': 200}, print_progress=False)
    build = time.time() - t0
    idx.setQueryTimeParams({'efSearch': 64})
    t0 = time.time()
    results = idx.knnQueryBatch(q, k=K)
    search = time.time() - t0
    pred = np.array([r[0] for r in results])
    return {"recall": recall(gt, pred), "qps": round(len(q)/search), "build": build}


def bench_bitcache_flat(db, q, gt):
    idx = BinaryIndex(dim=DIM)
    t0 = time.time(); idx.add(db); build = time.time() - t0
    t0 = time.time(); _, pred = idx.search_batch(q, k=K); search = time.time() - t0
    return {"recall": recall(gt, pred), "qps": round(len(q)/search), "build": build}


def bench_bitcache_twostage(db, q, gt):
    idx = TwoStageIndex(dim=DIM, rerank_factor=100)
    t0 = time.time(); idx.add(db); build = time.time() - t0
    t0 = time.time(); _, pred = idx.search_batch(q, k=K); search = time.time() - t0
    return {"recall": recall(gt, pred), "qps": round(len(q)/search), "build": build}


def main():
    print("=" * 70)
    print(f"Vector DB Comparison — {N_VECTORS:,} vectors, dim={DIM}, k={K}")
    print("=" * 70)

    database, queries = generate_data()
    gt = ground_truth(queries, database)

    benchmarks = [
        ("FAISS Flat (exact)", bench_faiss_flat),
        ("FAISS Binary Flat", bench_faiss_binary),
        ("FAISS IVF (nprobe=10)", bench_faiss_ivf),
        ("FAISS HNSW (M=32, ef=64)", bench_faiss_hnsw),
        ("FAISS PQ (m=48, nbits=8)", bench_faiss_pq),
        ("hnswlib (M=32, ef=64)", bench_hnswlib),
        ("Annoy (n_trees=10)", bench_annoy),
        ("USearch (HNSW)", bench_usearch),
        ("PyNNDescent", bench_pynndescent),
        ("Voyager (HNSW)", bench_voyager),
        ("sklearn BallTree", bench_sklearn_ball),
        ("nmslib HNSW", bench_nmslib),
        ("bitcache flat (binary)", bench_bitcache_flat),
        ("bitcache two-stage rf=100", bench_bitcache_twostage),
    ]

    print(f"\n{'Method':<35} {'Recall':>7} {'QPS':>7} {'Build':>7}")
    print("-" * 60)

    results = {}
    for name, fn in benchmarks:
        r = run_bench(name, fn, database, queries, gt)
        results[name] = r

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "eval_all_dbs.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
