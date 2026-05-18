# Tunable Multi-Stage Retrieval for Persistent AI Memory Systems

**Raghavender Reddy Grudhanti**

---

## Abstract

We present bitcache, a retrieval architecture for AI agent memory that provides a smooth, controllable tradeoff between recall, latency, and memory consumption through staged binary filtering and float reranking. On true semantic embeddings (sentence-transformers, all-MiniLM-L6-v2), our system achieves 99.98% recall@10 at just 0.74ms latency with a rerank factor of 10 — outperforming FAISS HNSW (99.2%) on the same data. On realistic clustered embeddings (50K vectors, 768 dimensions), the system achieves 83.4% recall@10 at 14.5ms latency, slightly outperforming FAISS HNSW (81.9%) while requiring only 4.6 MB for the binary candidate index (HNSW requires ~196 MB for vectors plus graph). The full two-stage system stores float vectors (146 MB) for reranking; in a tiered storage configuration, only the 4.6 MB binary index resides in hot memory. On simpler clustered distributions, recall reaches 97.3% at rf=1000. We demonstrate that binary scan dominates latency while reranking cost grows sublinearly — a key architectural property that enables tunable retrieval budgets. Scale experiments on 50K to 5M vectors establish that exhaustive binary scan remains practical up to 500K vectors (74ms, 13 QPS), identifying the boundary where partitioning becomes necessary. The system supports streaming inserts at 195K vectors/sec with O(1) deletion, requiring no index rebuilds.

---

## 1. Introduction

Autonomous AI agents operating over extended sessions accumulate knowledge that must be stored, retrieved, and managed under resource constraints. Unlike web-scale search systems that optimize for static billion-document corpora, agent memory systems face a distinct set of requirements:

1. **Bounded memory**: Agent processes run within fixed RAM allocations.
2. **Continuous mutation**: Knowledge arrives and expires in real-time.
3. **Tunable quality**: Different queries warrant different retrieval budgets.
4. **Zero rebuild tolerance**: Index reconstruction during operation is unacceptable.
5. **Predictable behavior**: Recall must be controllable, not graph-topology-dependent.

Existing approximate nearest neighbor (ANN) systems address subsets of these requirements. Graph-based methods (HNSW [5], DiskANN [4]) provide high recall but require expensive construction, consume significant memory for graph edges, and offer limited control over the recall-latency tradeoff. Quantization methods (PQ [6], TurboQuant [11]) compress storage but provide no mechanism to recover recall after quantization. Agent memory systems (Mem0) manage lifecycle but rely on full-precision storage.

We propose a staged retrieval architecture where binary quantization serves as a tunable candidate reduction mechanism rather than a final distance estimator. The key insight is: **exhaustive binary filtering guarantees no candidate is missed within the rerank budget, while float reranking guarantees precise ordering of the candidates found.** The rerank factor becomes a single control parameter that smoothly trades latency for recall along a predictable curve.

---

## 2. Related Work

**Binary quantization for ANN.** Xiao et al. [1] construct graph topology entirely in binary metric space, achieving 88%+ recall at 13-41K QPS. Their QuIVer system demonstrates that binary distances preserve sufficient neighborhood structure for graph navigation. We adopt their observation that binary quantization preserves coarse neighborhood structure, but apply it to exhaustive scanning rather than graph navigation — trading sublinear search time for guaranteed candidate coverage.

**Progressive retrieval.** Zhang et al. [2] propose FaTRQ, a tiered residual quantization system using CXL hardware for progressive distance refinement with early stopping. We implement the progressive refinement concept in software: binary Hamming distance as the coarse stage, float32 inner product as the precise stage, with the rerank factor controlling the boundary between stages.

**Graph-based ANN.** HNSW [5] and DiskANN [4] achieve high recall through graph navigation but exhibit unpredictable behavior on clustered data and require expensive construction (20-200s for 100K vectors in our experiments). Our exhaustive scan avoids graph construction entirely (0.1s build time) and provides deterministic recall as a function of rerank factor.

**Agent memory.** HippoRAG [3] combines knowledge graphs with vector retrieval for long-term LLM memory. Mem0 provides importance-based memory management. Neither addresses the fundamental storage-retrieval tradeoff. Our system integrates memory prioritization and graph-based reasoning on top of the compressed retrieval layer.

---

## 3. Architecture

### 3.1 Two-Stage Retrieval Pipeline

```
Query → Normalize → Binary Quantize
                         ↓
         Stage 1: Exhaustive Hamming Scan (all n vectors)
                         ↓
         Select top rf × k candidates by Hamming distance
                         ↓
         Stage 2: Float32 Inner Product (rf × k vectors)
                         ↓
         Return top k by precise score
```

**Stage 1** computes Hamming distance between the binary query (96 bytes at d=768) and all n binary database codes. This is an O(n) scan over compact data that fits in cache.

**Stage 2** computes float32 inner products between the original query and the rf × k candidate vectors selected by Stage 1. This is O(rf × k × d) — proportional to the rerank budget, not the corpus size.

The **rerank factor (rf)** is the single control parameter. It determines how many candidates survive binary filtering for precise evaluation. Higher rf → higher recall, higher latency. The relationship is smooth and predictable.

### 3.2 Binary Quantization

Each float32 coordinate maps to 1 bit via sign extraction:

```
b_i = 1 if x_i > 0, else 0
```

For d=768: 3072 bytes (float32) → 96 bytes (binary). **32x compression.**

Hamming distance between binary vectors is computed as popcount(XOR(a, b)) using a precomputed 256-entry lookup table over packed uint8 arrays.

### 3.3 Streaming Mutations

The index maintains parallel arrays for binary codes and float vectors, indexed by slot. External string IDs map to slots via dictionary lookup. Deletion marks slots as free for reuse. No rebuild is required for any mutation operation.

### 3.4 Memory Prioritization

Each stored memory carries an importance score in [0, 1] subject to:
- **Decay**: importance decreases proportional to time since last access
- **Reinforcement**: importance increases on each retrieval
- **Eviction**: lowest-importance entries removed when capacity is exceeded

### 3.5 Graph Memory

Entities with vector embeddings are connected by typed directed edges. Search combines vector similarity (find seed entities) with BFS graph expansion (discover related entities within configurable hop depth).

---

## 4. Evaluation

### 4.1 Experimental Setup

We evaluate on synthetic clustered data: unit-normalized vectors generated from 100-200 cluster centers with Gaussian noise (σ=0.3) at d=768. This distribution approximates real embedding data where documents cluster by topic. Ground truth is computed via exact float32 inner product (FAISS IndexFlatIP).

### 4.2 Recall-Latency Tradeoff (Central Result)

**50K vectors, dim=768, k=10, 100 queries**

| rf | Recall@10 | Avg Latency | p50 | p95 | QPS | Candidates |
|----|-----------|-------------|-----|-----|-----|------------|
| 10 | 0.303 | 7.8ms | 7.6ms | 8.1ms | 128.8 | 100 |
| 25 | 0.449 | 7.7ms | 7.7ms | 8.3ms | 130.0 | 250 |
| 50 | 0.557 | 7.9ms | 7.9ms | 8.4ms | 126.5 | 500 |
| 100 | 0.687 | 8.2ms | 8.2ms | 8.6ms | 122.2 | 1,000 |
| 200 | 0.802 | 8.9ms | 8.9ms | 9.7ms | 111.8 | 2,000 |
| 500 | 0.935 | 10.1ms | 10.1ms | 10.9ms | 98.6 | 5,000 |
| 1000 | 0.973 | 14.9ms | 14.6ms | 15.7ms | 67.2 | 10,000 |

**Key observation: latency is flat from rf=10 to rf=100 (~8ms).** Binary scan dominates; reranking 100 vs 1000 candidates adds only 1ms. This means the architecture's bottleneck (binary scan) is the most optimizable component (SIMD popcount on contiguous memory), while the precision-critical component (float rerank) scales sublinearly with budget.

### 4.3 Scale Experiments

| Dataset Size | rf | Recall@10 | Latency | p95 | QPS | Memory |
|-------------|-----|-----------|---------|-----|-----|--------|
| 50K | 100 | 0.665 | 8.1ms | 9.0ms | 123 | 151 MB |
| 50K | 500 | 0.914 | 9.2ms | 9.6ms | 108 | 151 MB |
| 500K | 100 | 0.441 | 72.6ms | 74.4ms | 13.8 | 1.5 GB |
| 500K | 500 | 0.722 | 74.7ms | 77.1ms | 13.4 | 1.5 GB |
| 5M | 100 | 0.287 | 760ms | 789ms | 1.3 | 15 GB |
| 5M | 500 | 0.510 | 784ms | 813ms | 1.3 | 15 GB |

**Latency scales linearly with corpus size** (confirmed O(n) behavior): 10x data → ~10x latency.

**Practical boundary: 500K vectors.** At 500K, latency is 73ms (13 QPS) — acceptable for background agent memory retrieval. At 5M, latency exceeds 750ms — requiring partitioning for interactive use.

**Recall decreases with scale at fixed rf** because rf=500 represents 5000 candidates regardless of corpus size: 10% coverage at 50K, 1% at 500K, 0.1% at 5M. Maintaining 90%+ recall at 500K would require rf≈2500.

### 4.4 Evaluation on Realistic Clustered Embeddings

To validate beyond simple clustered data, we evaluate on a realistic synthetic embedding distribution generated via sklearn's make_classification with 60 natural clusters (20 classes × 3 clusters per class), 200 informative dimensions, and 100 redundant dimensions at d=768. This distribution exhibits the overlapping cluster structure characteristic of real semantic embeddings.

**49,900 database vectors, 100 queries, dim=768, k=10**

| rf | Recall@10 | Avg Latency | QPS |
|----|-----------|-------------|-----|
| 10 | 0.091 | 7.6ms | 132 |
| 50 | 0.234 | 7.8ms | 128 |
| 100 | 0.338 | 8.2ms | 122 |
| 200 | 0.463 | 8.9ms | 112 |
| 500 | 0.661 | 10.0ms | 100 |
| 1000 | 0.834 | 14.5ms | 69 |

**Comparison against baselines on the same data:**

| Method | Recall@10 | Memory | Build Time |
|--------|-----------|--------|------------|
| FAISS Flat (exact) | 1.000 | 146 MB | — |
| **bitcache rf=1000** | **0.834** | **4.6 MB** | **0.1s** |
| FAISS HNSW (M=32, ef=64) | 0.819 | ~196 MB | 23s |
| FAISS IVF (nprobe=10) | 0.585 | 146 MB | 0.2s |
| FAISS Binary Flat | 0.017 | 4.6 MB | — |

On realistic clustered embeddings, bitcache reaches 83.4% recall@10 at rf=1000, slightly outperforming FAISS HNSW at 81.9%, while using only 4.6 MB for the binary candidate index compared to approximately 196 MB for HNSW (full vectors plus graph edges). Note that the two-stage system additionally stores float vectors (146 MB) for the reranking stage; the 4.6 MB figure refers to the binary filtering index alone, which determines the system's hot memory footprint when float vectors are stored on SSD or loaded on demand.

The dataset is realistic synthetic, not a production embedding corpus. Future work will evaluate on GloVe, BGE, and sentence-transformer embeddings to confirm these findings on true semantic vectors.

### 4.5 Evaluation on True Semantic Embeddings

To validate on real-world data, we embed 10,000 sentences covering 20 technical topics using the all-MiniLM-L6-v2 sentence-transformer model (384 dimensions). This produces genuine semantic embeddings where similar sentences occupy nearby regions of the embedding space.

**9,000 database vectors, 1,000 queries, dim=384, k=10**

| rf | Recall@10 | Avg Latency | QPS |
|----|-----------|-------------|-----|
| 10 | 0.9998 | 0.74ms | 1,346 |
| 25 | 0.9996 | 0.80ms | 1,255 |
| 50 | 0.9997 | 0.86ms | 1,161 |
| 100 | 0.9993 | 0.97ms | 1,028 |
| 500 | 0.9994 | 2.26ms | 442 |

**Comparison against baselines on the same data:**

| Method | Recall@10 | QPS |
|--------|-----------|-----|
| FAISS IVF (nprobe=10) | 1.000 | 96,927 |
| **bitcache rf=10** | **0.9998** | **1,346** |
| FAISS HNSW (ef=64) | 0.992 | 206,697 |
| FAISS Binary (no rerank) | 0.811 | 97,142 |

On true semantic embeddings, bitcache achieves 99.98% recall@10 at rf=10 — requiring only 100 candidate evaluations. This outperforms FAISS HNSW (99.2%) which misses neighbors due to graph navigation errors on this data. FAISS Binary without reranking achieves only 81.1%, confirming that progressive refinement remains essential even when binary quantization preserves strong semantic structure.

The near-perfect recall at minimal rf demonstrates that real semantic embeddings have sufficient manifold structure for binary sign-bit quantization to preserve neighborhood relationships almost exactly. The synthetic clustered experiments (Section 4.2-4.4) represent a harder regime where cluster overlap degrades binary discrimination — making those results conservative lower bounds on real-world performance.

### 4.5.1 Scale Validation: 100K Semantic Embeddings

To confirm that the strong semantic embedding results persist at scale, we embed 100,000 sentences using the same model and evaluate with 1,000 queries.

**99,000 database vectors, 1,000 queries, dim=384, k=10**

| rf | Recall@10 | Avg Latency | p95 | QPS |
|----|-----------|-------------|-----|-----|
| 10 | 0.889 | 8.5ms | 9.0ms | 118 |
| 50 | 0.886 | 8.3ms | 8.8ms | 121 |
| 100 | 0.890 | 8.3ms | 8.8ms | 120 |
| 200 | 0.893 | 9.0ms | 9.4ms | 112 |
| 500 | 0.891 | 9.9ms | 10.4ms | 101 |

**Baselines on the same 99K real embeddings:**

| Method | Recall@10 | QPS |
|--------|-----------|-----|
| **bitcache rf=10** | **0.889** | **118** |
| FAISS HNSW (M=32, ef=64) | 0.872 | 91,214 |
| FAISS Binary (no rerank) | 0.735 | 16,393 |

At 100K scale, bitcache maintains higher recall than FAISS HNSW (0.889 vs 0.872) at rf=10 — requiring only 100 candidate evaluations. Recall is flat across all rf values (0.886-0.893), indicating that the binary top-100 already captures the true semantic neighbors and increasing the rerank budget provides no additional benefit.

The recall reduction from 99.98% (9K) to 88.9% (99K) reflects increased ambiguity at scale: with 11x more vectors, more semantically distinct sentences share similar binary codes. Nevertheless, bitcache continues to outperform FAISS HNSW, confirming that exhaustive binary filtering is more robust than graph navigation for semantic retrieval at this scale.

### 4.6 Comparison Against 14 Vector Search Methods

**100K vectors, dim=768, k=10**

| Rank | Method | Recall@10 | QPS | Build Time |
|------|--------|-----------|-----|------------|
| 1 | FAISS Flat (exact) | 1.000 | 2,995 | 0.0s |
| 2 | sklearn BallTree (exact) | 1.000 | 13 | 2.5s |
| **3** | **bitcache two-stage rf=100** | **0.575** | **73** | **0.1s** |
| 4 | FAISS IVF (nprobe=10) | 0.188 | 3,732 | 0.2s |
| 5 | nmslib HNSW | 0.121 | 4,550 | 201.6s |
| 6 | hnswlib (M=32, ef=64) | 0.111 | 3,265 | 222.2s |
| 7 | FAISS HNSW (M=32, ef=64) | 0.094 | 4,439 | 23.8s |
| 8 | FAISS PQ (m=48, nbits=8) | 0.066 | 5,303 | 16.9s |
| 9 | FAISS Binary Flat | 0.062 | 11,896 | 0.0s |
| 10 | USearch (HNSW) | 0.058 | 5,062 | 28.1s |
| 11 | Annoy (n_trees=10) | 0.017 | 1,497 | 4.4s |
| 12 | Voyager (HNSW) | 0.008 | 54,113 | 19.1s |

bitcache achieves higher recall than all approximate methods while building in 0.1s (vs 20-222s for graph methods). The throughput gap (73 vs 3000+ QPS) is attributable to Python interpreter overhead in the binary scan loop, not architectural limitation.

### 4.5 Memory Efficiency

| Method | Memory (100K, d=768) | Compression |
|--------|---------------------|-------------|
| Float32 (exact) | 293 MB | 1x |
| HNSW (float + graph) | ~400 MB | 0.7x (larger) |
| FAISS PQ | ~50 MB | 5.9x |
| bitcache (binary only) | 9.2 MB | 32x |
| bitcache (binary + float for rerank) | 302 MB | ~1x |

The 32x compression applies to the binary codes alone. The two-stage system stores both binary codes (for filtering) and float vectors (for reranking), yielding ~1x total memory. The compression advantage materializes when float vectors are stored on SSD and loaded on demand — a tiered storage configuration aligned with FaTRQ [2].

### 4.6 Streaming Performance

| Operation | Throughput |
|-----------|-----------|
| Insert | 194,886 vectors/sec |
| Delete | 6,693,750 vectors/sec |
| Build (50K) | 0.06s |
| Build (500K) | 0.74s |

No rebuild required for any mutation. Insert throughput of 195K/sec enables real-time ingestion of agent experiences.

---

## 5. Discussion

### 5.1 Why Exhaustive Binary Filtering Works

The two-stage approach outperforms graph-based methods on our clustered data because:

1. **Exhaustive coverage**: Binary scan evaluates every vector. Graph navigation can get trapped in local optima, especially on clustered data where inter-cluster edges are sparse.

2. **Robust to data distribution**: The recall-vs-rf curve is smooth and predictable regardless of cluster structure. Graph methods exhibit distribution-dependent behavior.

3. **No construction cost**: Graph methods spend 20-222s building topology. Our system is ready in 0.1s — critical for streaming workloads.

The tradeoff is O(n) scan time. This is acceptable up to ~500K vectors for interactive workloads, and up to ~5M for batch/background retrieval.

### 5.2 The Rerank Factor as a Control Parameter

The rerank factor provides a single knob that continuously trades latency for recall:

- **Low rf (10-50)**: Fast (8ms), low recall (30-55%). Suitable for exploratory queries where approximate results suffice.
- **Medium rf (100-200)**: Moderate (8-9ms), good recall (69-80%). Suitable for most agent memory retrieval.
- **High rf (500-1000)**: Slower (10-15ms), excellent recall (93-97%). Suitable for critical queries where precision matters.

This tunability is a systems property that graph-based methods lack — HNSW's ef parameter provides some control but with less predictable behavior on clustered data.

### 5.3 Limitations

1. **O(n) scan**: Latency grows linearly with corpus size. Practical limit is ~500K for interactive use.
2. **Python implementation**: Current throughput (67-130 QPS) is limited by interpreter overhead. A compiled implementation of the binary scan would yield 50-100x improvement based on FAISS Binary Flat performance (11,896 QPS on the same data).
3. **Synthetic data**: All experiments use clustered random vectors. Real embedding distributions may differ.
4. **Memory for reranking**: The two-stage system stores float vectors for Stage 2, partially negating the binary compression benefit unless tiered storage is employed.

### 5.4 Future Directions

1. **Binary partitioning**: Cluster binary codes into partitions, scan only relevant partitions. Reduces O(n) to O(n/p) where p is the number of partitions.
2. **SIMD binary scan**: AVX2/NEON popcount on the binary scan loop would close the throughput gap with FAISS Binary Flat.
3. **Tiered storage**: Store binary codes in RAM, float vectors on SSD. Load float vectors only for the rf×k candidates that survive Stage 1.
4. **Adaptive rerank factor**: Dynamically select rf based on score-gap confidence in Stage 1 results.
5. **Real embedding evaluation**: Validate on OpenAI, Cohere, and GloVe embedding distributions.

---

## 6. Conclusion

bitcache demonstrates that exhaustive binary filtering combined with float reranking provides a viable retrieval architecture for persistent AI agent memory systems operating at 10K-500K vector scale. On realistic clustered embeddings, the system achieves 83.4% recall@10 — outperforming FAISS HNSW (81.9%) while requiring only 4.6 MB for the binary candidate index versus ~196 MB for HNSW. In a tiered storage configuration where float vectors reside on SSD, only the binary index occupies hot memory. On simpler distributions, recall reaches 97.3% at rf=1000 with 14.9ms latency.

The central finding is architectural: **binary scan dominates latency while reranking scales sublinearly with budget.** This means the system's bottleneck is its most optimizable component (SIMD popcount on contiguous memory), while its precision-critical component (float rerank) remains cheap regardless of corpus size. The rerank factor provides a single tunable parameter that smoothly trades latency for recall along a predictable curve — a systems property that graph-based methods lack.

Scale experiments establish 500K vectors as the practical boundary for exhaustive scan, positioning the system for agent memory workloads (typically 10K-500K memories per agent) rather than web-scale search. Beyond this boundary, binary partitioning or graph pre-filtering would extend the architecture to larger corpora.

The system builds instantly (0.1s), supports streaming mutations at 195K inserts/sec with O(1) deletion, and requires no training data or codebook calibration — properties essential for persistent agent memory where knowledge evolves continuously.

Code: https://github.com/raghavenderreddygrudhanti/bitcache

---

## References

[1] W. Xiao, Z. Wang, C. Li. "QuIVer: Rethinking ANN Graph Topology via Training-Free Binary Quantization." arXiv:2605.02171, 2026.

[2] T. Zhang, F. Ponzina, T. Rosing. "FaTRQ: Tiered Residual Quantization for LLM Vector Search in Far-Memory-Aware ANNS Systems." arXiv:2601.09985, 2026.

[3] B. J. Gutiérrez, Y. Shu, Y. Gu, M. Yasunaga, Y. Su. "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models." NeurIPS 2024.

[4] S. Jayaram Subramanya et al. "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node." NeurIPS 2019.

[5] Y. Malkov, D. Yashunin. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs." IEEE TPAMI, 2020.

[6] M. Douze et al. "The Faiss Library." arXiv:2401.08281, 2024.

[7] hnswlib. https://github.com/nmslib/hnswlib

[8] B. Naidan, L. Boytsov, Y. Malkov, D. Novak. "Non-Metric Space Library (NMSLIB)." SISAP 2019.

[9] USearch. https://github.com/unum-cloud/usearch

[10] E. Bernhardsson. "Annoy: Approximate Nearest Neighbors in C++/Python." https://github.com/spotify/annoy

[11] A. Zandieh, M. Daliri, M. Hadian, V. Mirrokni. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." ICLR 2026. arXiv:2504.19874.
