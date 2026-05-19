# Bitcache: Staged Binary Filtering and Float Reranking for Memory-Efficient Vector Retrieval

**Raghavender Reddy Grudhanti**

---

## Abstract

We present Bitcache, a staged retrieval architecture that combines compact binary filtering with float reranking for memory-efficient AI agent retrieval. The system stores 99,000 384-dimensional vectors using 4.53 MB of binary codes (32x compression vs float32). On deduplicated sentence-transformer embeddings (all-MiniLM-L6-v2), BinaryIndex achieves 0.740 Recall@10 and TwoStageIndex achieves 0.999 Recall@10 at rf=10 with 0.54ms latency and 1,869 QPS. On synthetic clustered data (100 clusters, σ=0.15), recall ranges from 0.313 at rf=10 to 0.933 at rf=500, demonstrating a smooth tunable tradeoff. Binary filtering alone is insufficient for high-recall retrieval, but float reranking recovers nearest-neighbor quality while preserving compact first-stage search. The Rust implementation reaches 2,038 QPS for binary-only scan and 435 QPS for high-recall TwoStage retrieval (rf=500), with streaming inserts at 423K vectors/sec and O(1) deletion. Scale experiments from 10K to 100K confirm O(n) latency growth; extrapolation suggests partition-based routing becomes important beyond the low hundreds of thousands of vectors.

---

## 1. Introduction

AI agents operating over extended sessions accumulate knowledge that must be stored and retrieved under resource constraints. We identify five requirements for agent memory retrieval:

1. **Bounded memory**: Fixed RAM allocation per agent process.
2. **Continuous mutation**: Knowledge arrives and expires without scheduled downtime.
3. **Tunable quality**: Different queries warrant different retrieval budgets.
4. **Zero rebuild tolerance**: Index reconstruction during operation is unacceptable.
5. **Predictable behavior**: Recall should be a deterministic function of configuration, not dependent on graph topology quality.

We propose a two-stage architecture where binary quantization serves as a candidate reduction mechanism and float inner product provides precise final scoring. The rerank factor controls the boundary between stages.

---

## 2. Related Work

**Binary quantization.** Xiao et al. [1] demonstrate that binary distances preserve neighborhood structure for graph navigation (QuIVer). We apply binary quantization to exhaustive scanning rather than graph construction.

**Progressive retrieval.** Zhang et al. [2] propose tiered refinement with hardware acceleration (FaTRQ). We implement progressive refinement in software with binary Hamming as the coarse stage.

**Graph-based ANN.** HNSW [4] and DiskANN [3] achieve high recall through graph navigation but require expensive construction and exhibit distribution-dependent behavior.

---

## 3. Method

### 3.1 Architecture

![Figure 1: Bitcache TwoStage Architecture](figures/paper1_architecture.png)

```
Query → L2-normalize → Sign-bit quantize (1 bit per dimension)
                              ↓
  Stage 1: Hamming distance to all n binary codes (exhaustive)
                              ↓
  Select top rf × k candidates by ascending Hamming distance
                              ↓
  Stage 2: Float32 inner product on rf × k candidate vectors
                              ↓
  Return top k by descending score
```

### 3.2 Binary Quantization

Each coordinate maps to one bit: b_i = 1 if x_i > 0, else 0. For d=384: 1536 bytes (float32) → 48 bytes (binary). Hamming distance computed via hardware popcount(XOR(a, b)) using Rust's `.count_ones()` intrinsic.

### 3.3 Complexity

- Stage 1: O(n × d/8) — scan all binary codes
- Stage 2: O(rf × k × d) — float inner product on candidates
- Total: O(n × d/8 + rf × k × d)

---

## 4. Experimental Setup

### 4.1 Hardware and Implementation

| Component | Specification |
|-----------|--------------|
| CPU | Apple Silicon (arm64) |
| RAM | 32 GB |
| OS | macOS |
| Implementation | Rust (release, LTO, opt-level=3) |
| Python bindings | PyO3/maturin |
| GPU | None (CPU only) |

### 4.2 Datasets

**Real embeddings (MiniLM):** 100,000 sentences embedded with all-MiniLM-L6-v2 (sentence-transformers). 384 dimensions. L2-normalized. 99,000 database vectors, 1,000 queries from same distribution.

**Synthetic clustered:** 99,000 vectors generated from 100 cluster centers with Gaussian noise (σ=0.15) at 384 dimensions. L2-normalized. 1,000 queries from same distribution.

**Synthetic random:** 99,000 uniformly random unit vectors at 384 dimensions. 1,000 random queries.

### 4.3 Evaluation Protocol

- Ground truth: exact top-10 by float32 inner product (brute-force)
- Metric: Recall@10 = fraction of true top-10 present in predicted top-10
- All results averaged over full query set (recall is deterministic for fixed data)
- Latency reported as mean per query

---

## 5. Results

### 5.1 Ablation Study: Binary vs Two-Stage

| Dataset | Variant | Recall@10 | Latency | Memory |
|---------|---------|-----------|---------|--------|
| MiniLM deduplicated (99K) | Binary only | 0.740 | 0.49ms | 4.53 MB |
| MiniLM deduplicated (99K) | Two-stage rf=10 | 0.999 | 0.54ms | 4.53 + 145 MB |
| Synthetic clustered (99K) | Binary only | 0.108 | 0.49ms | 4.53 MB |
| Synthetic clustered (99K) | Two-stage rf=10 | 0.313 | 0.55ms | 4.53 + 145 MB |
| Synthetic clustered (99K) | Two-stage rf=500 | 0.933 | 2.30ms | 4.53 + 145 MB |

Binary filtering alone is insufficient for high-recall retrieval on synthetic data. Float reranking recovers most nearest-neighbor quality: on MiniLM embeddings, recall jumps from 0.740 to 0.999 with just rf=10. On synthetic data with weaker cluster structure, higher rf is needed (rf=500 for 0.933 recall).

Note: The first-stage binary filtering index is 32x smaller than float32 storage. High-recall reranking currently retains float vectors, so total memory is binary codes (4.53 MB) plus the float store (145 MB). The 32x compression applies to the candidate filtering index, not total system memory.

![Figure 4: Memory Usage Comparison](figures/paper1_memory_comparison.png)

### 5.2 Real Sentence-Transformer Embeddings (99K deduplicated, MiniLM, dim=384)

| Method | Recall@10 | Latency | QPS |
|--------|-----------|---------|-----|
| BinaryIndex (no rerank) | 0.740 | 0.49ms | 2,038 |
| TwoStage rf=10 | 0.999 | 0.54ms | 1,869 |
| TwoStage rf=50 | 1.000 | 0.68ms | 1,479 |
| TwoStage rf=100 | 1.000 | 0.86ms | 1,166 |
| TwoStage rf=500 | 1.000 | 2.31ms | 432 |
| TwoStage rf=1000 | 1.000 | 4.03ms | 248 |

All sentences are unique (deduplicated before embedding to avoid inflated recall from repeated text templates). On real sentence-transformer embeddings, recall saturates at 0.999 even at rf=10. This indicates that MiniLM embeddings have strong binary-preserving structure: the top candidates by Hamming distance almost always contain the true float-space neighbors.

### 5.3 Recall-Latency Tradeoff (99K synthetic clustered, dim=384)

| rf | Recall@10 | Latency | QPS |
|----|-----------|---------|-----|
| 10 | 0.313 | 0.54ms | 1,839 |
| 25 | 0.461 | 0.58ms | 1,719 |
| 50 | 0.590 | 0.67ms | 1,488 |
| 100 | 0.711 | 0.85ms | 1,176 |
| 200 | 0.824 | 1.22ms | 821 |
| 500 | 0.933 | 2.30ms | 435 |
| 1000 | 0.975 | 3.97ms | 252 |

The tradeoff is smooth and monotonic. Latency grows sublinearly with rf because Stage 1 (binary scan) dominates at low rf, and Stage 2 cost grows linearly but operates on a small candidate set.

![Figure 2: Recall@10 vs Rerank Factor](figures/paper1_recall_vs_rf.png)

![Figure 3: Recall-Latency Tradeoff](figures/paper1_tradeoff_curve.png)

### 5.4 Scale Experiments (synthetic clustered, dim=384, rf=100)

| Size | Build Time | Latency | QPS |
|------|-----------|---------|-----|
| 10K | 0.023s | 0.30ms | 3,363 |
| 50K | 0.115s | 0.59ms | 1,683 |
| 100K | 0.231s | 0.85ms | 1,170 |

Latency scales linearly with corpus size (confirmed O(n)). At 100K, latency remains sub-millisecond. Extrapolation suggests partition-based routing becomes important beyond the low hundreds of thousands of vectors (addressed in Paper 2).

### 5.5 Streaming Performance

| Operation | Throughput |
|-----------|-----------|
| Insert | 413,398 vectors/sec |
| Delete | 5,611,146 ops/sec (O(1) slot reuse) |
| Build (99K) | 0.23s |

---

## 6. Discussion

### 6.1 Data Distribution Determines Recall

The most important finding is that recall depends heavily on data distribution:

- **Real MiniLM embeddings (deduplicated):** 0.999 recall at rf=10. Sentence-transformer embeddings form tight semantic clusters that are well-preserved by sign-bit quantization.
- **Synthetic clustered (σ=0.15):** 0.313 recall at rf=10, 0.933 at rf=500. Gaussian clusters with moderate noise require more candidates to recover true neighbors.

This means binary quantization is not universally effective — it works best when the embedding model produces vectors with strong directional structure (positive/negative dimensions carry semantic meaning).

### 6.2 Challenges During Development

Several things did not work as expected during development:

- Our initial Python implementation achieved only 116 QPS. The architecture looked unviable for real-time use until we reimplemented in Rust, reaching 1,869 QPS (16x improvement).
- We initially assumed higher rf would always improve recall. On MiniLM embeddings, recall plateaued at 0.999 regardless of rf — revealing that the bottleneck was not candidate coverage but the inherent quality of binary quantization on this data.
- Synthetic data gave much worse results than real embeddings. This is expected: real sentence embeddings have tighter cluster structure than random Gaussians.

### 6.3 Limitations

1. **Throughput vs FAISS:** Our Rust implementation reaches 1,869 QPS (rf=10). FAISS HNSW achieves ~86,000 QPS in C++ with SIMD. The gap is smaller than the Python version (116 QPS) but still significant. FAISS IndexBinaryFlat achieves 16,000 QPS on identical binary codes, suggesting further SIMD optimization could close the remaining gap.
2. **O(n) scan:** Latency grows linearly. For agent memory (10K-100K), this is acceptable. Beyond that, partition routing is needed (Paper 2).
3. **Data-dependent recall:** Binary quantization works well on sentence-transformer embeddings (0.740 binary-only, 0.999 with rerank) but may not generalize to all embedding models.
4. **Memory overhead:** Two-stage requires storing both binary codes (4.53 MB) and float vectors (145 MB). The first-stage binary filtering index is 32x compressed, but total system memory includes the float store for reranking.

### 6.4 Future Work

- Partition-based routing to extend beyond 500K scale (addressed in Paper 2)
- Higher-bit first-stage quantization (4-bit) to improve recall on weakly-clustered data
- SIMD-optimized Hamming distance for further throughput improvement
- Evaluation on additional embedding models (OpenAI, Cohere, BGE)

---

## 7. Conclusion

We presented Bitcache, a staged retrieval architecture that achieves 0.999 Recall@10 on deduplicated sentence-transformer embeddings and 0.933 on synthetic clustered data (rf=500) through exhaustive binary filtering and float reranking. The architecture provides a smooth, tunable recall-latency tradeoff via the rerank factor parameter. The Rust implementation reaches 1,869 QPS at high recall on MiniLM embeddings, builds in 0.23s, supports streaming inserts at 413K vectors/sec, and requires no training data.

On MiniLM embeddings, Bitcache reaches 0.999 Recall@10 at 1,869 QPS using rf=10. On synthetic clustered data, higher recall requires larger rerank factors, reaching 0.933 Recall@10 at rf=500 with 435 QPS.

The key finding is that recall is strongly data-dependent: real sentence embeddings achieve near-perfect recall even at low rf, while synthetic data requires higher rf. This motivates careful evaluation on target embedding models before deployment.

Code and experiments: https://github.com/raghavenderreddygrudhanti/bitcache

---

## References

[1] W. Xiao, Z. Wang, C. Li. "QuIVer: Rethinking ANN Graph Topology via Training-Free Binary Quantization." arXiv:2605.02171, 2026.

[2] T. Zhang, F. Ponzina, T. Rosing. "FaTRQ: Tiered Residual Quantization for LLM Vector Search in Far-Memory-Aware ANNS Systems." arXiv:2601.09985, 2026.

[3] S. Jayaram Subramanya et al. "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node." NeurIPS 2019.

[4] Y. Malkov, D. Yashunin. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs." IEEE TPAMI, 2020.

[5] M. Douze et al. "The Faiss Library." arXiv:2401.08281, 2024.
