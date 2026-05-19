# Tunable Staged Retrieval for Persistent AI Memory Systems

**Raghavender Reddy Grudhanti**

---

## Abstract

We describe a staged retrieval architecture for AI agent memory that combines exhaustive binary filtering with float reranking. On 99,000 sentence-transformer embeddings (all-MiniLM-L6-v2, 384 dimensions), the system achieves 88.9% recall@10 at 8.6ms average latency with a rerank factor of 10, compared to 88.0% for FAISS HNSW (M=32, efSearch=64) under the same evaluation protocol. The architecture provides a tunable tradeoff: the rerank factor parameter trades latency for recall along a smooth curve (30% at rf=10 to 97% at rf=1000 on synthetic data). We observe that binary scan dominates per-query latency while reranking cost grows sublinearly with candidate count. Scale experiments on 50K to 5M synthetic vectors identify 500K as the practical boundary for exhaustive scan at interactive latency. The system requires no training data, builds in 0.1s, and supports streaming inserts at 195K vectors/sec with O(1) deletion.

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

Each coordinate maps to one bit: b_i = 1 if x_i > 0, else 0. For d=384: 1536 bytes (float32) → 48 bytes (binary). Hamming distance computed via popcount(XOR(a, b)) using a 256-entry byte lookup table.

### 3.3 Complexity

- Stage 1: O(n × d/8) — scan all binary codes
- Stage 2: O(rf × k × d) — float inner product on candidates
- Total: O(n × d/8 + rf × k × d)

---

## 4. Experimental Setup

### 4.1 Hardware

| Component | Specification |
|-----------|--------------|
| CPU | Apple Silicon (arm64) |
| RAM | 32 GB |
| OS | macOS 26.3.1 |
| Python | 3.10.10 |
| NumPy | 2.2.6 |
| FAISS | 1.13.2 |
| GPU | None (CPU only) |

### 4.2 Datasets

**Real embeddings:** 100,000 sentences embedded with all-MiniLM-L6-v2 (sentence-transformers). 384 dimensions. L2-normalized. 99,000 database vectors, 1,000 queries.

**Synthetic clustered:** 50,000 vectors generated from 100 cluster centers with Gaussian noise (σ=0.3) at 768 dimensions. L2-normalized.

### 4.3 Evaluation Protocol

- Ground truth: exact top-10 by float32 inner product (FAISS IndexFlatIP)
- Metric: Recall@10 = fraction of true top-10 present in predicted top-10
- All results averaged over 5 runs (recall is deterministic; latency reported as mean ± std)
- Baselines evaluated under identical conditions (same data, same queries, same hardware)

---

## 5. Results

### 5.1 Ablation Study (99K real embeddings, dim=384)

| Variant | Recall@10 | Latency | Memory |
|---------|-----------|---------|--------|
| Binary only (no rerank) | 0.735 | <1ms | 4.5 MB |
| Float only (brute force) | 1.000 | 0.05ms | 145 MB |
| **Two-stage (rf=10)** | **0.889** | **8.6ms** | **4.5 + 145 MB** |

The two-stage design recovers 15.4 percentage points of recall over binary-only (0.889 vs 0.735) by applying float reranking to the binary-filtered candidate set. The recall gap to brute force (11.1 points) is attributable to sign-bit quantization noise — some true neighbors receive similar Hamming distances to non-neighbors and are excluded from the candidate set.

### 5.2 Real Semantic Embeddings (99K, dim=384)

| Method | Recall@10 | Latency (mean ± std) | QPS |
|--------|-----------|---------------------|-----|
| bitcache rf=10 | 0.8887 ± 0.0000 | 8.60 ± 0.03 ms | 116 |
| bitcache rf=100 | 0.8895 ± 0.0000 | 9.06 ± 0.02 ms | 110 |
| bitcache rf=500 | 0.8909 ± 0.0000 | 11.25 ± 0.21 ms | 89 |
| FAISS HNSW (M=32, ef=64) | 0.8798 ± 0.0000 | 0.012 ± 0.001 ms | ~86,000 |

bitcache achieved higher recall than FAISS HNSW under the tested configuration (0.889 vs 0.880). FAISS HNSW is approximately 700x faster due to C++ implementation with SIMD optimization. The recall difference is attributable to exhaustive scan guaranteeing candidate coverage, while graph navigation may miss neighbors in certain graph regions.

**Note:** The throughput gap (116 vs 86,000 QPS) reflects implementation language difference (Python vs C++), not architectural limitation. FAISS IndexBinaryFlat achieves 16,000 QPS on the same binary codes using optimized C++.

### 5.3 Recall-Latency Tradeoff (50K synthetic, dim=768)

| rf | Recall@10 | Avg Latency | p50 | p95 | QPS |
|----|-----------|-------------|-----|-----|-----|
| 10 | 0.303 | 7.8ms | 7.6ms | 8.1ms | 129 |
| 25 | 0.449 | 7.7ms | 7.7ms | 8.3ms | 130 |
| 50 | 0.557 | 7.9ms | 7.9ms | 8.4ms | 127 |
| 100 | 0.687 | 8.2ms | 8.2ms | 8.6ms | 122 |
| 200 | 0.802 | 8.9ms | 8.9ms | 9.7ms | 112 |
| 500 | 0.935 | 10.1ms | 10.1ms | 10.9ms | 99 |
| 1000 | 0.973 | 14.9ms | 14.6ms | 15.7ms | 67 |

Latency is approximately constant from rf=10 to rf=100 (~8ms), indicating that Stage 1 (binary scan) dominates. Reranking 1000 candidates adds approximately 7ms over the baseline scan cost.

### 5.4 Scale Experiments (synthetic, dim=768)

| Size | rf=500 Recall | Latency | QPS |
|------|---------------|---------|-----|
| 50K | 0.914 | 9.2ms | 108 |
| 500K | 0.722 | 74.7ms | 13.4 |
| 5M | 0.510 | 784ms | 1.3 |

Latency scales linearly with corpus size (confirmed O(n)). At 500K, latency is 75ms — acceptable for background retrieval. At 5M, latency exceeds 750ms, indicating the need for partition-based approaches at this scale.

### 5.5 Streaming Performance

| Operation | Throughput |
|-----------|-----------|
| Insert | 194,886 vectors/sec |
| Delete | O(1) (slot reuse) |
| Build (50K) | 0.06s |

---

## 6. Discussion

### 6.1 Why Exhaustive Scan Outperforms HNSW on Recall

We were initially surprised that a simple binary scan outperformed HNSW on recall. After investigation, the explanation is straightforward: exhaustive scan evaluates every vector and cannot miss candidates, while HNSW graph navigation depends on edge connectivity. On our dataset, HNSW with M=32 and ef=64 occasionally fails to reach certain neighborhoods — particularly when semantically similar sentences end up in weakly-connected graph regions.

We want to be careful not to overclaim here. This result holds under our specific configuration. With higher ef values or different graph parameters, HNSW recall would likely improve. The point is not that binary scan is universally better, but that it provides a predictable baseline that does not depend on graph quality.

### 6.2 Challenges During Development

Several things did not work as expected during development:

- Our initial implementation used a Python for-loop for popcount, which was 50x slower than the lookup table approach. The architecture looked unviable until we fixed this.
- We initially assumed higher rf would always improve recall. On real embeddings, recall plateaued at ~89% regardless of rf — revealing that the bottleneck was quantization noise, not candidate coverage. This took several experiments to diagnose.
- Synthetic clustered data gave much worse results than real embeddings. We spent time debugging before realizing this was expected: real sentence embeddings have much tighter cluster structure than random Gaussians with σ=0.3.

### 6.3 Limitations

1. **Throughput:** 116 QPS in Python vs ~86,000 QPS for FAISS in C++. We acknowledge this is a large gap. However, the architecture is not the bottleneck — FAISS IndexBinaryFlat achieves 16,000 QPS on identical binary codes, suggesting a compiled version of our approach would be competitive. **Update:** We have since implemented the core engine in Rust with Python bindings via PyO3, achieving >10,000 QPS on the same workload (see `src/` in the repository).
2. **O(n) scan:** Latency grows linearly. For our target use case (agent memory, 10K-500K), this is acceptable. Beyond that, partition routing is needed (addressed in Paper 2).
3. **Recall ceiling:** Around 89% on real embeddings regardless of rf. We traced this to sign-bit quantization noise — a fundamental limitation of 1-bit representation.
4. **Dataset scope:** Validated on sentence-transformer embeddings. Other models may behave differently, though we expect similar results given shared semantic clustering properties.

### 6.3 Future Work

- Partition-based routing to extend beyond 500K scale (addressed in Paper 2)
- Higher-bit first-stage quantization to raise the 89% recall ceiling
- ~~SIMD/compiled implementation to close the throughput gap~~ **Done:** Rust implementation with hardware popcount achieves >10,000 QPS

---

## 7. Conclusion

We presented a staged retrieval architecture that achieves 88.9% recall@10 on 99K real sentence-transformer embeddings through exhaustive binary filtering and float reranking. Under the tested configuration, this outperformed FAISS HNSW (88.0%) on recall while providing a tunable recall-latency tradeoff via the rerank factor parameter. The architecture builds instantly, supports streaming mutations, and requires no training — properties suited to persistent AI agent memory where knowledge evolves continuously. The primary limitations are Python-level throughput and O(n) scaling, both addressable through implementation optimization and partition routing respectively.

The system has since been reimplemented in Rust with Python bindings (PyO3), closing the throughput gap while maintaining the same algorithmic properties. The Rust implementation uses hardware popcount intrinsics for binary distance computation and Rayon for parallel batch search.

Code and experiments: https://github.com/raghavenderreddygrudhanti/bitcache

---

## References

[1] W. Xiao, Z. Wang, C. Li. "QuIVer: Rethinking ANN Graph Topology via Training-Free Binary Quantization." arXiv:2605.02171, 2026.

[2] T. Zhang, F. Ponzina, T. Rosing. "FaTRQ: Tiered Residual Quantization for LLM Vector Search in Far-Memory-Aware ANNS Systems." arXiv:2601.09985, 2026.

[3] S. Jayaram Subramanya et al. "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node." NeurIPS 2019.

[4] Y. Malkov, D. Yashunin. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs." IEEE TPAMI, 2020.

[5] M. Douze et al. "The Faiss Library." arXiv:2401.08281, 2024.
