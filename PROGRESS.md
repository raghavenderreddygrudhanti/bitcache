# bitcache — Progress Tracker

---

## Core Insight (Discovered During Development)

**bitcache is not a compression algorithm. It is a retrieval budget control system.**

Binary filtering is a tunable candidate reduction mechanism. The rerank factor (rf) is the control parameter that trades latency for recall along a smooth, predictable curve.

---

## Key Findings

### Finding 1: Two-stage rf=500 achieves 92.5% recall@10
- 50K clustered vectors, dim=768
- Exhaustive binary scan + float rerank on top 5000 candidates
- Competitive with HNSW-class methods
- Build time: 0.1s (vs 20-200s for graph methods)

### Finding 2: Recall-vs-rf is a smooth log-linear curve
```
rf=10   → 30.3%   | 7.76ms  | 128.8 QPS
rf=25   → 44.9%   | 7.69ms  | 130.0 QPS
rf=50   → 55.7%   | 7.90ms  | 126.5 QPS
rf=100  → 68.7%   | 8.18ms  | 122.2 QPS
rf=200  → 80.2%   | 8.94ms  | 111.8 QPS
rf=500  → 93.5%   | 10.14ms | 98.6 QPS
rf=1000 → 97.3%   | 14.88ms | 67.2 QPS
```
This is the strongest contribution — a controllable, predictable tradeoff.

### Finding 3: Binary scan dominates latency, NOT reranking
- Latency is flat from rf=10 to rf=100 (~8ms)
- Reranking 100 vs 1000 candidates adds only 1ms
- Binary scan of 50K vectors costs ~7.5ms regardless of rf
- This means: reranking is cheap, binary scan is the bottleneck
- Implication: SIMD/Rust optimization of binary scan would give massive speedup

### Finding 4: 97.3% recall at rf=1000 with only 15ms latency
- Near-perfect retrieval quality
- Latency only doubles from rf=10 to rf=1000 (8ms → 15ms)
- QPS stays above 67 even at highest rf
- Sweet spot: rf=200-500 (80-93% recall at 9-10ms)

### Finding 5: Three-stage only makes sense with tiered storage
- In-RAM: three-stage uses MORE memory than two-stage (stores binary + 4-bit + float)
- With SSD: three-stage keeps only binary in RAM, loads float on demand
- Three-stage recall: 77% (vs two-stage 58% at matched budget)
- Real value appears when Stage 3 is on disk (FaTRQ/DiskANN territory)

### Finding 4: Binary filtering value diminishes as rf grows
- At rf=500, binary stage filters out only 90% of vectors
- The "32x compression" story is about storage, not about search speedup at high rf
- The real value of binary is: instant build + streaming mutations + memory efficiency

### Finding 5: 14-method comparison — bitcache outperforms 11 of 14
- Beats: FAISS HNSW, hnswlib, nmslib, USearch, Annoy, Voyager, PyNNDescent, FAISS PQ, FAISS Binary, FAISS IVF
- Loses to: FAISS Flat (exact), sklearn BallTree (exact) — both use 32x more memory
- Unique advantage: no other binary method recovers recall after quantization

---

## Architecture Evolution

### What we started with:
```
vectors → binary quantization → flat scan → results
```
Recall: 6%. Useless.

### What we built:
```
vectors → binary codes + float storage
query → binary scan (all vectors) → top rf*k candidates → float rerank → top k
```
Recall: 57-92% depending on rf. Competitive.

### What we discovered the architecture actually is:
```
Query Execution Pipeline:
  Query
    ↓
  Cheap exhaustive binary scan (O(n), but fast per-vector)
    ↓
  Candidate budget selection (rf = control parameter)
    ↓
  Precise float reranking (O(rf*k))
    ↓
  Final results

Control parameters:
  rf → recall/latency tradeoff
  capacity → memory/eviction tradeoff
  decay_rate → freshness/retention tradeoff
```

### What the future architecture looks like (not yet built):
```
Stage 1: Binary codes in L2/RAM (96 bytes/vec) — scan all
Stage 2: 4-bit codes in RAM (384 bytes/vec) — score survivors
Stage 3: Float vectors on SSD (3072 bytes/vec) — load on demand

Only Stage 1 must fit in RAM for the full corpus.
Stage 2 and 3 are loaded proportional to rf, not n.
```

---

## What's Built (Code)

| Module | File | Tests | Status |
|--------|------|-------|--------|
| Binary quantization | `bitcache/quantize.py` | 3 | ✅ |
| Hamming search | `bitcache/search.py` | 5 | ✅ |
| BinaryIndex (flat) | `bitcache/index.py` | 11 | ✅ |
| VamanaIndex (graph) | `bitcache/graph.py` | 6 | ✅ |
| TwoStageIndex | `bitcache/two_stage.py` | 7 | ✅ |
| ThreeStageIndex | `bitcache/three_stage.py` | — | Built, not committed |
| StreamingIndex | `bitcache/streaming.py` | 11 | ✅ |
| AgentMemory | `bitcache/memory.py` | 12 | ✅ |
| GraphMemory | `bitcache/graph_memory.py` | 13 | ✅ |
| **Total tests** | | **60** | **All passing** |

---

## Benchmarks Run

| Benchmark | File | Key Result |
|-----------|------|------------|
| Recall + memory (100K) | `benchmarks/eval_full.py` | 58% recall, 32x compression |
| FAISS comparison | `benchmarks/eval_real_data.py` | bitcache beats FAISS binary 9x on recall |
| 14-method comparison | `benchmarks/eval_all_dbs.py` | #3 overall, #1 among approximate methods |
| **Recall vs rf curve** | `benchmarks/eval_rf_curve.py` | **97.3% at rf=1000, 93.5% at rf=500** |
| Three-stage test | (run inline) | 77% recall with 4-bit intermediate |

### Recall-vs-RF Full Table (50K vectors, dim=768, k=10)

| rf | Recall@10 | Avg Latency | p50 | p95 | QPS | Candidates | Float Ops |
|----|-----------|-------------|-----|-----|-----|------------|-----------|
| 10 | 0.303 | 7.76ms | 7.55ms | 8.10ms | 128.8 | 100 | 76,800 |
| 25 | 0.449 | 7.69ms | 7.66ms | 8.26ms | 130.0 | 250 | 192,000 |
| 50 | 0.557 | 7.90ms | 7.87ms | 8.41ms | 126.5 | 500 | 384,000 |
| 100 | 0.687 | 8.18ms | 8.17ms | 8.60ms | 122.2 | 1,000 | 768,000 |
| 200 | 0.802 | 8.94ms | 8.91ms | 9.69ms | 111.8 | 2,000 | 1,536,000 |
| 500 | 0.935 | 10.14ms | 10.14ms | 10.93ms | 98.6 | 5,000 | 3,840,000 |
| 1000 | 0.973 | 14.88ms | 14.64ms | 15.65ms | 67.2 | 10,000 | 7,680,000 |

### Critical Architectural Insight

Binary scan dominates latency. Reranking is cheap. This means:
- The architecture is sound (reranking scales well)
- The bottleneck (binary scan) is the most optimizable part (SIMD popcount)
- Increasing rf from 10→1000 only doubles latency (8ms→15ms) while recall goes 30%→97%

### Charts Generated (Local)

- `benchmarks/results/recall_vs_rf.png` — recall curve (central paper figure)
- `benchmarks/results/recall_vs_latency.png` — tradeoff curve
- `benchmarks/results/qps_vs_rf.png` — throughput degradation
- `docs/recall_comparison.png` — 14-method bar chart
- `docs/pareto.png` — recall vs QPS scatter
- `docs/memory_comparison.png` — memory usage bars
- `docs/recall_vs_rf.png` — earlier version of rf curve

---

## Paper Status

| Section | Content | Status |
|---------|---------|--------|
| Abstract | 92.5% recall, 32x compression, 14-method comparison | Needs update |
| Introduction | 5 requirements for agent memory | ✅ |
| Related work | QuIVer, FaTRQ, HippoRAG, DiskANN, HNSW, FAISS | ✅ |
| Architecture | 6-layer system description | ✅ |
| Evaluation | 14-method table, memory, streaming, decay | Needs rf curve + 92.5% |
| Discussion | Why two-stage works, limitations, future | Needs tiered storage insight |
| Conclusion | Positioning as retrieval budget control system | Needs rewrite |

---

## What's Still Weak

1. **Throughput:** 15-73 QPS (Python overhead). Architecture is sound, implementation is slow.
2. **O(n) scan:** Works at 50K-100K. At 1M+ needs partitioning or graph pre-filter.
3. **No real embeddings:** All benchmarks on synthetic clustered data. Need OpenAI/Cohere/GloVe.
4. **No latency decomposition:** Don't know what % of time is Stage 1 vs Stage 2.
5. **No scale experiment:** Haven't tested 500K or 5M to find where O(n) breaks.

---

## Next Actions (7-Day Plan)

1. ~~**Run scale test (50K, 500K, 5M)**~~ — DONE. Boundary at 500K.
2. ~~**Run realistic embeddings benchmark**~~ — DONE. 83.4% beats HNSW 81.9%.
3. ~~**Run TRUE public embedding dataset**~~ — DONE. 99.98% recall on sentence-transformers. BEATS HNSW.
4. **NumPy vectorize binary scan** — biggest engineering ROI.
5. **Improve paper figures** — cleaner styling, Pareto frontier line.
6. **Final paper rewrite** — clarity, credibility, no hype.
7. **Submit for early review** — get feedback.

## STRONGEST RESULT (Real Embeddings)

### 9K scale (all-MiniLM-L6-v2, 384d):
- **bitcache rf=10: 99.98% recall@10 at 0.74ms**
- FAISS HNSW: 99.2% recall
- FAISS Binary (no rerank): 81.1% recall

### 100K scale (all-MiniLM-L6-v2, 384d):
- **bitcache rf=10: 88.9% recall@10 at 8.5ms**
- FAISS HNSW: 87.2% recall
- FAISS Binary (no rerank): 73.5% recall

Key insight: On real semantic embeddings, bitcache outperforms FAISS HNSW
at BOTH scales. Binary sign-bit quantization preserves semantic manifold
structure. Recall is flat across rf values at 100K — binary top-100 already
contains the true neighbors. The synthetic experiments are conservative
lower bounds.

## DO NOT DO YET
- ~~Partition routing~~ — DONE (Gen2). Works at 100K, fails at 500K+ (routing quality).
- Float-space routing (Gen3) — future work, identified but not started
- Rust rewrite — not needed for paper
- Distributed retrieval — not needed for paper
- Adaptive rf — described in paper, not implemented

## Gen2 Diagnostic (Critical Finding)

Partition hit rate at scale:
- 99K: ~90% (works — true neighbors are in probed partitions)
- 500K: 9.2% (fails — binary k-means doesn't preserve semantic neighborhoods)
- 1M: 11.2% (fails — same reason)

Root cause: binary k-means centroids don't capture float-space semantic structure at scale.
Solution identified: float-space centroid routing (Gen3 future work).

Gen1 exhaustive still works at 500K (82.5% recall at rf=1000, 81ms). The retrieval
architecture is sound. Only the routing mechanism needs improvement.

## PROJECT STATUS: FROZEN FOR SUBMISSION

Paper is ready. Code is stable. 68 tests passing. Results validated on real embeddings.
Do not add features. Focus on submission and review.

## Scale Test Results (Critical Finding)

| Size | rf | Recall | Latency | QPS | Verdict |
|------|-----|--------|---------|-----|---------|
| 50K | 500 | 91.4% | 9.2ms | 108 | Excellent |
| 500K | 500 | 72.2% | 74.7ms | 13.4 | Usable (agent memory) |
| 5M | 500 | 51.0% | 784ms | 1.3 | Boundary — needs partitioning |

**Key finding:** Exhaustive binary scan is practical up to ~500K vectors for interactive workloads. Beyond 500K, partitioning or graph pre-filtering is required.

**Latency scales linearly:** 10x data → 10x latency (confirmed O(n) behavior).

**Recall drops with scale at fixed rf:** Because rf=500 means 5000 candidates regardless of corpus size. At 50K that's 10% coverage. At 5M that's 0.1%.

**Implication for paper:** bitcache is positioned for persistent AI agent memory (typically 10K-500K memories per agent), not billion-scale web search.

---

## Framing for Paper/Presentation

**Wrong framing:** "We built a better vector database"
**Right framing:** "Binary filtering + float reranking provides a smooth tunable recall-latency tradeoff for persistent AI memory retrieval"

**Wrong headline:** "32x compression"
**Right headline:** "92.5% recall with controllable retrieval budget and instant index construction"

**Wrong comparison:** "bitcache vs Pinecone"
**Right comparison:** "bitcache's recall-vs-rf curve vs fixed-recall graph methods"

---

## External Contributions (For EB1)

| Project | What | Status |
|---------|------|--------|
| langchain-community | Salesforce CRM integration | Issue #661 open, PR blocked by access |
| agno | Wolfram Alpha tool | PR #7941 open |
| turbovec | Agno integration | PR #24 closed, superseded by maintainer's #29 (credited) |
| turbovec | Metadata filtering | PR #26 closed |
| turbovec | Distortion benchmark | PR #28 closed |
| turbovec | Issue #23 (Agno integration) | Closed as completed by #29 |
| turbovec | Issue #25 (metadata filtering) | Open |
| turbovec | Issue #27 (distortion benchmark) | Open |

---

## Repository

https://github.com/raghavenderreddygrudhanti/bitcache
