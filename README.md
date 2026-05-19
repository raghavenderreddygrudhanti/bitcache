# Bitcache

**Routed binary vector retrieval engine for AI agent memory. Rust core with Python bindings.**

FAISS is a search engine. Bitcache is a memory system.

FAISS answers: *"find similar vectors fast."*
Bitcache answers: *"what should an AI agent remember, forget, and retrieve over time?"*

---

## Why Bitcache

| Capability | Bitcache | FAISS / HNSW |
|-----------|----------|-------------|
| Streaming inserts (no rebuild) | ✅ 423K/sec | ❌ Must retrain/rebuild |
| O(1) deletes | ✅ 5.6M/sec | ❌ Tombstones or rebuild |
| Time-to-first-query | ✅ 0.23s | ❌ 2-15s (graph build) |
| Memory decay + reinforcement | ✅ Built-in | ❌ Not a concept |
| Capacity-based eviction | ✅ Automatic | ❌ Grows unbounded |
| Graph reasoning (multi-hop) | ✅ 12,200 QPS | ❌ Not supported |
| Tunable recall (single param) | ✅ Just set rf | ❌ Multiple knobs |
| Predictable latency | ✅ Deterministic | ❌ Graph-dependent |
| Mixed workload (read+write+delete) | ✅ Inline | ❌ Must batch mutations |
| Pure search throughput | 20,418 QPS | 44,597 QPS |

---

## Performance

### Throughput by Scale (Apple Silicon arm64, parallel batch)

| Scale | Exhaustive (NEON) | Routed (6.2% scan) | Best Strategy |
|-------|-------------------|--------------------|----|
| **99K** (4.5 MB) | **20,418 QPS** / 49µs | 18,464 QPS / 54µs | Exhaustive (fits L2 cache) |
| **500K** (23 MB) | 2,409 QPS / 415µs | **10,943 QPS** / 91µs | Routed (4.5x speedup) |
| **1M** (46 MB) | 942 QPS / 1.1ms | **3,998 QPS** / 250µs | Routed (4.2x speedup) |

### vs FAISS (same hardware, same dataset — 99K MiniLM, dim=384)

| Method | Recall@10 | QPS | Streaming? |
|--------|-----------|-----|-----------|
| FAISS HNSW (M=32, ef=64) | 0.973 | 44,597 | ❌ |
| hnswlib (M=32, ef=64) | 0.981 | 25,160 | ❌ |
| **Bitcache Exhaustive (NEON+Rayon)** | **0.999** | **20,418** | ✅ |
| FAISS FlatIP (brute force) | 1.000 | 19,639 | ❌ |
| FAISS IVF (nlist=128, nprobe=8) | 0.998 | 15,071 | ❌ |
| FAISS BinaryFlat | 0.741 | 17,109 | ❌ |
| Annoy (n_trees=50) | 0.468 | 4,304 | ❌ |

Bitcache **beats FAISS FlatIP and FAISS IVF** on throughput while providing streaming mutations, tunable recall, and agent memory lifecycle.

### Optimization Journey

| Stage | QPS | Improvement |
|-------|-----|-------------|
| Python prototype | 116 | 1x |
| Rust rewrite | 1,869 | 16x |
| u64 popcount | 2,221 | 19x |
| + Rayon parallel | 13,291 | 115x |
| + ARM NEON SIMD | 18,554 | 160x |
| **+ Prefetch + 4-wide unroll** | **20,418** | **176x** |

### Streaming & Memory Operations

| Operation | Throughput |
|-----------|-----------|
| Streaming insert | 423,174 vectors/sec |
| Streaming delete | 5,611,146 ops/sec |
| Agent memory save | 144,512 memories/sec |
| Agent memory retrieve (5K memories) | 4,887 QPS |
| Graph entity add | 401,869 entities/sec |
| Graph relation add | 2,905,921 relations/sec |
| Graph search + 2-hop expand | 12,197 QPS |

---

## Key Results by Paper

### Paper 1: Staged Retrieval

| Dataset | Method | Recall@10 | Latency | QPS |
|---------|--------|-----------|---------|-----|
| MiniLM 99K | BinaryOnly | 0.740 | 0.49ms | 2,038 |
| MiniLM 99K | TwoStage rf=10 | **0.999** | 0.05ms | **20,418** (batch) |
| MiniLM 99K | TwoStage rf=500 | 1.000 | 2.31ms | 432 |
| Synthetic 99K | TwoStage rf=500 | 0.933 | 2.30ms | 435 |
| SIFT1M | TwoStage rf=500 | 0.533 | 4.0ms | 250 |

**Where Bitcache works / fails:**

| Dataset | Type | Verdict |
|---------|------|---------|
| MiniLM / sentence embeddings | Semantic | ✅ 0.999 recall |
| Synthetic clustered (σ=0.15) | Gaussian | ✅ 0.933 recall (rf=500) |
| SIFT1M (image descriptors) | Non-semantic | ❌ 0.533 recall |
| Random vectors | No structure | ❌ Fails |

Binary quantization works on **semantic embeddings** (sentence-transformers, OpenAI, BGE). It fails on non-semantic data (SIFT, GIST).

### Paper 2: Semantic Routing

| Scale | Exhaustive | Routed (6.2%) | Speedup |
|-------|-----------|---------------|---------|
| 99K | 20,418 QPS | 18,464 QPS | 0.9x (fits cache) |
| 500K | 2,409 QPS | **10,943 QPS** | **4.5x** |
| 1M | 942 QPS | **3,998 QPS** | **4.2x** |

Routing works when index exceeds L2 cache. At agent-memory scale (10K-100K), exhaustive NEON scan is optimal.

Float routing vs binary routing (20K clustered):

| Method | Recall@10 |
|--------|-----------|
| Float routing (P=32, probe=4) | 0.728 |
| Binary routing (P=32, probe=4) | 0.501 |

### Paper 3: Agent Memory

| Step | Result |
|------|--------|
| Insert 10 memories | All stored with importance |
| Retrieve (reinforces) | Importance: 0.4 → 0.55 → 0.70 → 0.85 |
| Decay (rate=0.05, 5 days) | 42% importance reduction |
| Eviction at capacity=5 | Lowest 5 evicted, min remaining = 0.60 |
| Graph search + 2-hop expand | 0.011ms (12,200 QPS) |

---

## Architecture

```
Layer 6: GraphMemory        — entity-relation + multi-hop traversal
Layer 5: AgentMemory        — importance, decay, reinforcement, eviction
Layer 4: StreamingIndex     — insert/update/delete + metadata filter
Layer 3: FloatRoutedIndex   — float-space partition routing (4.5x at 500K)
Layer 2: TwoStageIndex      — binary filter + float rerank (20,418 QPS)
Layer 1: BinaryIndex        — sign-bit quantization (32x compression)
```

Each layer is independently usable and composable.

---

## Install

```bash
git clone https://github.com/raghavenderreddygrudhanti/bitcache.git
cd bitcache

# Rust
cargo test       # 23 tests
cargo build --release

# Python bindings
pip install maturin
maturin develop --release
```

---

## Quick Start

### Rust

```rust
use bitcache::{TwoStageIndex, FloatRoutedIndex, AgentMemory};

// High-throughput search (20,418 QPS parallel)
let mut index = TwoStageIndex::new(384, 10);
index.add(&vectors);
let (all_scores, all_indices) = index.search_batch(&queries, 10);

// Routed search for large scale (10,943 QPS at 500K)
let mut routed = FloatRoutedIndex::new(384, 32, 2, 100, 5);
routed.build(&vectors);
let (scores, indices) = routed.search(&query, 10);

// Agent memory with decay and eviction
let mut mem = AgentMemory::new(384, 10000, 0.1, 0.1, 10);
mem.save_memory(&vector, "user prefers morning meetings", 0.8, None, None);
let results = mem.retrieve_memory(&query, 5, 0.0);
```

### Python

```python
from bitcache import TwoStageIndex, FloatRoutedIndex, AgentMemory, GraphMemory

index = TwoStageIndex(dim=384, rerank_factor=10)
index.add(vectors)
scores, indices = index.search(query, k=10)

mem = AgentMemory(dim=384, capacity=10000, decay_rate=0.1)
mem.save_memory(vector, content="important fact", importance=0.8)
results = mem.retrieve_memory(query_vector, k=5)
```

---

## Implementation

```
src/
├── search.rs         # ARM NEON SIMD + prefetch + 4-wide unroll + Rayon
├── quantize.rs       # Sign-bit binary quantization (32x compression)
├── index.rs          # Flat binary index
├── two_stage.rs      # Binary filter → float rerank (parallel batch)
├── three_stage.rs    # Binary → 4-bit → float progressive
├── partitioned.rs    # Binary k-means routing
├── float_routed.rs   # Float-space semantic routing (parallel batch)
├── streaming.rs      # Mutable index (insert/update/delete)
├── memory.rs         # Agent memory (decay, reinforcement, eviction)
├── graph_memory.rs   # Knowledge graph + vector retrieval
└── python.rs         # PyO3 bindings
```

---

## Benchmarks

```bash
cargo run --release --bin batch_test        # Parallel throughput (20K QPS)
cargo run --release --bin routed_bench      # Routed vs exhaustive at scale
cargo run --release --bin benchmark         # Full suite
cargo run --release --bin real_embeddings_bench  # MiniLM recall
cargo run --release --bin paper_validation  # Paper claim validation
python scripts/generate_figures.py          # FAISS baselines + figures
```

---

## Papers

1. **Bitcache: Staged Binary Filtering and Float Reranking for Memory-Efficient Vector Retrieval**
2. **Partition-Local Retrieval: When Float-Space Routing Works for Low-Scan Vector Search**
3. **Bitcache Memory: A Layered Architecture for Persistent Agent Memory**

All papers in `papers/` (.md + .docx with embedded figures).

---

## Research Roadmap

| # | Direction | Status |
|---|-----------|--------|
| 1 | Binary quantized retrieval | ✅ 20,418 QPS |
| 2 | Routed search (sublinear) | ✅ 4.5x at 500K |
| 3 | Agent memory lifecycle | ✅ Decay + eviction |
| 4 | ARM NEON SIMD | ✅ 176x vs Python |
| 5 | Persistence (mmap) | 🔜 Future |
| 6 | Multi-agent shared memory | 🔜 Future |
| 7 | TurboQuant integration (2-4 bit) | 🔜 Future |

---

## References

- [QuIVer: Binary Quantization for ANN](https://arxiv.org/abs/2605.02171) (Xiao et al., 2026)
- [FaTRQ: Tiered Residual Quantization](https://arxiv.org/abs/2601.09985) (Zhang et al., 2026)
- [HippoRAG: Long-Term Memory for LLMs](https://arxiv.org/abs/2405.14831) (NeurIPS 2024)
- [HNSW](https://arxiv.org/abs/1603.09320) (Malkov & Yashunin, 2020)
- [FAISS](https://arxiv.org/abs/2401.08281) (Douze et al., 2024)

---

## License

MIT
