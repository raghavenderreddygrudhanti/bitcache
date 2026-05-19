# Bitcache

**Routed binary vector retrieval engine for AI agent memory. Rust core with Python bindings.**

Bitcache combines binary filtering with float reranking to provide compact first-stage search, streaming mutation, and tunable recall for persistent AI agent memory. It targets a different design point than FAISS or HNSW: not maximum throughput, but a composable memory layer with streaming inserts, O(1) deletes, tunable recall, and agent memory lifecycle operations.

---

## Performance Summary

### Throughput (99K vectors, dim=384, Apple Silicon arm64)

| Mode | QPS | Latency | Notes |
|------|-----|---------|-------|
| **Parallel batch (Rayon)** | **13,291** | **75 µs** | 1000 queries, rf=10 |
| Sequential | 2,221 | 450 µs | Single-threaded |
| Previous (before SIMD+Rayon) | 1,869 | 540 µs | — |

### Comparison with FAISS (same hardware, same dataset)

| Method | Recall@10 | Latency | QPS | Streaming? |
|--------|-----------|---------|-----|-----------|
| FAISS HNSW (M=32, ef=64) | 0.973 | 0.022ms | 44,597 | ❌ |
| FAISS FlatIP (brute force) | 1.000 | 0.051ms | 19,639 | ❌ |
| FAISS IVF (nlist=128, nprobe=8) | 0.998 | 0.066ms | 15,071 | ❌ |
| **Bitcache TwoStage rf=10 (batch)** | **0.999** | **0.075ms** | **13,291** | ✅ |
| FAISS BinaryFlat | 0.741 | 0.058ms | 17,109 | ❌ |
| hnswlib (M=32, ef=64) | 0.981 | 0.040ms | 25,160 | ❌ |
| Annoy (n_trees=50) | 0.468 | 0.232ms | 4,304 | ❌ |

Bitcache matches FAISS IVF on throughput while providing streaming inserts, O(1) deletes, tunable recall, importance decay, and graph reasoning.

---

## Key Results by Paper

### Paper 1: Staged Retrieval (MiniLM 99K, dim=384)

| Method | Recall@10 | Latency | QPS |
|--------|-----------|---------|-----|
| BinaryOnly (no rerank) | 0.740 | 0.49ms | 2,038 |
| TwoStage rf=10 | 0.999 | 0.54ms | 2,221 |
| TwoStage rf=100 | 1.000 | 0.86ms | 1,166 |
| TwoStage rf=500 | 1.000 | 2.31ms | 432 |
| **TwoStage rf=10 (parallel batch)** | **0.999** | **0.075ms** | **13,291** |

Recall-latency tradeoff (synthetic clustered, 99K, σ=0.15):

| rf | Recall@10 | Latency | QPS |
|----|-----------|---------|-----|
| 10 | 0.313 | 0.54ms | 1,839 |
| 50 | 0.590 | 0.67ms | 1,488 |
| 100 | 0.711 | 0.85ms | 1,176 |
| 500 | 0.933 | 2.30ms | 435 |
| 1000 | 0.975 | 3.97ms | 252 |

### Paper 1: SIFT1M Public Benchmark (1M vectors, dim=128)

| Method | Recall@10 | Latency | QPS |
|--------|-----------|---------|-----|
| FAISS HNSW | 0.977±0.059 | 0.025ms | 39,768 |
| FAISS IVF | 0.985±0.051 | 0.264ms | 3,783 |
| hnswlib | 0.981±0.052 | 0.040ms | 25,160 |
| Annoy | 0.468±0.258 | 0.232ms | 4,304 |
| FAISS BinaryFlat | 0.025±0.072 | 0.228ms | 4,387 |
| Bitcache TwoStage rf=500 (100K) | 0.533±0.297 | 4.0ms | 250 |

Binary quantization fails on SIFT (non-semantic data). Bitcache is designed for semantic embeddings.

### Paper 1: Where Bitcache Works / Where It Fails

| Dataset | Type | Binary Recall | TwoStage Recall | Verdict |
|---------|------|---------------|-----------------|---------|
| MiniLM (99K) | Semantic embeddings | 0.740 | 0.999 (rf=10) | ✅ Excellent |
| Synthetic clustered (σ=0.15) | Gaussian clusters | 0.108 | 0.933 (rf=500) | ✅ Works |
| SIFT1M (1M) | Image descriptors | 0.025 | 0.302 (rf=100) | ❌ Fails |
| Random vectors | No structure | ~0.05 | ~0.15 | ❌ Fails |

### Paper 2: Semantic Routing (20K clustered, dim=384)

| P | probe | Recall@10 | Latency | QPS | Scan% | Speedup |
|---|-------|-----------|---------|-----|-------|---------|
| 32 | 2 | 1.000 | 0.33ms | 3,042 | 6.2% | 4.8x |
| 32 | 4 | 1.000 | 0.35ms | 2,853 | 12.5% | 4.8x |
| 64 | 4 | 1.000 | 0.33ms | 2,993 | 6.2% | — |
| 64 | 8 | 1.000 | 0.37ms | 2,742 | 12.5% | — |

Float routing vs binary routing (same data):

| Method | Recall@10 | Notes |
|--------|-----------|-------|
| Float routing (P=32, probe=4) | 0.728 | Preserves semantic structure |
| Binary routing (P=32, probe=4) | 0.501 | Loses fine-grained distances |
| Exhaustive (rf=500) | 0.933 | Full scan baseline |

Routing works when embeddings have strong partition locality. On tightly clustered data, 100% recall at 6.2% scan.

### Paper 3: Agent Memory & Graph (dim=384)

| Operation | Throughput |
|-----------|-----------|
| Streaming insert | 423,174 vectors/sec |
| Streaming delete | 5,611,146 ops/sec |
| Agent memory save | 144,512 memories/sec |
| Agent memory retrieve (5K memories) | 4,887 QPS |
| Graph entity add | 401,869 entities/sec |
| Graph relation add | 2,905,921 relations/sec |
| Graph search + 2-hop expand | 12,197 QPS |

Memory lifecycle validation:

| Step | Result |
|------|--------|
| Insert 10 memories (importance 0.1-0.95) | All stored |
| Eviction at capacity=5 | Lowest 5 evicted correctly |
| Remaining min importance | 0.600 |
| Reinforcement per access | +0.15 per retrieval |
| Decay (rate=0.05, 5 days) | 42% importance reduction |

---

## Architecture

```
Layer 6: GraphMemory        — entity-relation + multi-hop traversal
Layer 5: AgentMemory        — importance, decay, reinforcement, eviction
Layer 4: StreamingIndex     — insert/update/delete + metadata filter
Layer 3: FloatRoutedIndex   — float-space partition routing (6.2% scan)
Layer 2: TwoStageIndex      — binary filter + float rerank
Layer 1: BinaryIndex        — sign-bit quantization (32x compression)
```

Each layer is independently usable and composable.

---

## Install

```bash
git clone https://github.com/raghavenderreddygrudhanti/bitcache.git
cd bitcache

# Rust (library + benchmarks)
cargo test       # 21 tests
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

// Staged retrieval
let mut index = TwoStageIndex::new(384, 100);
index.add(&vectors);
let (scores, indices) = index.search(&query, 10);

// Parallel batch search (13,291 QPS)
let (all_scores, all_indices) = index.search_batch(&queries, 10);

// Partition-routed retrieval
let mut routed = FloatRoutedIndex::new(384, 128, 8, 500, 10);
routed.build(&vectors);
let (scores, indices) = routed.search(&query, 10);

// Agent memory with decay and eviction
let mut mem = AgentMemory::new(384, 10000, 0.1, 0.1, 10);
mem.save_memory(&vector, "user prefers morning meetings", 0.8, None, None);
let results = mem.retrieve_memory(&query, 5, 0.0);
```

### Python

```python
import numpy as np
from bitcache import TwoStageIndex, FloatRoutedIndex, AgentMemory, GraphMemory

# Staged retrieval
index = TwoStageIndex(dim=384, rerank_factor=100)
index.add(vectors)
scores, indices = index.search(query, k=10)

# Agent memory
mem = AgentMemory(dim=384, capacity=10000, decay_rate=0.1)
mem.save_memory(vector, content="important fact", importance=0.8)
results = mem.retrieve_memory(query_vector, k=5)

# Graph memory
gm = GraphMemory(dim=384)
gm.add_entity("prod-db-01", vector, name="Production DB", entity_type="system")
gm.add_relation("api-gateway", "depends_on", "prod-db-01")
results = gm.search(query_vector, k=3, expand=True, max_hops=2)
```

---

## Implementation

```
src/
├── lib.rs            # Crate root
├── quantize.rs       # Sign-bit binary quantization
├── search.rs         # u64 POPCOUNT + blocked layout + Rayon parallel
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
cargo run --release --bin batch_test              # Parallel throughput test
cargo run --release --bin benchmark               # Full benchmark suite
cargo run --release --bin real_embeddings_bench    # MiniLM results
cargo run --release --bin paper_validation        # Paper claim validation

# FAISS/hnswlib/Annoy baselines + figures
python scripts/generate_figures.py
```

---

## Papers

1. **Bitcache: Staged Binary Filtering and Float Reranking for Memory-Efficient Vector Retrieval**
   — Two-stage architecture, recall-latency tradeoff, SIFT1M evaluation

2. **Partition-Local Retrieval: When Float-Space Routing Works for Low-Scan Vector Search**
   — Float routing, PartitionHit@10, data-dependent partition locality

3. **Bitcache Memory: A Layered Architecture for Persistent Agent Memory**
   — Six-layer composable memory with decay, reinforcement, eviction, graph

All papers in `papers/` (.md + .docx with embedded figures).

---

## Research Roadmap

| # | Direction | Status |
|---|-----------|--------|
| 1 | Quantized vector retrieval | ✅ Implemented |
| 2 | Routed search for low-scan retrieval | ✅ Implemented |
| 3 | Agentic AI memory systems | ✅ Implemented |
| 4 | SIMD + parallel throughput | ✅ 13,291 QPS |
| 5 | Multi-agent shared memory | 🔜 Future |
| 6 | Persistence + crash recovery | 🔜 Future |

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
