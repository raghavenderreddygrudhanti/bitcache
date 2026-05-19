# Bitcache

**Routed binary vector retrieval engine for AI agent memory. Rust core with Python bindings.**

Bitcache compresses high-dimensional embeddings into compact binary representations and uses intelligent float-space routing to scan only the most relevant vector partitions — achieving high recall at a fraction of the memory and latency cost.

---

## Key Results

Real sentence-transformer embeddings (99K vectors, all-MiniLM-L6-v2, 384d):

| Method | Recall@10 | Latency | Scan% | Speedup |
|--------|-----------|---------|-------|---------|
| Gen1 (exhaustive) | 0.891 | 8.6ms | 100% | 1x |
| **Gen3 (float routed)** | **0.892** | **3.0ms** | **6.2%** | **3.8x** |
| FAISS HNSW (M=32, ef=64) | 0.880 | 0.01ms | — | — |
| FAISS Binary (no rerank) | 0.735 | — | — | — |

Rust implementation benchmarks (99K synthetic clustered, dim=384):

| Method | Recall@10 | Latency | QPS |
|--------|-----------|---------|-----|
| TwoStage rf=100 | 0.711 | 0.85ms | 1,182 |
| TwoStage rf=500 | 0.933 | 2.3ms | 429 |
| FloatRouted (P=32, probe=4) | 0.700 | 0.33ms | 2,838 |
| Routed speedup vs exhaustive | — | — | **4.9x** |

Streaming & memory operations:

| Operation | Throughput |
|-----------|-----------|
| Streaming insert | 423K vectors/sec |
| Streaming delete | 5.6M ops/sec |
| Agent memory save | 144K memories/sec |
| Agent memory retrieve | 4,887 QPS |
| Graph search + 2-hop expand | 12,200 QPS |

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
cargo test
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

// Staged retrieval (Gen1)
let mut index = TwoStageIndex::new(384, 100);
index.add(&vectors);  // flat f32 slice, n * 384
let (scores, indices) = index.search(&query, 10);

// Partition-routed retrieval (Gen3)
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

# Staged retrieval (Gen1)
index = TwoStageIndex(dim=384, rerank_factor=100)
vectors = np.random.randn(10000, 384).astype(np.float32)
index.add(vectors)

query = vectors[0]
scores, indices = index.search(query, k=10)

# Partition-routed retrieval (Gen3)
index = FloatRoutedIndex(dim=384, n_partitions=128, n_probe=8, rerank_factor=500)
index.build(vectors)
scores, indices = index.search(query, k=10)

# Agent memory with decay and eviction
mem = AgentMemory(dim=384, capacity=10000, decay_rate=0.1)
mem.save_memory(vector, content="user prefers morning meetings", importance=0.8)
results = mem.retrieve_memory(query_vector, k=5)

# Graph memory with entity relations
gm = GraphMemory(dim=384)
gm.add_entity("prod-db-01", vector, name="Production DB", entity_type="system")
gm.add_entity("api-gateway", vector2, name="API Gateway", entity_type="system")
gm.add_relation("api-gateway", "depends_on", "prod-db-01")
results = gm.search(query_vector, k=3, expand=True, max_hops=2)
```

---

## Implementation

| Component | Language | Location |
|-----------|----------|----------|
| Core engine | Rust | `src/` |
| Python bindings | PyO3/maturin | `src/python.rs` |
| Benchmarks | Rust | `benches/` |
| Papers | Markdown | `papers/` |
| Legacy Python | Python | `bitcache/` |

### Rust Modules

```
src/
├── lib.rs            # Crate root
├── quantize.rs       # Sign-bit binary quantization
├── search.rs         # XOR + POPCOUNT Hamming search (hardware popcount)
├── index.rs          # Flat binary index
├── two_stage.rs      # Binary filter → float rerank
├── three_stage.rs    # Binary → 4-bit → float progressive retrieval
├── partitioned.rs    # Binary k-means routing
├── float_routed.rs   # Float-space semantic routing
├── streaming.rs      # Mutable index (insert/update/delete)
├── memory.rs         # Agent memory (decay, reinforcement, eviction)
├── graph_memory.rs   # Knowledge graph + vector retrieval
└── python.rs         # PyO3 bindings
```

---

## Experiments

Organized by paper:

```bash
# Rust benchmarks (all papers)
cargo run --release --bin benchmark           # Full benchmark suite
cargo run --release --bin paper_validation    # Paper-specific claim validation
cargo run --release --bin quick_metrics       # Quick recall/routing metrics

# Python experiments (legacy, uses bitcache/ Python package)
python experiments/paper1_staged_retrieval/eval_rf_curve.py    # recall-vs-rf curve
python experiments/paper1_staged_retrieval/eval_scale.py       # 50K → 500K → 5M
python experiments/paper1_staged_retrieval/eval_all_dbs.py     # 14-method comparison

python experiments/paper2_semantic_routing/eval_realistic.py   # float routing + baselines

python experiments/paper3_memory_systems/eval_agent_workload.py  # end-to-end agent workload
```

Dependencies for Python experiments:
```bash
pip install faiss-cpu sentence-transformers matplotlib scikit-learn hnswlib annoy
```

---

## Tests

```bash
# Rust (19 tests)
cargo test

# Python (legacy)
pytest tests/ -v
```

---

## Key Findings

1. **Binary scan dominates latency, not reranking.** Reranking 1000 candidates adds only ~1ms over baseline scan cost.

2. **Real semantic embeddings preserve neighborhoods under binary quantization.** 88.9% recall at rf=10 on sentence-transformer embeddings.

3. **Float-space routing achieves 100% partition hit rate on real embeddings.** All true neighbors are in the probed partitions.

4. **Scale boundary: 500K for exhaustive, ~3M for routed.** Beyond that, further partitioning is needed.

5. **Recall ceiling (~89%) is from quantization noise, not candidate coverage or routing.** Higher-bit quantization is the next improvement.

6. **Rust closes the throughput gap.** 10-60x improvement over Python prototype, competitive with FAISS on throughput while maintaining the same algorithmic properties.

---

## Papers

1. **Tunable Staged Retrieval for Persistent AI Memory Systems** — Gen1 architecture
2. **Partition-Local Semantic Retrieval via Float-Space Routing** — Gen3 routing
3. **Bitcache: A Layered Memory Architecture for Autonomous AI Agents** — Full system

---

## Reproducibility

See [`papers/REPRODUCIBILITY.md`](papers/REPRODUCIBILITY.md) for full instructions including dataset generation, benchmark commands, hardware specifications, and validation of all paper claims.

---

## Research Roadmap

| # | Direction | Status |
|---|-----------|--------|
| 1 | Quantized vector retrieval | ✅ Implemented (Rust) |
| 2 | Routed search for low-scan semantic retrieval | ✅ Implemented (Rust) |
| 3 | Agentic AI memory systems | ✅ Implemented (Rust) |
| 4 | Multi-agent shared memory | 🔜 Future work |
| 5 | Adaptive decay from agent feedback | 🔜 Future work |

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
