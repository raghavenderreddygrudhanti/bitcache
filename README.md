# Bitcache

**Routed binary vector retrieval engine for AI agent memory. Rust core with Python bindings.**

Bitcache compresses high-dimensional embeddings into compact binary representations and uses intelligent float-space routing to scan only the most relevant vector partitions — achieving high recall at a fraction of the memory and latency cost.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Layer 6: Graph Memory                          │
│  Entity-relation storage + multi-hop traversal  │
├─────────────────────────────────────────────────┤
│  Layer 5: Agent Memory                          │
│  Importance scoring + decay + eviction          │
├─────────────────────────────────────────────────┤
│  Layer 4: Streaming Mutations                   │
│  Insert / update / delete + metadata filter     │
├─────────────────────────────────────────────────┤
│  Layer 3: Float-Space Routing                   │
│  Semantic partition routing (6.2% scan)         │
├─────────────────────────────────────────────────┤
│  Layer 2: Staged Retrieval                      │
│  Binary filter + float rerank                   │
├─────────────────────────────────────────────────┤
│  Layer 1: Binary Quantization                   │
│  Sign-bit encoding (32x compression)            │
└─────────────────────────────────────────────────┘
```

Each layer builds on the previous. A minimal agent uses Layer 2. A full-featured agent uses all six.

---

## Key Results

| Metric | Value |
|--------|-------|
| Recall@10 (99K real embeddings) | 89.2% |
| Partition hit rate (float routing) | 100% |
| Scan volume (P=128, probe=8) | 6.2% |
| Memory compression | 32x vs float32 |
| Latency (routed search, 99K) | 3.0ms |
| Streaming insert | 195K vectors/sec |

---

## Implementation

| Component | Language | Location |
|-----------|----------|----------|
| Core engine | Rust | `src/` |
| Python bindings | PyO3 | `src/python.rs` |
| Benchmarks | Rust | `cargo bench` |
| Papers | Markdown | `papers/` |

### Rust Modules

```
src/
├── lib.rs            # Crate root
├── quantize.rs       # Sign-bit binary quantization
├── search.rs         # XOR + POPCOUNT Hamming search
├── index.rs          # Flat binary index
├── two_stage.rs      # Binary filter → float rerank
├── three_stage.rs    # Binary → 4-bit → float
├── partitioned.rs    # Binary k-means routing
├── float_routed.rs   # Float-space semantic routing
├── streaming.rs      # Mutable index (insert/update/delete)
├── memory.rs         # Agent memory (decay, reinforcement, eviction)
├── graph_memory.rs   # Knowledge graph + vector retrieval
└── python.rs         # PyO3 bindings
```

---

## Getting Started

### From Rust

```rust
use bitcache::{FloatRoutedIndex, TwoStageIndex, AgentMemory};

// Build a routed index
let mut index = FloatRoutedIndex::new(384, 128, 8, 100, 10);
index.build(&vectors);  // flat f32 slice of shape (n, 384)

// Search
let (scores, indices) = index.search(&query, 10);
```

### From Python

```bash
pip install maturin
maturin develop --release
```

```python
import numpy as np
from bitcache import FloatRoutedIndex, AgentMemory

# Build index
index = FloatRoutedIndex(dim=384, n_partitions=128, n_probe=8)
index.build(vectors)  # numpy array (n, 384)

# Search
scores, indices = index.search(query, k=10)

# Agent memory
mem = AgentMemory(dim=384, capacity=10000)
mem.save_memory(vector, "important fact", importance=0.9)
results = mem.retrieve_memory(query, k=5)
```

---

## Research Background

Bitcache is part of a broader research direction around **efficient retrieval and memory systems for AI agents**.

### Primary Paper

1. **Partition-Local Semantic Retrieval via Float-Space Routing**
   — Demonstrates that real embeddings exhibit strong partition locality: 100% of true top-10 neighbors reside in 6.2% of partitions under float k-means routing.

### Supporting Research

2. **Tunable Staged Retrieval for Persistent AI Memory Systems**
   — Validates the two-stage architecture (binary filter + float rerank) achieving 88.9% recall@10 on 99K real embeddings.

3. **Bitcache: A Layered Memory Architecture for Autonomous AI Agents**
   — Presents the full six-layer composable memory system with importance, decay, eviction, and graph reasoning.

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

## Building

```bash
# Rust library + tests
cargo test

# Release build
cargo build --release

# Python wheel
pip install maturin
maturin build --release
```

---

## License

MIT
