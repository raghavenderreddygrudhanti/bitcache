# bitcache — What We Built & What's Next

## What is bitcache?

bitcache is a memory engine for AI agents. It stores vectors (embeddings) in compressed binary format and searches them fast.

Think of it like this:
- Your AI agent reads 10,000 documents
- Each document becomes a vector (list of 1536 numbers)
- Storing all those numbers takes 60 MB of RAM
- bitcache compresses them to 1.8 MB (32x smaller)
- When the agent asks "find me similar documents", bitcache searches in milliseconds

---

## What We Built So Far (Phase 1)

### Layer 1: Binary Quantization (`bitcache/quantize.py`)

**What it does:** Converts float vectors into binary (0s and 1s).

**How:** If a number is positive → 1. If negative → 0.

```
Original:  [0.23, -0.82, 0.11, -0.45, 0.67, -0.12, 0.89, -0.33]
Binary:    [1,     0,     1,     0,     1,     0,     1,     0]
Packed:    10101010 (1 byte instead of 32 bytes)
```

**Compression:** 1536 floats × 4 bytes = 6,144 bytes → 1536 bits = 192 bytes. **32x smaller.**

### Layer 2: Flat Search (`bitcache/search.py`)

**What it does:** Finds similar vectors using XOR + POPCOUNT.

**How:**
```
Query:    10101010
Vector 1: 10101010  → XOR = 00000000 → popcount = 0 (identical!)
Vector 2: 11001100  → XOR = 01100110 → popcount = 4 (4 bits different)
Vector 3: 01010101  → XOR = 11111111 → popcount = 8 (completely opposite)
```

Lower popcount = more similar.

**Problem:** Scans ALL vectors. Slow at 100K+ vectors. Low recall (6%) because binary is lossy.

### Layer 3: Graph Index (`bitcache/graph.py`)

**What it does:** Builds a navigation graph so search only visits ~100 vectors instead of all 100K.

**How:** Like a social network — each vector is connected to ~32 neighbors. To find something, you hop from neighbor to neighbor, getting closer each hop.

```
Start at random vector
  → check its 32 neighbors (which is closest to query?)
  → jump to closest neighbor
  → check ITS 32 neighbors
  → jump again
  → ... after 5-10 hops, you're in the right neighborhood
  → rerank final candidates with full precision
```

**Result:** 30%+ recall (vs 6% flat), and much faster because only ~100-200 distance computations instead of 100K.

---

## How to Run

### Install

```bash
cd /Users/Raghavender/lang-chain/bitcache
pip install -e .
```

### Run Tests

```bash
pytest tests/ -v
```

You should see 17 tests pass (11 for flat index, 6 for graph index).

### Run Benchmark

```bash
python benchmarks/bench_recall_memory.py
```

This tests flat binary search. Shows compression ratio and recall.

### Try It in Python

```python
import numpy as np
from bitcache import BinaryIndex, VamanaIndex

# === Flat Index (simple, fast to build, low recall) ===
index = BinaryIndex(dim=128)
vectors = np.random.randn(1000, 128).astype(np.float32)
index.add(vectors)

query = vectors[0]
distances, indices = index.search(query, k=5)
print(f"Flat search - nearest indices: {indices}")

# === Graph Index (slower to build, high recall) ===
graph = VamanaIndex(dim=128, R=32, L_build=50)
graph.build(vectors)

scores, indices = graph.search(query, k=5, ef=50)
print(f"Graph search - nearest indices: {indices}")
print(f"Graph search - scores: {scores}")
```

---

## What's Next (Phases 2-5)

### Phase 2: Progressive Retrieval

**Problem:** Binary search is fast but imprecise. Float search is precise but slow.

**Solution:** Do both — binary first (cheap filter), then float on top candidates only.

```
100K vectors
  → binary scan: find top 1000 candidates (fast, cheap)
  → float rerank: score only those 1000 with full precision (accurate)
  → return top 10
```

This gives you the speed of binary + the accuracy of float.

**What to build:** A `TwoStageIndex` that combines `BinaryIndex` (stage 1) with float reranking (stage 2).

---

### Phase 3: Streaming Updates

**Problem:** Current index requires rebuilding the graph when you add new vectors.

**Solution:** Support live inserts and deletes without full rebuild.

```
Agent learns something new → insert into index immediately
Agent forgets something → delete from index
No rebuild needed
```

**What to build:** Add `insert()` and `delete()` methods to `VamanaIndex` that update the graph incrementally.

---

### Phase 4: Memory Prioritization

**Problem:** Agent remembers everything equally. But some memories are more important.

**Solution:** Score each memory by importance. Decay old unused memories. Promote frequently accessed ones.

```
Memory: "User's name is Raghavender" → importance: HIGH (used often)
Memory: "Weather was sunny on Jan 3" → importance: LOW (never used again)

After 30 days without access → compress or evict low-importance memories
```

**What to build:** Add `importance_score`, `access_count`, `last_accessed` metadata to each vector. Add `reinforce()` and `decay()` methods.

---

### Phase 5: Graph Memory

**Problem:** Vectors find similar things. But agents need to find RELATED things.

**Solution:** Store entity-relationship triples alongside vectors.

```
Vector search: "MuleSoft" → finds documents mentioning MuleSoft
Graph search:  "MuleSoft" → integrates_with → Oracle
                          → owned_by → Salesforce
                          → competes_with → Boomi
```

**What to build:** Add a knowledge graph layer that stores (entity, relation, entity) triples. Combine vector similarity + graph traversal for multi-hop reasoning.

---

## Summary Table

| Phase | What | Status | Recall | Speed |
|-------|------|--------|--------|-------|
| 1a | Binary flat search | ✅ Done | 6% | Fast build, slow search |
| 1b | Graph index (Vamana) | ✅ Done | 30%+ | Slow build, fast search |
| 2 | Progressive retrieval | Next | 80%+ | Fast |
| 3 | Streaming updates | Later | Same | Live inserts |
| 4 | Memory prioritization | Later | Same | Smart eviction |
| 5 | Graph memory | Later | Multi-hop | Reasoning |

---

## File Structure

```
bitcache/
├── bitcache/
│   ├── __init__.py      # Exports BinaryIndex, VamanaIndex
│   ├── quantize.py      # Float → binary conversion
│   ├── search.py        # XOR + POPCOUNT distance
│   ├── index.py         # BinaryIndex (flat search)
│   └── graph.py         # VamanaIndex (graph search)
├── tests/
│   ├── test_bitcache.py # 11 tests for flat index
│   └── test_graph.py    # 6 tests for graph index
├── benchmarks/
│   └── bench_recall_memory.py  # Performance benchmark
├── pyproject.toml       # Package config
└── README.md            # Public docs
```

---

## Key Numbers to Remember

| Metric | Value |
|--------|-------|
| Compression | 32x (float32 → binary) |
| Memory for 1M vectors (d=1536) | 192 MB (vs 6 GB float32) |
| Memory for 10M vectors (d=1536) | 1.9 GB (vs 60 GB float32) |
| Flat search recall | ~6% |
| Graph search recall | 30%+ (tunable with ef) |
| Target recall (after Phase 2) | 80%+ |

---

## GitHub

https://github.com/raghavenderreddygrudhanti/bitcache
