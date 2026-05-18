# bitcache

Staged semantic retrieval architecture for persistent AI agent memory.

![Architecture](papers/figures/paper3_architecture.png)

## Key Results

**Real sentence-transformer embeddings (99K vectors, all-MiniLM-L6-v2, 384d):**

| Method | Recall@10 | Latency | Scan% | Speedup |
|--------|-----------|---------|-------|---------|
| Gen1 (exhaustive) | 0.891 | 8.6ms | 100% | 1x |
| **Gen3 (float routed)** | **0.892** | **3.0ms** | **6.2%** | **3.8x** |
| FAISS HNSW (M=32, ef=64) | 0.880 | 0.01ms | — | — |
| FAISS Binary (no rerank) | 0.735 | — | — | — |

**Recall-vs-rf tradeoff (50K synthetic, 768d):**

![Recall Curve](papers/figures/paper1_recall_vs_rf.png)

## Install

```bash
git clone https://github.com/raghavenderreddygrudhanti/bitcache.git
cd bitcache
pip install -e .
```

## Quick Start

```python
import numpy as np
from bitcache import TwoStageIndex

# Staged retrieval (Gen1)
index = TwoStageIndex(dim=384, rerank_factor=100)
vectors = np.random.randn(10000, 384).astype(np.float32)
index.add(vectors)

query = vectors[0]
scores, indices = index.search(query, k=10)
```

```python
from bitcache.float_routed import FloatRoutedIndex

# Partition-routed retrieval (Gen3)
index = FloatRoutedIndex(dim=384, n_partitions=128, n_probe=8, rerank_factor=500)
index.build(vectors)

scores, indices = index.search(query, k=10)
```

```python
from bitcache import AgentMemory

# Agent memory with decay and eviction
mem = AgentMemory(dim=384, capacity=10000, decay_rate=0.1)
mem.save_memory(vector, content="user prefers morning meetings", importance=0.8)
results = mem.retrieve_memory(query_vector, k=5)
```

```python
from bitcache import GraphMemory

# Graph memory with entity relations
gm = GraphMemory(dim=384)
gm.add_entity("prod-db-01", vector, name="Production DB", entity_type="system")
gm.add_entity("api-gateway", vector2, name="API Gateway", entity_type="system")
gm.add_relation("api-gateway", "depends_on", "prod-db-01")

results = gm.search(query_vector, k=3, expand=True, max_hops=2)
```

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

## Experiments

Organized by paper:

```bash
# Paper 1: Staged Retrieval
python experiments/paper1_staged_retrieval/eval_rf_curve.py    # recall-vs-rf curve
python experiments/paper1_staged_retrieval/eval_scale.py       # 50K → 500K → 5M
python experiments/paper1_staged_retrieval/eval_all_dbs.py     # 14-method comparison

# Paper 2: Semantic Routing
python experiments/paper2_semantic_routing/eval_realistic.py   # float routing + baselines

# Paper 3: Memory Systems
python experiments/paper3_memory_systems/eval_agent_workload.py  # end-to-end agent workload
```

Dependencies:
```bash
pip install faiss-cpu sentence-transformers matplotlib scikit-learn hnswlib annoy
```

## Tests

```bash
pytest tests/ -v
# 75 tests passing
```

## Key Findings

1. **Binary scan dominates latency, not reranking.** Reranking 1000 candidates adds only 1ms over baseline scan cost.
2. **Real semantic embeddings preserve neighborhoods under binary quantization.** 88.9% recall at rf=10 on sentence-transformer embeddings.
3. **Float-space routing achieves 100% partition hit rate** on real embeddings. All true neighbors are in the probed partitions.
4. **Scale boundary: 500K for exhaustive, ~3M for routed.** Beyond that, further partitioning is needed.
5. **Recall ceiling (~89%) is from quantization noise**, not candidate coverage or routing. Higher-bit quantization is the next improvement.

## Papers

- [Paper 1: Tunable Staged Retrieval](papers/paper1_staged_retrieval.md) — Gen1 architecture
- [Paper 2: Partition-Local Semantic Routing](papers/paper2_semantic_routing.md) — Gen3 routing
- [Paper 3: Persistent Memory Architecture](papers/paper3_memory_systems.md) — Full system

## Reproducibility

See [papers/REPRODUCIBILITY.md](papers/REPRODUCIBILITY.md) for full instructions including dataset generation, benchmark commands, and hardware specifications.

## References

- [QuIVer: Binary Quantization for ANN](https://arxiv.org/abs/2605.02171) (Xiao et al., 2026)
- [FaTRQ: Tiered Residual Quantization](https://arxiv.org/abs/2601.09985) (Zhang et al., 2026)
- [HippoRAG: Long-Term Memory for LLMs](https://arxiv.org/abs/2405.14831) (NeurIPS 2024)
- [HNSW](https://arxiv.org/abs/1603.09320) (Malkov & Yashunin, 2020)
- [FAISS](https://github.com/facebookresearch/faiss)

## License

MIT
