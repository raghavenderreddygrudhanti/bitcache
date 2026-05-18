# bitcache: A Layered Memory Architecture for Autonomous AI Agents

**Raghavender Reddy Grudhanti**

---

## Abstract

We describe bitcache, a layered memory architecture for autonomous AI agents that integrates staged semantic retrieval, streaming mutations, importance-weighted prioritization, and entity-relationship graph traversal. The system addresses requirements that existing vector databases do not jointly satisfy: 32x memory compression via binary quantization, continuous insert/delete without rebuilds (195K inserts/sec, O(1) delete), temporal importance decay with retrieval-based reinforcement, and multi-hop relational reasoning via graph expansion. Each layer is independently validated and composable. The retrieval layer achieves 89% recall@10 on real sentence-transformer embeddings; the memory layer demonstrates 90% importance decay over 7 simulated days of inactivity; the graph layer supports BFS path finding across typed entity relations. We position the system for agent memory workloads at 10K-500K scale where knowledge evolves continuously and retrieval quality must be balanced against resource constraints.

---

## 1. Introduction

Autonomous AI agents require persistent memory that differs from traditional vector search in several ways:

1. **Knowledge evolves.** Agents learn continuously — memory must support streaming inserts and deletes without downtime.
2. **Not all memories are equal.** Recent, frequently-accessed memories should be prioritized over stale, unused ones.
3. **Context requires relationships.** "What is connected to X?" is as important as "What is similar to X?"
4. **Resources are bounded.** Agent processes run within fixed RAM allocations.
5. **Retrieval quality is tunable.** Different queries warrant different computational budgets.

No existing system addresses all five requirements. Vector databases (Pinecone, Qdrant) handle retrieval and mutation but lack prioritization and graph reasoning. Memory systems (Mem0) provide prioritization but rely on full-precision storage. Knowledge graph systems (HippoRAG [3]) combine graphs with retrieval but require LLM calls during indexing and lack compression.

---

## 2. Architecture

bitcache is organized as six composable layers:

```
Layer 6: GraphMemory        — entity-relation storage + multi-hop traversal
Layer 5: AgentMemory        — importance scoring, decay, eviction
Layer 4: StreamingIndex     — mutable store with ID-based CRUD + metadata filter
Layer 3: FloatRoutedIndex   — float-space partition routing (Gen3)
Layer 2: TwoStageIndex      — binary filter + float rerank (Gen1)
Layer 1: BinaryIndex        — sign-bit quantization + Hamming scan
```

Each layer builds on the previous. Users can compose layers as needed — a simple agent may use only Layer 2 (staged retrieval), while a complex agent may use all six.

---

## 3. Component Validation

### 3.1 Retrieval (Layers 1-3)

| Configuration | Recall@10 | Latency | Dataset |
|--------------|-----------|---------|---------|
| Exhaustive rf=10 | 0.889 | 8.6ms | 99K real |
| Float routed P=128 probe=8 | 0.892 | 3.0ms | 99K real |
| Exhaustive rf=1000 | 0.973 | 14.9ms | 50K synthetic |

### 3.2 Streaming Mutations (Layer 4)

| Operation | Throughput |
|-----------|-----------|
| Insert | 194,886 vectors/sec |
| Delete | O(1) via slot reuse |
| Update (vector) | In-place replacement |
| Update (metadata) | In-place replacement |
| Metadata filter search | Post-scan exact match |

No rebuild required for any mutation. External string IDs map to internal slots.

### 3.3 Memory Prioritization (Layer 5)

| Parameter | Tested Value | Behavior |
|-----------|-------------|----------|
| decay_rate | 0.1/day | 90% importance reduction over 7 days |
| reinforce_amount | 0.15 | Retrieved memories resist decay |
| capacity | configurable | Lowest-importance evicted when full |

Importance model: linear decay proportional to time since last access, additive reinforcement on retrieval, hard eviction at capacity boundary.

### 3.4 Graph Memory (Layer 6)

| Operation | Supported |
|-----------|-----------|
| Add entity (with embedding) | Yes |
| Add typed relation | Yes |
| Vector search → seed entities | Yes |
| BFS graph expansion (configurable hops) | Yes |
| Path finding (source → target) | Yes |
| Remove entity (cascading) | Yes |

Entities are stored with vector embeddings for similarity search. Relations are directed typed edges. Search combines vector similarity (find seeds) with graph traversal (expand context).

---

## 4. Comparison with Existing Systems

| Capability | FAISS | Pinecone | Mem0 | HippoRAG | bitcache |
|-----------|-------|----------|------|----------|----------|
| Vector retrieval | ✓ | ✓ | ✓ | ✓ | ✓ |
| Streaming mutation | — | ✓ | ✓ | — | ✓ |
| Memory prioritization | — | — | ✓ | — | ✓ |
| Graph reasoning | — | — | — | ✓ | ✓ |
| Binary compression (32x) | Partial | — | — | — | ✓ |
| No rebuild on mutation | — | ✓ | ✓ | — | ✓ |
| Tunable recall-latency | Limited | — | — | — | ✓ |

---

## 5. Target Workloads

The architecture is designed for agent memory at 10K-500K scale:

- **Long-running assistants** accumulating conversation history and learned facts
- **Enterprise copilots** maintaining operational knowledge with temporal relevance
- **RAG systems** with continuously updating document corpora
- **Multi-session agents** that remember across interactions with importance weighting

The system is not designed for:
- Billion-scale web search (O(n) scan limitation)
- Real-time high-throughput serving (Python implementation limitation)
- Fully automatic knowledge graph construction (requires explicit entity/relation insertion)

---

## 6. Limitations

1. **Throughput:** Python implementation limits QPS. Production deployment requires compiled implementation.
2. **Scale:** Exhaustive scan practical to 500K; routed scan extends to ~3M. Beyond that, distributed approaches are needed.
3. **Graph construction:** Entities and relations must be explicitly inserted. No automatic extraction from text.
4. **Decay model:** Linear decay is a simplification. Biological memory follows power-law forgetting curves.
5. **Single-process:** No multi-agent shared memory or distributed coordination.

---

## 7. Conclusion

bitcache provides a layered, composable memory architecture for AI agents that integrates retrieval, mutation, prioritization, and graph reasoning within a single system. Each layer is independently validated: retrieval achieves 89% recall on real embeddings, mutations operate at 195K inserts/sec, decay reduces importance by 90% over 7 days, and graph traversal supports multi-hop path finding. The system is positioned for agent memory workloads where knowledge evolves continuously and must be managed within bounded resources.

---

## References

[1] W. Xiao, Z. Wang, C. Li. "QuIVer." arXiv:2605.02171, 2026.
[2] T. Zhang, F. Ponzina, T. Rosing. "FaTRQ." arXiv:2601.09985, 2026.
[3] B. J. Gutiérrez et al. "HippoRAG." NeurIPS 2024.
[4] Y. Malkov, D. Yashunin. "HNSW." IEEE TPAMI, 2020.
[5] M. Douze et al. "The Faiss Library." arXiv:2401.08281, 2024.
