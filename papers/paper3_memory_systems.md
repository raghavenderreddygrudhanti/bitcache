# bitcache: A Layered Memory Architecture for Autonomous AI Agents

**Raghavender Reddy Grudhanti**

---

## Abstract

We present bitcache, a composable persistent memory architecture for autonomous AI agents that integrates retrieval, mutation, prioritization, and relational reasoning under bounded resources. The system answers a central question: how should AI agents manage long-term memory when knowledge evolves continuously, not all memories are equally important, and context requires both similarity and relationships? We validate the architecture through an end-to-end enterprise operations copilot workload demonstrating: semantic retrieval (0.55ms), graph expansion (<0.01ms), importance reinforcement on access, temporal decay (90% importance reduction over 5 days), and capacity-based eviction. Each layer is independently composable: agents may use staged retrieval alone, or combine all six layers for full memory lifecycle management.

---

## 1. Introduction

Autonomous AI agents require persistent memory that differs fundamentally from vector search:

| Requirement | Vector Database | Agent Memory |
|-------------|----------------|--------------|
| Knowledge lifecycle | Static corpus | Continuous insert/delete |
| Importance | All equal | Temporal decay + reinforcement |
| Context | Similarity only | Similarity + relationships |
| Resources | Elastic | Bounded per agent |
| Retrieval quality | Fixed | Tunable per query |

No existing system addresses all five requirements jointly:

- **Vector databases** (Pinecone, Qdrant, FAISS) handle retrieval and mutation but lack temporal memory semantics — no concept of importance, decay, or eviction.
- **ANN systems** (HNSW, IVF, Annoy) optimize search speed but provide no mutation support, no prioritization, and no relational reasoning.
- **Agent memory tools** (Mem0) provide prioritization but rely on full-precision storage without compressed retrieval pipelines.
- **Graph systems** (HippoRAG [3]) combine graphs with retrieval but require LLM calls during indexing and lack mutable compressed retrieval.

We propose a layered architecture where each layer adds a specific capability, and agents compose only the layers they need.

---

## 2. Architecture

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
│  Layer 3: Partition Routing                     │
│  Float-space semantic routing (6.2% scan)       │
├─────────────────────────────────────────────────┤
│  Layer 2: Staged Retrieval                      │
│  Binary filter + float rerank                   │
├─────────────────────────────────────────────────┤
│  Layer 1: Binary Quantization                   │
│  Sign-bit encoding (32x compression)            │
└─────────────────────────────────────────────────┘
```

Each layer builds on the previous. A minimal agent uses Layer 2 (staged retrieval). A full-featured agent uses all six.

---

## 3. Memory Semantics

bitcache defines five memory operations that distinguish agent memory from vector search:

### 3.1 Recency

Each memory carries a timestamp. Retrieval can filter by time window. Newer memories are implicitly preferred through the decay mechanism.

### 3.2 Reinforcement

When a memory is retrieved and used by the agent, its importance score increases:

```
importance += reinforce_amount  (capped at 1.0)
```

Frequently useful memories become resistant to decay and eviction.

### 3.3 Decay

Unused memories lose importance over time:

```
importance -= decay_rate × days_since_last_access
```

This models forgetting: knowledge that is never accessed gradually fades.

### 3.4 Importance

Each memory has a score in [0, 1] reflecting its current value to the agent. Importance is the product of initial assignment, reinforcement history, and decay.

### 3.5 Eviction

When memory capacity is reached, the lowest-importance memory is removed. This enforces bounded resource usage without manual cleanup.

---

## 4. End-to-End Validation: Enterprise Operations Copilot

We validate the full architecture through a realistic workload: an AI copilot supporting telecom operations that receives incidents, runbooks, and system notes.

### 4.1 Workload

| Step | Operation | Description |
|------|-----------|-------------|
| 1 | Insert | 10 operational events (incidents, runbooks, notes) |
| 2 | Retrieve | Semantic search for "database connection issues" |
| 3 | Graph expand | Find systems connected to the affected component |
| 4 | Reinforce | Retrieved memories gain importance |
| 5 | Decay | Simulate 5 days of inactivity |
| 6 | Evict | Enforce capacity=5, remove lowest importance |

### 4.2 Results

| Step | Latency | Outcome |
|------|---------|---------|
| Insert 10 memories | 2.3s (includes embedding) | All stored with importance scores |
| Semantic retrieval (k=3) | 0.55ms | Found: incident, runbook, and note about prod-db-01 |
| Graph expansion | <0.01ms | Found: api-gateway depends_on prod-db-01, app-server connects_to prod-db-01 |
| Reinforcement | automatic | Retrieved memories: importance 0.4 → 0.55, 0.9 → 1.0 |
| Decay (5 days) | automatic | Mean importance: 0.52 → 0.11 (79% reduction) |
| Eviction (capacity=5) | automatic | 5 lowest-importance memories removed |

### 4.3 Timeline Example

| Day | Event | Memory Importance |
|-----|-------|-------------------|
| 0 | Incident inserted | 0.90 |
| 0 | Retrieved by operator query | 1.00 (reinforced) |
| 2 | No access | 0.80 (decayed) |
| 5 | No access | 0.50 (decayed) |
| 5 | Retrieved again | 0.65 (reinforced) |
| 10 | No access | 0.15 (decayed) |
| 10 | Capacity reached | Evicted |

---

## 5. Per-Layer Complexity

| Layer | Time Complexity | Space Complexity | Purpose |
|-------|----------------|------------------|---------|
| Binary Quantization | O(n × d) build | O(n × d/8) | 32x compression |
| Staged Retrieval | O(n × d/8 + rf×k×d) query | O(n × d) float storage | Recall recovery |
| Partition Routing | O(P×d) routing + O(n/P × d/8) scan | O(P×d) centroids | Sublinear search |
| Streaming Mutations | O(1) insert/delete | O(n) slot array | Live updates |
| Memory Prioritization | O(n) decay sweep | O(n) importance scores | Temporal relevance |
| Graph Memory | O(V+E) BFS | O(V+E) adjacency | Relational context |

---

## 6. Comparison with Existing Systems

| Capability | FAISS | Pinecone | Mem0 | HippoRAG | bitcache |
|-----------|-------|----------|------|----------|----------|
| Semantic retrieval | ✓ | ✓ | ✓ | ✓ | ✓ |
| Streaming mutation | — | ✓ | ✓ | — | ✓ |
| Temporal decay | — | — | ✓ | — | ✓ |
| Reinforcement | — | — | ✓ | — | ✓ |
| Capacity eviction | — | — | Partial | — | ✓ |
| Graph reasoning | — | — | — | ✓ | ✓ |
| Binary compression | Partial | — | — | — | ✓ (32x) |
| Tunable recall | Limited | — | — | — | ✓ |
| No rebuild on mutation | — | ✓ | ✓ | — | ✓ |

---

## 7. Limitations and Threats to Validity

1. **Simulated workload.** The end-to-end validation uses simulated operational events, not a live LLM agent. Real agent behavior may differ in access patterns and query distribution.
2. **Linear decay model.** Biological memory follows power-law forgetting curves (Ebbinghaus). Our linear model is a simplification that may not match optimal agent behavior.
3. **Manual graph construction.** Entities and relations must be explicitly inserted. No automatic extraction from text is provided.
4. **Single-process.** No multi-agent shared memory or distributed coordination.
5. **Python throughput.** Insert latency is dominated by embedding computation (2.3s for 10 events). In production, embedding would be pre-computed or batched.
6. **No live evaluation.** We have not measured downstream task performance (e.g., answer quality improvement from memory retrieval).

---

## 8. Future Research Directions

1. **Automatic entity extraction**: Use LLMs to extract entities and relations from inserted text, populating the graph layer automatically.
2. **Multi-agent shared memory**: Distributed memory with access control, enabling agent teams to share knowledge.
3. **Adaptive decay**: Learn decay rates from agent behavior — memories that are consistently useful should decay slower.
4. **Reinforcement-driven ranking**: Use retrieval feedback (was the memory actually useful?) to adjust importance beyond simple access counting.
5. **Live agent evaluation**: Measure downstream task performance (QA accuracy, task completion rate) with and without the memory system.

---

## 9. Conclusion

bitcache provides a composable persistent memory architecture for AI agents that integrates retrieval, mutation, prioritization, and relational reasoning. The central contribution is not any single component, but the layered design that allows agents to compose memory capabilities as needed — from simple staged retrieval (Layer 2) to full lifecycle management with graph reasoning (all six layers).

The end-to-end validation demonstrates that the architecture supports realistic agent workloads: semantic retrieval in 0.55ms, graph expansion in <0.01ms, automatic reinforcement and decay, and capacity-based eviction. Each layer is independently validated and composable.

The system is positioned for agent memory workloads at 10K-500K scale where knowledge evolves continuously and must be managed within bounded resources — not as a replacement for web-scale vector databases, but as the memory infrastructure layer between the agent and its knowledge.

Code: https://github.com/raghavenderreddygrudhanti/bitcache

---

## References

[1] W. Xiao, Z. Wang, C. Li. "QuIVer: Rethinking ANN Graph Topology via Training-Free Binary Quantization." arXiv:2605.02171, 2026.

[2] T. Zhang, F. Ponzina, T. Rosing. "FaTRQ: Tiered Residual Quantization for LLM Vector Search in Far-Memory-Aware ANNS Systems." arXiv:2601.09985, 2026.

[3] B. J. Gutiérrez, Y. Shu, Y. Gu, M. Yasunaga, Y. Su. "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models." NeurIPS 2024.

[4] Y. Malkov, D. Yashunin. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs." IEEE TPAMI, 2020.

[5] M. Douze et al. "The Faiss Library." arXiv:2401.08281, 2024.
