# bitcache: Architectural Analysis and Improvement Proposals

**Author: Raghavender Reddy Grudhanti**

---

## 1. Why Two-Stage Architecture Improves Recall

The recall jump from 6.2% (flat binary) to 57.5% (two-stage) is explained by the information recovery mechanism:

**Binary quantization is a lossy projection.** Sign-bit encoding maps each float coordinate to 1 bit. Two vectors that are close in float space may differ on coordinates near zero — producing high Hamming distance despite high cosine similarity. This is the source of the 6% recall floor.

**The two-stage approach exploits a key property: binary quantization preserves neighborhood structure at coarse granularity.** While the exact top-10 may not appear in the binary top-10, they almost always appear in the binary top-1000. The true nearest neighbors are rarely far from the query in Hamming space — they are just not precisely ranked.

Formally: let N_k(q) be the true top-k neighbors and B_m(q) be the binary top-m candidates. The probability P(N_k ⊂ B_m) increases rapidly with m. At rf=100 (m=1000), this probability reaches ~0.58 for k=10 on our clustered data. The float rerank then perfectly orders the candidates within B_m.

**Why other methods fail to achieve this:**
- FAISS Binary Flat: returns binary top-k directly. No reranking. Stuck at 6%.
- FAISS HNSW: navigates a graph in float space but with inner product metric on clustered data, the graph structure doesn't bridge clusters efficiently.
- FAISS IVF: partitions into 100 cells, probes 10. Misses vectors in unprobed cells.

**bitcache's insight: scan everything cheaply (binary), then score precisely (float).** The binary scan is exhaustive — it never misses a candidate. The float rerank is precise — it never misjudges a candidate. The combination is more robust than any single-pass approximate method.

---

## 2. Proposed 3-Stage Retrieval Architecture

```
Stage 1: Binary Filter (1-bit per dim)
  Input:  100K vectors
  Output: top-2000 candidates
  Cost:   O(n) Hamming distance, ~96 bytes/vector
  
Stage 2: Compressed Rerank (4-bit per dim)
  Input:  2000 candidates
  Output: top-100 candidates
  Cost:   O(m) quantized inner product, ~384 bytes/vector
  
Stage 3: Full Precision Rerank (32-bit per dim)
  Input:  100 candidates
  Output: top-10 results
  Cost:   O(k) float32 inner product, ~3072 bytes/vector
```

**Memory hierarchy mapping:**

| Stage | Precision | Bytes/vector (d=768) | Storage tier | Vectors scored |
|-------|-----------|---------------------|--------------|----------------|
| 1 | 1-bit | 96 | L2 cache / RAM | All (100K) |
| 2 | 4-bit | 384 | RAM | 2000 |
| 3 | 32-bit | 3072 | RAM / SSD | 100 |

**Expected recall improvement:** Stage 2 (4-bit quantized scoring) provides much finer distance estimation than binary. TurboQuant [ref] achieves MSE of 0.009 at 4-bit — nearly lossless. This intermediate stage would push recall from 57.5% toward 85-90% without loading all float vectors.

**Total memory for 1M vectors:**
- Stage 1 codes: 1M × 96 bytes = 91.6 MB (always in RAM)
- Stage 2 codes: 1M × 384 bytes = 366 MB (in RAM or mmap)
- Stage 3 vectors: 1M × 3072 bytes = 2.93 GB (on SSD, loaded on demand)

Compared to full float32: 2.93 GB all in RAM. The 3-stage approach keeps only 91.6 MB hot.

---

## 3. Adaptive Candidate Refinement

Fixed rf=100 is suboptimal. Some queries are easy (true neighbors are very close in binary space), others are hard (true neighbors are scattered).

**Proposed strategies:**

### 3.1 Score-Gap Adaptive

Monitor the Hamming distance gap between the k-th and (k+1)-th candidate in Stage 1. If the gap is large, the top-k is confident — use small rf. If the gap is small (many candidates at similar distance), increase rf.

```
gap = hamming_dist[k] - hamming_dist[k-1]
if gap > threshold_high:
    rf = 10   # confident
elif gap < threshold_low:
    rf = 200  # uncertain
else:
    rf = 50   # moderate
```

### 3.2 Recall-Target Adaptive

Set a target recall (e.g., 0.9) and iteratively increase rf until the reranked scores stabilize. If the top-k scores after reranking don't change when rf doubles, stop.

### 3.3 Query-Difficulty Estimation

Precompute the average Hamming distance from each query to its binary top-100. Queries with low average distance (clustered near many vectors) need higher rf. Queries with high average distance (isolated) need lower rf.

### 3.4 Budget-Constrained

Given a latency budget (e.g., 10ms), allocate rf based on the time remaining after Stage 1. Faster Stage 1 → more budget for Stage 2 → higher rf.

---

## 4. Proposed Benchmark Metrics

### 4.1 Current Metrics (Already Measured)
- Recall@k
- QPS (queries per second)
- Build time
- Compression ratio

### 4.2 Metrics to Add

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| **Memory footprint per stage** | Bytes used by each stage's data structures | Validates the memory hierarchy claim |
| **Rerank cost ratio** | Time in Stage 2 / total search time | Shows where latency is spent |
| **Latency percentiles** | p50, p95, p99 search latency | Production systems care about tail latency |
| **Recall vs rf curve** | Recall@10 as a function of rerank_factor | Shows diminishing returns, guides rf selection |
| **Cache hit ratio** | Fraction of Stage 2 candidates that appear in final top-k | Measures Stage 1 filtering quality |
| **Recall vs memory** | Recall@10 plotted against total memory used | The fundamental tradeoff curve |
| **Insert throughput under load** | Inserts/sec while concurrent searches run | Streaming workload realism |
| **Decay effectiveness** | Recall on "important" vs "decayed" memories | Validates prioritization |
| **Graph expansion utility** | Additional relevant entities found via graph vs vector-only | Measures Phase 5 value |

### 4.3 Benchmark Methodology Improvements

1. **Multiple datasets:** Clustered (current), uniform random, real embeddings (GloVe, OpenAI), skewed (Zipf popularity distribution).
2. **Scale sweep:** 10K, 100K, 1M, 10M vectors — report how metrics change with scale.
3. **Dimension sweep:** 128, 384, 768, 1536, 3072 — validate dimension-independence of compression.
4. **Workload patterns:** Read-heavy (99% search), write-heavy (50% insert), mixed (search + insert + delete concurrent).

---

## 5. Conceptual Comparison Against Existing Architectures

### 5.1 bitcache vs HNSW

| Aspect | HNSW | bitcache |
|--------|------|----------|
| Graph construction | Float metric space | Binary Hamming space |
| Memory during search | Full vectors + graph edges | Binary codes (32x smaller) |
| Recall mechanism | Graph navigation | Exhaustive binary scan + rerank |
| Build time | O(n log n), 20-200s | O(n), 0.1s |
| Mutation support | Expensive (graph repair) | O(1) insert/delete |
| Failure mode | Gets stuck in wrong graph region | Misses candidates beyond rf |

**bitcache advantage:** Exhaustive scan guarantees no candidate is missed (within rf). HNSW can get trapped in local optima during navigation.

**HNSW advantage:** Sublinear search time. bitcache Stage 1 is O(n).

### 5.2 bitcache vs IVF

| Aspect | IVF | bitcache |
|--------|-----|----------|
| Partitioning | k-means clusters | None (exhaustive) |
| Recall loss source | Unprobed partitions | Quantization noise |
| Training required | Yes (k-means) | No |
| Streaming support | Requires rebalancing | Native |

**bitcache advantage:** No training, no partition assignment errors, no rebalancing.

**IVF advantage:** Sublinear search with nprobe << nlist.

### 5.3 bitcache vs FaTRQ

| Aspect | FaTRQ | bitcache |
|--------|-------|----------|
| Refinement stages | Tiered residuals from far memory | Binary → float from RAM |
| Hardware | Custom CXL accelerator | Standard CPU |
| Early stopping | Provable bounds | Fixed rf (adaptive proposed) |
| Memory hierarchy | DRAM → CXL → SSD | L2 → RAM (current) |

**bitcache advantage:** Runs on commodity hardware. No custom accelerator.

**FaTRQ advantage:** Provable early stopping, hardware-accelerated refinement.

### 5.4 bitcache vs DiskANN

| Aspect | DiskANN | bitcache |
|--------|---------|----------|
| Graph construction | Vamana in float space | Vamana in binary space |
| Disk usage | Full vectors on SSD, graph in RAM | Binary codes in RAM, float on demand |
| Search | Graph navigation + SSD reads | Binary scan + RAM rerank |
| Scale | Billion-point | Million-point (current) |

**bitcache advantage:** No SSD reads during search (binary codes fit in RAM). Faster build.

**DiskANN advantage:** Proven at billion scale. Optimized Rust/C++ implementation.

---

## 6. Enterprise AI Use Cases

### 6.1 Agent Memory (Primary Use Case)

Long-running AI agents accumulate knowledge across sessions. bitcache provides:
- **Compressed storage:** 10M memories at d=768 fit in 916 MB (binary) vs 29 GB (float32)
- **Instant retrieval:** No cold-start rebuild when agent restarts
- **Memory lifecycle:** Importance decay removes stale knowledge automatically
- **Relationship context:** Graph memory expands retrieved context via entity relations

**Example:** Customer support agent remembers past interactions, prioritizes recent/frequent topics, forgets resolved issues.

### 6.2 Long-Term Conversation Memory

Multi-turn conversations generate hundreds of embeddings per session. bitcache enables:
- **Streaming inserts:** Each message embedded and indexed in real-time (195K/sec)
- **Session isolation:** Metadata filtering by session_id, user_id, tenant_id
- **Temporal relevance:** Decay ensures recent context is prioritized over old

**Example:** Enterprise chatbot maintains per-user conversation history across months, retrieving relevant past exchanges without loading entire history.

### 6.3 Streaming Enterprise Retrieval

Document corpora that change continuously (news feeds, support tickets, code repositories):
- **No rebuild:** New documents indexed immediately without stopping search
- **Delete propagation:** Removed documents disappear from results instantly
- **Metadata filtering:** Route queries to specific document subsets (department, access level)

**Example:** Internal knowledge base where policies update weekly, new procedures are added daily, and deprecated docs are removed — all without index downtime.

### 6.4 Operational AI Copilots

AI assistants embedded in operational workflows (DevOps, SRE, data engineering):
- **Graph memory:** Store system topology (service A → depends_on → service B)
- **Incident memory:** Past incidents with resolution steps, decayed by staleness
- **Multi-hop reasoning:** "What services are affected if database X goes down?" → traverse dependency graph

**Example:** SRE copilot that remembers past outages, knows system dependencies, and retrieves relevant runbooks based on current symptoms.

---

## 7. Research Directions (No Code Changes Required)

### 7.1 Theoretical Analysis

- **Recall bounds:** Derive P(N_k ⊂ B_m) as a function of m, k, d, and data distribution. This would provide theoretical justification for rf selection.
- **Optimal bit allocation:** Given a memory budget, what's the optimal split between Stage 1 (1-bit), Stage 2 (4-bit), and Stage 3 (32-bit) storage?
- **Decay convergence:** Prove that the linear decay + reinforcement model converges to a stable importance distribution under stationary query patterns.

### 7.2 Empirical Studies

- **Real embedding evaluation:** Run on OpenAI text-embedding-3-small, Cohere embed-v3, BGE-base outputs.
- **Scale study:** How does recall degrade as n grows from 100K to 10M at fixed rf?
- **Distribution sensitivity:** Compare recall on uniform, clustered, power-law, and real-world distributions.
- **Ablation study:** Contribution of each stage (binary alone, binary+4bit, binary+4bit+float).

### 7.3 System Design Papers

- **Memory-aware retrieval:** Formalize the 3-stage architecture as a memory-hierarchy-aware retrieval system. Compare against FaTRQ's CXL approach using commodity hardware.
- **Agent memory lifecycle:** Formalize the decay/reinforce/evict model as a Markov chain. Analyze steady-state behavior under different query patterns.
- **Graph-augmented retrieval:** Measure the marginal value of graph expansion (hop 1, hop 2, hop 3) on multi-hop QA benchmarks.

### 7.4 Integration Studies

- **Agno integration:** Implement bitcache as an Agno VectorDb backend. Benchmark against pgvector, Qdrant, Milvus on agent workloads.
- **LangChain integration:** Provide a LangChain VectorStore wrapper. Compare retrieval quality in RAG pipelines.
- **Streaming workload simulation:** Model real agent behavior (bursty inserts, periodic searches, gradual decay) and measure system behavior over simulated weeks.

---

## 8. Positioning Statement

bitcache is not another vector database. It is a **staged memory retrieval architecture** designed for AI agent systems where:

1. Memory grows continuously (streaming inserts)
2. Not all memories are equally important (prioritization)
3. Context requires relationships, not just similarity (graph)
4. RAM is constrained (32x compression)
5. Rebuild downtime is unacceptable (instant mutations)

The two-stage progressive retrieval is the core technical contribution: it demonstrates that exhaustive binary scanning followed by precise reranking outperforms sophisticated graph-based methods on recall while using 32x less memory. This positions bitcache as the retrieval layer for the next generation of persistent, memory-efficient AI agent systems.

---

## References

[1] Xiao et al. "QuIVer: Rethinking ANN Graph Topology via Training-Free Binary Quantization." arXiv:2605.02171, 2026.
[2] Zhang et al. "FaTRQ: Tiered Residual Quantization for LLM Vector Search." arXiv:2601.09985, 2026.
[3] Gutiérrez et al. "HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs." NeurIPS 2024.
[4] Subramanya et al. "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search." NeurIPS 2019.
[5] Malkov & Yashunin. "Efficient and Robust Approximate Nearest Neighbor Search Using HNSW." IEEE TPAMI, 2020.
[6] Douze et al. "The Faiss Library." arXiv:2401.08281, 2024.
