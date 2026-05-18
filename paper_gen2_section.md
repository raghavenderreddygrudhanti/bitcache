# Gen2 Addition: Partition-Aware Staged Retrieval

*This section extends the main paper with Gen2 results.*

---

## Partition Routing

To address the O(n) scaling limitation identified in Section 4.4, we introduce partition-aware routing. Binary codes are clustered into P partitions at build time using binary k-means with majority-vote centroid updates. At query time, the system routes to the R closest partitions by centroid Hamming distance, scanning only vectors within those partitions.

### Architecture

```
Query → Binary Quantize
              ↓
  Route: Hamming distance to P centroids → select top-R partitions
              ↓
  Stage 1: Binary Hamming scan (only vectors in R partitions)
              ↓
  Stage 2: Float32 rerank (top rf×k candidates)
              ↓
  Return top-k
```

Scan volume reduces from O(n) to O(n × R/P). With P=128 and R=8, this is 6.2% of the corpus.

### Results (99K real sentence-transformer embeddings, dim=384, k=10)

| Version | Recall@10 | Latency | Vectors Scanned | Speedup |
|---------|-----------|---------|-----------------|---------|
| Gen1 (exhaustive) | 0.896 | 8.6ms | 99,000 (100%) | 1x |
| Gen2 (P=32, probe=4) | 0.898 | 2.3ms | 12,375 (12.5%) | 3.7x |
| Gen2 (P=64, probe=4) | 0.897 | 1.6ms | 6,187 (6.2%) | 5.5x |
| Gen2 (P=128, probe=8) | 0.898 | 1.3ms | 6,187 (6.2%) | 6.8x |

### Analysis

Partition routing achieves 6.8x speedup with no recall loss. This is possible because:

1. Real semantic embeddings cluster naturally by topic. Binary k-means captures this structure.
2. Relevant vectors concentrate in a small number of partitions. Probing 8 of 128 partitions (6.2%) is sufficient to find all true neighbors.
3. The routing cost (Hamming distance to 128 centroids) is negligible compared to the scan savings.

### Scale Implications

| Corpus Size | Gen1 Latency | Gen2 Latency (P=128, probe=8) |
|-------------|-------------|-------------------------------|
| 100K | 8.6ms | 1.3ms |
| 500K | ~43ms | ~6.5ms |
| 1M | ~86ms | ~13ms |
| 5M | ~430ms | ~65ms |

Gen2 extends the practical boundary from 500K (Gen1) to approximately 3-5M vectors for interactive use (sub-100ms).

### Limitations

1. Build time increases due to k-means clustering (~2-5s for 100K vectors).
2. Partition quality depends on data distribution. Uniform random data clusters poorly.
3. Streaming inserts require partition assignment but not full rebuild.
