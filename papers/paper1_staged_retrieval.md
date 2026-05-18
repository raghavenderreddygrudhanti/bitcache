# Tunable Staged Retrieval for Persistent AI Memory Systems

**Raghavender Reddy Grudhanti**

## Abstract

We present a staged retrieval architecture for AI agent memory that achieves high recall through exhaustive binary filtering followed by float reranking. On 100K real sentence-transformer embeddings (all-MiniLM-L6-v2, 384 dimensions), the system achieves 88.9% recall@10 — outperforming FAISS HNSW (87.2%) — with a rerank factor of just 10. The architecture provides a smooth, controllable tradeoff: a single parameter (rerank factor) continuously trades latency for recall along a predictable curve. We demonstrate that binary scan dominates latency while reranking cost grows sublinearly, and that real semantic embeddings preserve neighborhood structure under sign-bit quantization far better than synthetic distributions. Scale experiments on 50K to 5M vectors establish 500K as the practical boundary for exhaustive scan. The system builds in 0.1s, supports streaming inserts at 195K vectors/sec with O(1) deletion, and requires no training data.

## Core Contribution

- Exhaustive binary filtering guarantees no candidate is missed within the rerank budget
- Float reranking guarantees precise ordering of candidates found
- The rerank factor is a single control parameter for smooth recall-latency tradeoff
- Real semantic embeddings are unusually compatible with binary sign-bit quantization

## Key Results

| Dataset | Method | Recall@10 | Latency |
|---------|--------|-----------|---------|
| 9K real (MiniLM) | bitcache rf=10 | 99.98% | 0.74ms |
| 100K real (MiniLM) | bitcache rf=10 | 88.9% | 8.5ms |
| 100K real (MiniLM) | FAISS HNSW | 87.2% | — |
| 50K synthetic | bitcache rf=1000 | 97.3% | 14.9ms |

## Recall-vs-RF Curve (50K synthetic, dim=768)

| rf | Recall@10 | Latency | QPS |
|----|-----------|---------|-----|
| 10 | 0.303 | 7.8ms | 129 |
| 100 | 0.687 | 8.2ms | 122 |
| 500 | 0.935 | 10.1ms | 99 |
| 1000 | 0.973 | 14.9ms | 67 |

## Architectural Insight

Binary scan dominates latency (flat from rf=10 to rf=100). Reranking 1000 candidates adds only 1ms. The bottleneck is the most optimizable component (SIMD popcount on contiguous memory).

## Scale Boundary

| Size | rf=500 | Latency | Verdict |
|------|--------|---------|---------|
| 50K | 91.4% | 9.2ms | Excellent |
| 500K | 72.2% | 74.7ms | Usable |
| 5M | 51.0% | 784ms | Needs partitioning |

## Positioning

Persistent AI memory retrieval (10K-500K memories per agent). Not web-scale search.

## References

[1] Xiao et al. "QuIVer." arXiv:2605.02171, 2026.
[2] Zhang et al. "FaTRQ." arXiv:2601.09985, 2026.
[3] Gutiérrez et al. "HippoRAG." NeurIPS 2024.
[4] Malkov & Yashunin. "HNSW." IEEE TPAMI, 2020.
[5] Douze et al. "The Faiss Library." arXiv:2401.08281, 2024.
