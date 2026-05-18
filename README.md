# bitcache

Staged binary retrieval for persistent AI agent memory.

Achieves **88.9% recall@10** on 100K real sentence-transformer embeddings — outperforming FAISS HNSW (87.2%) — using exhaustive binary filtering with float reranking.

## Results

On real semantic embeddings (all-MiniLM-L6-v2, 384 dimensions):

| Scale | bitcache Recall@10 | FAISS HNSW | FAISS Binary |
|-------|-------------------|------------|--------------|
| 9K | 99.98% | 99.2% | 81.1% |
| 100K | 88.9% | 87.2% | 73.5% |

On synthetic clustered data (50K vectors, 768 dimensions):

| rf | Recall@10 | Latency | QPS |
|----|-----------|---------|-----|
| 10 | 30.3% | 7.8ms | 129 |
| 100 | 68.7% | 8.2ms | 122 |
| 500 | 93.5% | 10.1ms | 99 |
| 1000 | 97.3% | 14.9ms | 67 |

## Install

```bash
pip install -e .
```

## Usage

```python
import numpy as np
from bitcache import TwoStageIndex

# Create index
index = TwoStageIndex(dim=384, rerank_factor=100)

# Add vectors
vectors = np.random.randn(10000, 384).astype(np.float32)
index.add(vectors)

# Search
query = np.random.randn(384).astype(np.float32)
scores, indices = index.search(query, k=10)
```

## How it works

1. **Binary quantize**: Each float dimension → 1 bit (sign). 32x compression.
2. **Exhaustive Hamming scan**: XOR + popcount against all binary codes. Selects top rf×k candidates.
3. **Float rerank**: Precise inner product on candidates only. Returns top-k.

The rerank factor (rf) is the single control parameter. Higher rf → higher recall, slightly higher latency.

## Architecture

```
bitcache/
├── quantize.py      # float → binary conversion
├── search.py        # XOR + popcount Hamming distance
├── index.py         # BinaryIndex (flat scan)
├── two_stage.py     # TwoStageIndex (binary filter + float rerank)
├── three_stage.py   # ThreeStageIndex (binary → 4-bit → float)
├── graph.py         # VamanaIndex (graph in binary space)
├── streaming.py     # StreamingIndex (insert/update/delete)
├── memory.py        # AgentMemory (importance, decay, eviction)
└── graph_memory.py  # GraphMemory (entity-relation + multi-hop)
```

## Benchmarks

```bash
# Recall-vs-rf curve (50K synthetic, produces charts)
python benchmarks/eval_rf_curve.py

# Realistic clustered embeddings + FAISS baselines
python benchmarks/eval_realistic.py

# Scale test (50K → 500K → 5M)
python benchmarks/eval_scale.py

# 14-method comparison
python benchmarks/eval_all_dbs.py
```

Requirements for benchmarks: `pip install faiss-cpu sentence-transformers matplotlib`

## Key findings

1. **Binary scan dominates latency, not reranking.** Latency is flat from rf=10 to rf=100. Reranking 1000 candidates adds only 1ms.
2. **Real semantic embeddings work better than synthetic.** Binary sign-bit quantization preserves semantic manifold structure.
3. **Scale boundary: 500K vectors.** Beyond that, partitioning is needed for interactive latency.
4. **Beats HNSW on recall.** Exhaustive scan never misses candidates that graph navigation can miss.

## Limitations

- **Throughput**: 118 QPS (Python) vs 91K QPS (FAISS C++). Implementation, not architecture.
- **O(n) scan**: Linear latency growth. Practical up to 500K for interactive use.
- **Memory**: Full system stores binary + float. 32x compression applies to binary index only.

## References

- [QuIVer: Binary Quantization for ANN](https://arxiv.org/abs/2605.02171) (Xiao et al., 2026)
- [FaTRQ: Tiered Residual Quantization](https://arxiv.org/abs/2601.09985) (Zhang et al., 2026)
- [HippoRAG: Long-Term Memory for LLMs](https://arxiv.org/abs/2405.14831) (NeurIPS 2024)
- [FAISS](https://github.com/facebookresearch/faiss)

## License

MIT
