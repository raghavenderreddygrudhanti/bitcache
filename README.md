# bitcache

Binary vector retrieval engine for AI agent memory.

32x memory compression via sign-bit quantization. Search via XOR + POPCOUNT.

Based on: ["QuIVer: Rethinking ANN Graph Topology via Training-Free Binary Quantization"](https://arxiv.org/abs/2605.02171) (Xiao et al., 2026)

## Install

```bash
pip install bitcache
```

## Usage

```python
import numpy as np
from bitcache import BinaryIndex

# Create index
index = BinaryIndex(dim=1536)

# Add vectors (float32 → binary quantization happens internally)
vectors = np.random.randn(100_000, 1536).astype(np.float32)
index.add(vectors)

# Search
query = np.random.randn(1536).astype(np.float32)
distances, indices = index.search(query, k=10)

# Persistence
index.save("./my_index")
loaded = BinaryIndex.load("./my_index")
```

## How it works

1. **Normalize** input vectors to unit length
2. **Quantize** each dimension to 1 bit (sign bit: positive=1, negative=0)
3. **Pack** into uint8 arrays (1536 dims → 192 bytes, down from 6144 bytes)
4. **Search** via Hamming distance: XOR packed arrays, count differing bits

A 1536-dim float32 vector takes 6,144 bytes. Binary quantization reduces it to 192 bytes. That's **32x compression**.

## Benchmarks

100K vectors, 1000 queries, k=10:

| Dim | Memory (fp32) | Memory (binary) | Compression | Build Time |
|-----|---------------|-----------------|-------------|------------|
| 384 | 146 MB | 4.6 MB | 32x | 0.08s |
| 768 | 293 MB | 9.2 MB | 32x | 0.12s |
| 1536 | 586 MB | 18.3 MB | 32x | ~0.2s |

## Roadmap

- [x] Phase 1: Binary quantization + flat search
- [ ] Phase 1b: Graph index in binary space (Vamana-style, per QuIVer)
- [ ] Phase 2: Progressive retrieval (binary coarse → float rerank)
- [ ] Phase 3: Streaming updates
- [ ] Phase 4: Memory prioritization (importance scoring, aging, eviction)
- [ ] Phase 5: Graph memory (entity-relation retrieval)

## License

MIT
