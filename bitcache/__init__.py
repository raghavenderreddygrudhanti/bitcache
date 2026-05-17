"""bitcache — Binary vector retrieval engine for AI agent memory.

Based on: "QuIVer: Rethinking ANN Graph Topology via Training-Free
Binary Quantization" (Xiao et al., arXiv:2605.02171, 2026)
"""

from bitcache.index import BinaryIndex
from bitcache.quantize import quantize, quantize_batch

__version__ = "0.1.0"
__all__ = ["BinaryIndex", "quantize", "quantize_batch"]
