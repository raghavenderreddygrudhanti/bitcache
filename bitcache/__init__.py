"""bitcache — Binary vector retrieval engine for AI agent memory.

Based on: "QuIVer: Rethinking ANN Graph Topology via Training-Free
Binary Quantization" (Xiao et al., arXiv:2605.02171, 2026)
"""

from bitcache.index import BinaryIndex
from bitcache.graph import VamanaIndex
from bitcache.two_stage import TwoStageIndex
from bitcache.streaming import StreamingIndex
from bitcache.memory import AgentMemory
from bitcache.graph_memory import GraphMemory
from bitcache.quantize import quantize, quantize_batch

__version__ = "0.1.0"
__all__ = ["BinaryIndex", "VamanaIndex", "TwoStageIndex", "StreamingIndex", "AgentMemory", "GraphMemory", "quantize", "quantize_batch"]
