"""BinaryIndex: the main user-facing class.

Handles quantization, storage, search, and persistence.
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from bitcache.quantize import quantize
from bitcache.search import search, search_batch


class BinaryIndex:
    """Binary vector index with XOR + POPCOUNT search.

    Quantizes float vectors to binary (sign-bit) and searches via
    Hamming distance. 32x memory compression vs float32.

    Example:
        >>> index = BinaryIndex(dim=1536)
        >>> index.add(vectors)  # shape (n, 1536) float32
        >>> distances, indices = index.search(query, k=10)
    """

    def __init__(self, dim: int):
        """Initialize empty index.

        Args:
            dim: Dimensionality of input vectors.
        """
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.n_bytes = (dim + 7) // 8
        self._codes: Optional[np.ndarray] = None
        self._norms: Optional[np.ndarray] = None
        self._n_vectors = 0

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to the index.

        Args:
            vectors: Float array of shape (n, dim).
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {vectors.shape[1]}")

        # Store norms for optional reranking
        norms = np.linalg.norm(vectors, axis=1)

        # Normalize before quantization
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        unit_vectors = vectors / safe_norms[:, None]

        codes = quantize(unit_vectors)

        if self._codes is None:
            self._codes = codes
            self._norms = norms
        else:
            self._codes = np.vstack([self._codes, codes])
            self._norms = np.concatenate([self._norms, norms])

        self._n_vectors = len(self._codes)

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest vectors.

        Args:
            query: Float vector of shape (dim,) or (1, dim).
            k: Number of results.

        Returns:
            Tuple of (distances, indices). Distances are Hamming distances
            (lower = more similar).
        """
        if self._codes is None or self._n_vectors == 0:
            return np.array([], dtype=np.uint32), np.array([], dtype=np.int64)

        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query[None, :]

        norm = np.linalg.norm(query, axis=1, keepdims=True)
        norm = np.where(norm > 1e-10, norm, 1.0)
        query_unit = query / norm

        query_codes = quantize(query_unit)
        return search(query_codes[0], self._codes, k)

    def search_batch(self, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Batch search for k nearest vectors per query.

        Args:
            queries: Float array of shape (nq, dim).
            k: Number of results per query.

        Returns:
            Tuple of (distances, indices) each shape (nq, k).
        """
        if self._codes is None or self._n_vectors == 0:
            nq = len(queries)
            return np.zeros((nq, 0), dtype=np.uint32), np.zeros((nq, 0), dtype=np.int64)

        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]

        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        queries_unit = queries / norms

        query_codes = quantize(queries_unit)
        return search_batch(query_codes, self._codes, k)

    def save(self, path: str) -> None:
        """Save index to disk.

        Args:
            path: Directory path to save into.
        """
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)

        np.save(str(folder / "codes.npy"), self._codes)
        np.save(str(folder / "norms.npy"), self._norms)

        meta = {"dim": self.dim, "n_vectors": self._n_vectors, "n_bytes": self.n_bytes}
        with open(folder / "meta.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "BinaryIndex":
        """Load index from disk.

        Args:
            path: Directory path to load from.

        Returns:
            Loaded BinaryIndex.
        """
        folder = Path(path)
        with open(folder / "meta.json") as f:
            meta = json.load(f)

        index = cls(dim=meta["dim"])
        index._codes = np.load(str(folder / "codes.npy"))
        index._norms = np.load(str(folder / "norms.npy"))
        index._n_vectors = meta["n_vectors"]
        return index

    def __len__(self) -> int:
        return self._n_vectors

    @property
    def memory_usage_bytes(self) -> int:
        """Memory used by the binary codes in bytes."""
        if self._codes is None:
            return 0
        return self._codes.nbytes

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs float32 storage."""
        if self._n_vectors == 0:
            return 0.0
        fp32_bytes = self._n_vectors * self.dim * 4
        return fp32_bytes / self.memory_usage_bytes
