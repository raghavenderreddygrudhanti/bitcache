"""Two-stage progressive retrieval: binary coarse filter + float32 rerank.

Stage 1: Scan all binary codes via Hamming distance (fast, cheap).
         Select top-N candidates (N >> k).
Stage 2: Rerank only those N candidates using float32 inner product (precise).
         Return top-k.

This gives binary-level speed with float-level accuracy.

Based on: "FaTRQ: Tiered Residual Quantization for LLM Vector Search"
(Zhang et al., arXiv:2601.09985, 2026) — progressive refinement concept.
"""

import numpy as np
from typing import Optional

from bitcache.quantize import quantize
from bitcache.search import hamming_distance_single


class TwoStageIndex:
    """Two-stage retrieval: binary filter → float32 rerank.

    Stores both binary codes (for fast filtering) and normalized float
    vectors (for precise reranking). Search scans binary codes first,
    then reranks a small candidate set with full precision.

    Args:
        dim: Vector dimensionality.
        rerank_factor: How many candidates to fetch in stage 1.
            Stage 1 fetches k * rerank_factor candidates.
            Higher = better recall, slower.
    """

    def __init__(self, dim: int, rerank_factor: int = 10):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.rerank_factor = rerank_factor
        self.n_bytes = (dim + 7) // 8

        self._codes: Optional[np.ndarray] = None
        self._vectors: Optional[np.ndarray] = None
        self._norms: Optional[np.ndarray] = None
        self._n_vectors: int = 0

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

        norms = np.linalg.norm(vectors, axis=1)
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        unit_vectors = vectors / safe_norms[:, None]

        codes = quantize(unit_vectors)

        if self._codes is None:
            self._codes = codes
            self._vectors = unit_vectors
            self._norms = norms
        else:
            self._codes = np.vstack([self._codes, codes])
            self._vectors = np.vstack([self._vectors, unit_vectors])
            self._norms = np.concatenate([self._norms, norms])

        self._n_vectors = len(self._codes)

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Two-stage search: binary filter → float32 rerank.

        Args:
            query: Float vector of shape (dim,) or (1, dim).
            k: Number of final results.

        Returns:
            Tuple of (scores, indices).
            Scores are float32 inner products (higher = more similar).
        """
        if self._codes is None or self._n_vectors == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

        query = np.asarray(query, dtype=np.float32).ravel()
        norm = np.linalg.norm(query)
        if norm > 1e-10:
            query_unit = query / norm
        else:
            query_unit = query

        # Stage 1: binary Hamming scan — fetch top candidates
        query_code = quantize(query_unit[None, :])[0]
        n_candidates = min(k * self.rerank_factor, self._n_vectors)

        hamming_dists = hamming_distance_single(query_code, self._codes)

        # Get top-N candidates by lowest Hamming distance
        if n_candidates < self._n_vectors:
            candidate_indices = np.argpartition(hamming_dists, n_candidates)[:n_candidates]
        else:
            candidate_indices = np.arange(self._n_vectors)

        # Stage 2: float32 rerank — precise scoring on candidates only
        candidate_vectors = self._vectors[candidate_indices]
        scores = candidate_vectors @ query_unit

        # Select top-k from candidates
        top_k_local = np.argsort(scores)[::-1][:k]
        final_indices = candidate_indices[top_k_local]
        final_scores = scores[top_k_local]

        return final_scores, final_indices

    def search_batch(
        self, queries: np.ndarray, k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch two-stage search.

        Args:
            queries: Float array of shape (nq, dim).
            k: Number of results per query.

        Returns:
            Tuple of (scores, indices) each shape (nq, k).
        """
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]

        nq = len(queries)
        all_scores = np.zeros((nq, k), dtype=np.float32)
        all_indices = np.zeros((nq, k), dtype=np.int64)

        for i in range(nq):
            scores, indices = self.search(queries[i], k=k)
            n_results = len(scores)
            all_scores[i, :n_results] = scores
            all_indices[i, :n_results] = indices

        return all_scores, all_indices

    def __len__(self) -> int:
        return self._n_vectors

    @property
    def memory_usage_bytes(self) -> int:
        """Total memory: binary codes + float vectors."""
        if self._codes is None:
            return 0
        return self._codes.nbytes + self._vectors.nbytes

    @property
    def binary_only_bytes(self) -> int:
        """Memory if only binary codes were stored (no float vectors)."""
        if self._codes is None:
            return 0
        return self._codes.nbytes
