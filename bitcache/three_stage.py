"""Three-stage progressive retrieval: binary → 4-bit → float32.

Stage 1: Binary Hamming scan (1-bit per dim). Cheap. Selects broad candidates.
Stage 2: 4-bit quantized inner product on candidates. Medium cost. Refines ranking.
Stage 3: Float32 inner product on top candidates. Precise. Final ranking.

Memory hierarchy:
  Stage 1 codes: 96 bytes/vector (d=768)  — always in RAM
  Stage 2 codes: 384 bytes/vector (d=768) — in RAM
  Stage 3 vectors: 3072 bytes/vector (d=768) — loaded on demand
"""

import numpy as np
from typing import Optional

from bitcache.quantize import quantize
from bitcache.search import hamming_distance_single


class ThreeStageIndex:
    """Three-stage retrieval: binary filter → 4-bit rerank → float32 rerank.

    Each stage narrows the candidate set with increasing precision.
    Achieves higher recall than two-stage by adding an intermediate
    quantized scoring step that catches candidates missed by binary
    ranking alone.

    Args:
        dim: Vector dimensionality.
        stage1_factor: Candidates from Stage 1 = k * stage1_factor.
        stage2_factor: Candidates from Stage 2 = k * stage2_factor.
        n_bits: Quantization bits for Stage 2 (default 4).
    """

    def __init__(
        self,
        dim: int,
        stage1_factor: int = 200,
        stage2_factor: int = 20,
        n_bits: int = 4,
    ):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.stage1_factor = stage1_factor
        self.stage2_factor = stage2_factor
        self.n_bits = n_bits
        self.n_levels = 2 ** n_bits  # 16 for 4-bit

        self._binary_codes: Optional[np.ndarray] = None
        self._quant_codes: Optional[np.ndarray] = None
        self._vectors: Optional[np.ndarray] = None
        self._centroids: Optional[np.ndarray] = None
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

        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        unit_vectors = vectors / norms

        # Stage 1: binary codes
        binary_codes = quantize(unit_vectors)

        # Stage 2: 4-bit uniform scalar quantization per dimension
        quant_codes, centroids = self._quantize_4bit(unit_vectors)

        if self._binary_codes is None:
            self._binary_codes = binary_codes
            self._quant_codes = quant_codes
            self._vectors = unit_vectors
            self._centroids = centroids
        else:
            self._binary_codes = np.vstack([self._binary_codes, binary_codes])
            self._quant_codes = np.vstack([self._quant_codes, quant_codes])
            self._vectors = np.vstack([self._vectors, unit_vectors])
            # Recompute centroids on full data
            self._quant_codes, self._centroids = self._quantize_4bit(self._vectors)

        self._n_vectors = len(self._binary_codes)

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Three-stage search.

        Args:
            query: Float vector of shape (dim,).
            k: Number of final results.

        Returns:
            Tuple of (scores, indices). Scores are float32 inner products.
        """
        if self._binary_codes is None or self._n_vectors == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

        query = np.asarray(query, dtype=np.float32).ravel()
        norm = np.linalg.norm(query)
        query_unit = query / norm if norm > 1e-10 else query

        # Stage 1: binary Hamming — broad candidate selection
        query_binary = quantize(query_unit[None, :])[0]
        n_stage1 = min(k * self.stage1_factor, self._n_vectors)

        hamming_dists = hamming_distance_single(query_binary, self._binary_codes)

        if n_stage1 < self._n_vectors:
            stage1_indices = np.argpartition(hamming_dists, n_stage1)[:n_stage1]
        else:
            stage1_indices = np.arange(self._n_vectors)

        # Stage 2: 4-bit quantized inner product — refine candidates
        n_stage2 = min(k * self.stage2_factor, len(stage1_indices))

        query_quant_scores = self._score_quantized(query_unit, stage1_indices)

        if n_stage2 < len(stage1_indices):
            stage2_local = np.argpartition(query_quant_scores, -n_stage2)[-n_stage2:]
        else:
            stage2_local = np.arange(len(stage1_indices))

        stage2_indices = stage1_indices[stage2_local]

        # Stage 3: float32 inner product — precise final ranking
        candidate_vectors = self._vectors[stage2_indices]
        scores = candidate_vectors @ query_unit

        top_k_local = np.argsort(scores)[::-1][:k]
        final_indices = stage2_indices[top_k_local]
        final_scores = scores[top_k_local]

        return final_scores, final_indices

    def search_batch(self, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Batch three-stage search.

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
            n = len(scores)
            all_scores[i, :n] = scores
            all_indices[i, :n] = indices

        return all_scores, all_indices

    def _quantize_4bit(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Uniform scalar quantization to n_bits per dimension.

        Maps each coordinate to one of 2^n_bits levels based on
        the observed min/max range per dimension.

        Returns:
            Tuple of (codes as uint8 array shape (n, dim), centroids shape (n_levels, dim)).
        """
        # Compute per-dimension min/max
        vmin = vectors.min(axis=0)
        vmax = vectors.max(axis=0)
        vrange = vmax - vmin
        vrange = np.where(vrange > 1e-10, vrange, 1.0)

        # Quantize to [0, n_levels-1]
        normalized = (vectors - vmin) / vrange
        codes = np.clip(normalized * self.n_levels, 0, self.n_levels - 1).astype(np.uint8)

        # Compute centroids (midpoint of each bucket)
        centroids = np.zeros((self.n_levels, self.dim), dtype=np.float32)
        for level in range(self.n_levels):
            centroids[level] = vmin + (level + 0.5) / self.n_levels * vrange

        # Store min/range for query quantization
        self._vmin = vmin
        self._vrange = vrange

        return codes, centroids

    def _score_quantized(self, query_unit: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Compute approximate inner product using 4-bit codes.

        For each candidate, reconstruct the vector from its quantized codes
        using the centroid lookup table, then dot with query.
        """
        # Precompute query dot with each centroid per dimension
        # query_lut[level, dim] = query[dim] * centroid[level, dim]
        # Score for vector i = sum over dims of query_lut[code[i, dim], dim]

        # Build lookup table: for each dimension, score contribution per level
        lut = np.zeros((self.n_levels, self.dim), dtype=np.float32)
        for level in range(self.n_levels):
            lut[level] = query_unit * self._centroids[level]

        # Score each candidate
        candidate_codes = self._quant_codes[indices]  # shape (m, dim)
        n_candidates = len(indices)

        scores = np.zeros(n_candidates, dtype=np.float32)
        for i in range(n_candidates):
            # Sum lut[code[i,j], j] for all j
            codes_i = candidate_codes[i]
            scores[i] = sum(lut[codes_i[j], j] for j in range(self.dim))

        return scores

    def __len__(self) -> int:
        return self._n_vectors

    @property
    def memory_usage(self) -> dict:
        """Memory breakdown by stage."""
        if self._n_vectors == 0:
            return {"stage1": 0, "stage2": 0, "stage3": 0, "total": 0}
        s1 = self._binary_codes.nbytes
        s2 = self._quant_codes.nbytes
        s3 = self._vectors.nbytes
        return {
            "stage1_bytes": s1,
            "stage2_bytes": s2,
            "stage3_bytes": s3,
            "total_bytes": s1 + s2 + s3,
            "stage1_mb": round(s1 / 1024**2, 1),
            "stage2_mb": round(s2 / 1024**2, 1),
            "stage3_mb": round(s3 / 1024**2, 1),
        }
