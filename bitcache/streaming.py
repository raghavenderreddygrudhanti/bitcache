"""Streaming index: supports live inserts, updates, and deletes.

No full rebuild needed. Vectors can be added and removed continuously,
making this suitable for long-running AI agent memory systems where
knowledge evolves over time.

Combines binary quantization (fast filter) with float32 storage (rerank)
and supports ID-based operations for stable references.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from bitcache.quantize import quantize
from bitcache.search import hamming_distance_single


class StreamingIndex:
    """Mutable vector index with insert, update, delete, and search.

    Designed for AI agent memory: vectors come and go as the agent
    learns and forgets. No rebuild required for mutations.

    Args:
        dim: Vector dimensionality.
        rerank_factor: Candidates fetched in binary stage = k * rerank_factor.
    """

    def __init__(self, dim: int, rerank_factor: int = 10):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.rerank_factor = rerank_factor
        self.n_bytes = (dim + 7) // 8

        # Storage: parallel arrays indexed by slot
        self._codes: List[np.ndarray] = []       # binary codes
        self._vectors: List[np.ndarray] = []     # unit float vectors
        self._metadata: List[Dict[str, Any]] = []  # per-vector metadata
        self._timestamps: List[float] = []       # insertion time

        # ID mapping: external string ID → internal slot
        self._id_to_slot: Dict[str, int] = {}
        self._slot_to_id: Dict[int, str] = {}

        # Deleted slots (reusable)
        self._free_slots: List[int] = []

        self._next_id: int = 0

    def insert(
        self,
        vector: np.ndarray,
        id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert a single vector.

        Args:
            vector: Float array of shape (dim,).
            id: Optional external ID. Auto-generated if not provided.
            metadata: Optional metadata dict stored alongside the vector.

        Returns:
            The ID assigned to this vector.
        """
        vector = np.asarray(vector, dtype=np.float32).ravel()
        if len(vector) != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {len(vector)}")

        if id is None:
            id = f"vec_{self._next_id}"
            self._next_id += 1

        # If ID already exists, update instead
        if id in self._id_to_slot:
            self.update(id, vector, metadata)
            return id

        # Normalize
        norm = np.linalg.norm(vector)
        unit = vector / norm if norm > 1e-10 else vector

        # Quantize
        code = quantize(unit[None, :])[0]

        # Find slot
        if self._free_slots:
            slot = self._free_slots.pop()
            self._codes[slot] = code
            self._vectors[slot] = unit
            self._metadata[slot] = metadata or {}
            self._timestamps[slot] = time.time()
        else:
            slot = len(self._codes)
            self._codes.append(code)
            self._vectors.append(unit)
            self._metadata.append(metadata or {})
            self._timestamps.append(time.time())

        self._id_to_slot[id] = slot
        self._slot_to_id[slot] = id

        return id

    def insert_batch(
        self,
        vectors: np.ndarray,
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Insert multiple vectors.

        Args:
            vectors: Float array of shape (n, dim).
            ids: Optional list of IDs. Auto-generated if not provided.
            metadatas: Optional list of metadata dicts.

        Returns:
            List of assigned IDs.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        n = len(vectors)

        if ids is None:
            ids = [None] * n
        if metadatas is None:
            metadatas = [None] * n

        result_ids = []
        for i in range(n):
            vid = self.insert(vectors[i], id=ids[i], metadata=metadatas[i])
            result_ids.append(vid)
        return result_ids

    def update(
        self,
        id: str,
        vector: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update an existing vector and/or its metadata.

        Args:
            id: The vector ID to update.
            vector: New float vector (optional, keeps old if None).
            metadata: New metadata (optional, keeps old if None).

        Returns:
            True if the ID existed and was updated.
        """
        if id not in self._id_to_slot:
            return False

        slot = self._id_to_slot[id]

        if vector is not None:
            vector = np.asarray(vector, dtype=np.float32).ravel()
            norm = np.linalg.norm(vector)
            unit = vector / norm if norm > 1e-10 else vector
            self._codes[slot] = quantize(unit[None, :])[0]
            self._vectors[slot] = unit

        if metadata is not None:
            self._metadata[slot] = metadata

        self._timestamps[slot] = time.time()
        return True

    def delete(self, id: str) -> bool:
        """Delete a vector by ID.

        Args:
            id: The vector ID to delete.

        Returns:
            True if the ID existed and was deleted.
        """
        if id not in self._id_to_slot:
            return False

        slot = self._id_to_slot[id]
        del self._id_to_slot[id]
        del self._slot_to_id[slot]
        self._free_slots.append(slot)

        return True

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
        """Search with optional metadata filtering.

        Args:
            query: Float vector of shape (dim,).
            k: Number of results.
            filter: Optional metadata filter (exact match on all keys).

        Returns:
            Tuple of (scores, ids, metadatas).
        """
        active_slots = list(self._slot_to_id.keys())
        if not active_slots:
            return np.array([]), [], []

        query = np.asarray(query, dtype=np.float32).ravel()
        norm = np.linalg.norm(query)
        query_unit = query / norm if norm > 1e-10 else query

        # Apply metadata filter first if provided
        if filter is not None:
            active_slots = [
                s for s in active_slots
                if self._matches_filter(self._metadata[s], filter)
            ]
            if not active_slots:
                return np.array([]), [], []

        # Stage 1: binary Hamming distance on active slots
        query_code = quantize(query_unit[None, :])[0]
        codes_matrix = np.array([self._codes[s] for s in active_slots])
        hamming_dists = hamming_distance_single(query_code, codes_matrix)

        # Select candidates
        n_candidates = min(k * self.rerank_factor, len(active_slots))
        if n_candidates < len(active_slots):
            cand_local = np.argpartition(hamming_dists, n_candidates)[:n_candidates]
        else:
            cand_local = np.arange(len(active_slots))

        # Stage 2: float32 rerank
        cand_slots = [active_slots[i] for i in cand_local]
        cand_vectors = np.array([self._vectors[s] for s in cand_slots])
        scores = cand_vectors @ query_unit

        # Top-k
        top_k_local = np.argsort(scores)[::-1][:k]
        result_slots = [cand_slots[i] for i in top_k_local]
        result_scores = scores[top_k_local]
        result_ids = [self._slot_to_id[s] for s in result_slots]
        result_meta = [self._metadata[s] for s in result_slots]

        return result_scores, result_ids, result_meta

    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a vector by ID.

        Returns:
            Metadata dict or None if ID not found.
        """
        if id not in self._id_to_slot:
            return None
        slot = self._id_to_slot[id]
        return {
            "id": id,
            "metadata": self._metadata[slot],
            "timestamp": self._timestamps[slot],
        }

    def save(self, path: str) -> None:
        """Save index to disk."""
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)

        np.save(str(folder / "codes.npy"), np.array(self._codes))
        np.save(str(folder / "vectors.npy"), np.array(self._vectors))

        state = {
            "dim": self.dim,
            "rerank_factor": self.rerank_factor,
            "id_to_slot": self._id_to_slot,
            "slot_to_id": {str(k): v for k, v in self._slot_to_id.items()},
            "metadata": self._metadata,
            "timestamps": self._timestamps,
            "free_slots": self._free_slots,
            "next_id": self._next_id,
        }
        with open(folder / "state.json", "w") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "StreamingIndex":
        """Load index from disk."""
        folder = Path(path)
        with open(folder / "state.json") as f:
            state = json.load(f)

        index = cls(dim=state["dim"], rerank_factor=state["rerank_factor"])
        index._codes = list(np.load(str(folder / "codes.npy")))
        index._vectors = list(np.load(str(folder / "vectors.npy")))
        index._metadata = state["metadata"]
        index._timestamps = state["timestamps"]
        index._id_to_slot = state["id_to_slot"]
        index._slot_to_id = {int(k): v for k, v in state["slot_to_id"].items()}
        index._free_slots = state["free_slots"]
        index._next_id = state["next_id"]
        return index

    def __len__(self) -> int:
        return len(self._id_to_slot)

    @staticmethod
    def _matches_filter(metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        for key, value in filter.items():
            if isinstance(value, list):
                if metadata.get(key) not in value:
                    return False
            else:
                if metadata.get(key) != value:
                    return False
        return True
