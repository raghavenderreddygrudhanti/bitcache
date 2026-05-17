"""Agent memory with prioritization, aging, and eviction.

Memories have importance scores that decay over time. Frequently
accessed memories get reinforced. Low-value memories are evicted
when capacity is reached.

Designed for long-running AI agents that need human-like memory:
- Remember important things
- Forget useless things
- Strengthen frequently used memories
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from bitcache.streaming import StreamingIndex


class AgentMemory:
    """Prioritized agent memory with decay and eviction.

    Each memory has:
    - vector: embedding for similarity search
    - content: the actual text/data
    - importance: score from 0.0 to 1.0
    - access_count: how many times retrieved
    - last_accessed: timestamp of last retrieval
    - created_at: insertion timestamp

    Importance decays over time. Retrieval reinforces importance.
    When capacity is reached, lowest-priority memories are evicted.

    Args:
        dim: Vector dimensionality.
        capacity: Max memories before eviction triggers.
        decay_rate: Importance decay per day (0.0 = no decay, 1.0 = full decay).
        reinforce_amount: How much retrieval boosts importance.
    """

    def __init__(
        self,
        dim: int,
        capacity: int = 10000,
        decay_rate: float = 0.05,
        reinforce_amount: float = 0.1,
        rerank_factor: int = 10,
    ):
        self.dim = dim
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.reinforce_amount = reinforce_amount

        self._index = StreamingIndex(dim=dim, rerank_factor=rerank_factor)
        self._memory_state: Dict[str, Dict[str, Any]] = {}

    def save_memory(
        self,
        vector: np.ndarray,
        content: str,
        importance: float = 0.5,
        id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a new memory.

        Args:
            vector: Embedding vector.
            content: The text content of the memory.
            importance: Initial importance score (0.0 to 1.0).
            id: Optional ID. Auto-generated if not provided.
            metadata: Optional additional metadata.

        Returns:
            Memory ID.
        """
        importance = max(0.0, min(1.0, importance))
        now = time.time()

        meta = metadata or {}
        meta["content"] = content
        meta["importance"] = importance

        mid = self._index.insert(vector, id=id, metadata=meta)

        self._memory_state[mid] = {
            "importance": importance,
            "access_count": 0,
            "last_accessed": now,
            "created_at": now,
        }

        # Evict if over capacity
        if len(self._index) > self.capacity:
            self._evict()

        return mid

    def retrieve_memory(
        self,
        query: np.ndarray,
        k: int = 5,
        min_importance: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories and reinforce them.

        Args:
            query: Query vector.
            k: Number of memories to return.
            min_importance: Minimum importance threshold.

        Returns:
            List of memory dicts with content, importance, score.
        """
        # Apply decay before searching
        self._apply_decay()

        # Build filter if min_importance > 0
        filter_dict = None
        # Can't filter by importance directly in metadata (it's a float comparison)
        # So we fetch more and filter post-hoc

        scores, ids, metas = self._index.search(query, k=k * 3)

        results = []
        for score, mid, meta in zip(scores, ids, metas):
            if mid not in self._memory_state:
                continue

            state = self._memory_state[mid]
            current_importance = state["importance"]

            if current_importance < min_importance:
                continue

            # Reinforce this memory (it was useful)
            self._reinforce(mid)

            results.append({
                "id": mid,
                "content": meta.get("content", ""),
                "importance": current_importance,
                "score": float(score),
                "access_count": state["access_count"],
                "metadata": {k: v for k, v in meta.items() if k not in ("content", "importance")},
            })

            if len(results) >= k:
                break

        return results

    def reinforce_memory(self, id: str, amount: Optional[float] = None) -> bool:
        """Manually reinforce a memory's importance.

        Args:
            id: Memory ID.
            amount: Reinforcement amount. Uses default if None.

        Returns:
            True if memory exists and was reinforced.
        """
        return self._reinforce(id, amount)

    def forget_memory(self, id: str) -> bool:
        """Explicitly forget (delete) a memory.

        Args:
            id: Memory ID.

        Returns:
            True if memory existed and was deleted.
        """
        if id in self._memory_state:
            del self._memory_state[id]
        return self._index.delete(id)

    def get_memory(self, id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID.

        Returns:
            Memory dict or None.
        """
        info = self._index.get(id)
        if info is None:
            return None

        state = self._memory_state.get(id, {})
        return {
            "id": id,
            "metadata": info["metadata"],
            "importance": state.get("importance", 0.0),
            "access_count": state.get("access_count", 0),
            "last_accessed": state.get("last_accessed"),
            "created_at": state.get("created_at"),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        if not self._memory_state:
            return {"total": 0, "capacity": self.capacity}

        importances = [s["importance"] for s in self._memory_state.values()]
        return {
            "total": len(self._memory_state),
            "capacity": self.capacity,
            "mean_importance": float(np.mean(importances)),
            "min_importance": float(np.min(importances)),
            "max_importance": float(np.max(importances)),
            "total_accesses": sum(s["access_count"] for s in self._memory_state.values()),
        }

    def _reinforce(self, id: str, amount: Optional[float] = None) -> bool:
        if id not in self._memory_state:
            return False

        state = self._memory_state[id]
        boost = amount if amount is not None else self.reinforce_amount
        state["importance"] = min(1.0, state["importance"] + boost)
        state["access_count"] += 1
        state["last_accessed"] = time.time()
        return True

    def _apply_decay(self) -> None:
        """Decay all memory importances based on time since last access."""
        now = time.time()
        for state in self._memory_state.values():
            days_since_access = (now - state["last_accessed"]) / 86400.0
            decay = self.decay_rate * days_since_access
            state["importance"] = max(0.0, state["importance"] - decay)

    def _evict(self) -> None:
        """Evict lowest-importance memories until under capacity."""
        while len(self._index) > self.capacity:
            # Find lowest importance memory
            worst_id = None
            worst_importance = float("inf")
            for mid, state in self._memory_state.items():
                if state["importance"] < worst_importance:
                    worst_importance = state["importance"]
                    worst_id = mid

            if worst_id is None:
                break

            self.forget_memory(worst_id)

    def __len__(self) -> int:
        return len(self._index)
