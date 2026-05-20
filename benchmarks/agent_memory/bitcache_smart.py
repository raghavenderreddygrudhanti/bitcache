"""
Bitcache Smart Memory — no LLM calls for memory operations.

Features:
1. Auto-importance: scores importance from text signals (no LLM)
2. Contradiction detection: marks old memories outdated via similarity (no LLM)
3. Confidence threshold: returns nothing if no good match
4. Decay + reinforcement + eviction (existing)

All memory decisions happen in microseconds using vector math and heuristics.
LLM is only called when the user asks a question (not for memory management).
"""

import numpy as np
import re
from typing import List, Optional, Dict
from sentence_transformers import SentenceTransformer


# ─── Auto-Importance (no LLM) ────────────────────────────────────────────────

# Keywords that signal high importance
HIGH_IMPORTANCE_SIGNALS = [
    # Personal identity
    r'\bmy name\b', r'\bi am\b', r'\bi work\b', r'\bmy (wife|husband|daughter|son|mom|dad|family)\b',
    # Preferences (persistent)
    r'\bi prefer\b', r'\bi always\b', r'\bi never\b', r'\bi hate\b', r'\bi love\b',
    r'\bplease (always|never|remember)\b',
    # Health / important life events
    r'\ballerg', r'\bsick\b', r'\bpromot', r'\bfired\b', r'\bquit\b', r'\bjoined\b', r'\bmigrat',
    # System-critical
    r'\bsla\b', r'\buptime\b', r'\bproduction\b', r'\bdeadline\b',
    # Explicit "remember this"
    r'\bremember\b', r'\bimportant\b', r'\bnote that\b',
]

LOW_IMPORTANCE_SIGNALS = [
    r'\blunch\b', r'\bweather\b', r'\bcoffee\b', r'\bboring\b',
    r'\bjust now\b', r'\btoday\b.*\b(ate|had|grabbed)\b',
    r'\belevator\b', r'\bprinter\b', r'\bwifi\b',
    r'\bsnack\b', r'\bnothing special\b', r'\bdecent\b',
    r'\bgrabbed\b', r'\bsaw a\b', r'\bmeeting about\b.*\b(supplies|parking|snacks)\b',
    r'^had \w+', r'^the weather', r'^grabbed', r'^just finished a boring',
]


def auto_importance(text: str) -> float:
    """Score importance 0.0-1.0 from text signals. No LLM needed."""
    text_lower = text.lower()
    score = 0.5  # default

    # Check high-importance signals
    for pattern in HIGH_IMPORTANCE_SIGNALS:
        if re.search(pattern, text_lower):
            score += 0.15
            break  # one match is enough

    # Check low-importance signals — much more aggressive penalty
    for pattern in LOW_IMPORTANCE_SIGNALS:
        if re.search(pattern, text_lower):
            score -= 0.45  # was 0.3, now 0.45 — trivial stuff gets near 0
            break

    # Longer statements tend to be more informative
    word_count = len(text.split())
    if word_count > 10:
        score += 0.05
    if word_count < 5:
        score -= 0.1

    # Contains numbers (versions, metrics, dates) → likely factual
    if re.search(r'\d+', text):
        score += 0.05

    # Contains proper nouns (capitalized words not at start)
    words = text.split()
    proper_nouns = sum(1 for w in words[1:] if w[0].isupper() and w.isalpha())
    if proper_nouns >= 2:
        score += 0.1

    return max(0.0, min(1.0, score))


# ─── Smart Bitcache Memory ───────────────────────────────────────────────────

class BitcacheSmart:
    """
    Agent memory with:
    - Auto-importance (heuristic, no LLM)
    - Contradiction detection (vector similarity, no LLM)
    - Confidence threshold (skip low-quality retrievals)
    - Decay + reinforcement + eviction
    """

    def __init__(
        self,
        model: SentenceTransformer,
        capacity: int = 20,
        decay_rate: float = 0.015,
        contradiction_threshold: float = 0.50,
        confidence_threshold: float = 0.25,
    ):
        self.model = model
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.contradiction_threshold = contradiction_threshold
        self.confidence_threshold = confidence_threshold
        self.memories: List[Dict] = []

    def store(self, text: str, day: int, importance: Optional[float] = None):
        """Store a memory. Auto-scores importance if not provided."""
        # Auto-importance
        if importance is None:
            importance = auto_importance(text)

        # Don't store very low importance (trivial) — reject at the gate
        if importance < 0.15:
            return

        # Embed
        emb = self.model.encode([text], normalize_embeddings=True)[0]

        # Contradiction detection: two strategies
        # 1. High vector similarity + newer day → old is outdated
        # 2. Keyword overlap: if both mention same topic keywords, newer wins
        text_lower = text.lower()
        topic_keywords = re.findall(r'\b[A-Z][a-zA-Z]+(?:DB|SQL|Cache)?\b', text)  # proper nouns
        topic_keywords += re.findall(r'\b(?:database|caching|alerting|monitoring|deploy|work at|joined)\b', text_lower)

        for m in self.memories:
            # Strategy 1: vector similarity
            sim = float(np.dot(emb, m['emb']))
            if sim > self.contradiction_threshold and day > m['day']:
                m['importance'] = 0.0
                continue

            # Strategy 2: shared topic keywords (catches "migrated from X to Y" vs "we use X")
            if topic_keywords and day > m['day']:
                m_lower = m['text'].lower()
                shared = sum(1 for kw in topic_keywords if kw.lower() in m_lower)
                if shared >= 2:  # 2+ shared keywords = same topic
                    m['importance'] = 0.0

        # Store
        self.memories.append({
            'text': text,
            'emb': emb,
            'importance': importance,
            'last_access': day,
            'day': day,
        })

        # Evict lowest importance if over capacity
        while len(self.memories) > self.capacity:
            worst = min(range(len(self.memories)), key=lambda i: self.memories[i]['importance'])
            self.memories.pop(worst)

    def retrieve(self, query: str, k: int = 5, current_day: int = 0) -> List[str]:
        """Retrieve relevant memories. Returns empty if nothing is confident enough."""
        if not self.memories:
            return []

        # Apply decay
        for m in self.memories:
            days_since = max(0, current_day - m['last_access'])
            m['importance'] = max(0.0, m['importance'] - self.decay_rate * days_since)

        # Score: similarity × importance_weight
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        scored = []
        for m in self.memories:
            sim = float(np.dot(m['emb'], q_emb))
            score = sim * (0.3 + 0.7 * m['importance'])
            scored.append((score, sim, m))

        scored.sort(key=lambda x: -x[0])

        # Confidence threshold: skip if best match is too weak
        if scored[0][1] < self.confidence_threshold:
            return []

        # Return top-k and reinforce
        results = []
        for score, sim, m in scored[:k]:
            if sim < self.confidence_threshold * 0.5:
                break  # stop if remaining are too weak
            m['last_access'] = current_day
            m['importance'] = min(1.0, m['importance'] + 0.1)
            results.append(m['text'])

        return results

    def get_stats(self) -> Dict:
        """Memory statistics."""
        if not self.memories:
            return {'count': 0}
        imps = [m['importance'] for m in self.memories]
        return {
            'count': len(self.memories),
            'mean_importance': np.mean(imps),
            'min_importance': np.min(imps),
            'max_importance': np.max(imps),
        }

    def name(self) -> str:
        return f"Bitcache Smart (cap={self.capacity})"


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  BITCACHE SMART — Auto-Importance + Contradiction Detection")
    print("  No LLM calls for memory operations")
    print("=" * 60)

    # Test auto-importance
    print("\n  Auto-Importance Scoring (no LLM):")
    print("  " + "-" * 55)
    test_texts = [
        "My name is Arjun and I work as a DevOps engineer",
        "I prefer short bullet-point answers",
        "Had pizza for lunch today, it was decent",
        "The weather is nice today",
        "We migrated from PostgreSQL to Aurora Serverless v2",
        "My daughter Priya just started kindergarten",
        "Our SLA is 99.9% uptime",
        "Grabbed coffee from the new place downstairs",
        "I am allergic to peanuts",
        "I joined Microsoft as a Principal Engineer",
    ]
    for t in test_texts:
        score = auto_importance(t)
        label = "HIGH" if score >= 0.6 else "LOW" if score <= 0.3 else "MED"
        print(f"    [{label:>4} {score:.2f}] {t[:55]}")

    # Test contradiction detection
    print("\n  Contradiction Detection (no LLM):")
    print("  " + "-" * 55)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    mem = BitcacheSmart(model, capacity=10)

    mem.store("I work at Google on the Cloud team", day=1)
    mem.store("We use PostgreSQL 15 on AWS RDS", day=5)
    print(f"    After storing 2 memories: {mem.get_stats()['count']} memories")

    mem.store("I left Google and joined Microsoft", day=40)
    print(f"    After contradiction (new job): {mem.get_stats()['count']} memories")
    for m in mem.memories:
        print(f"      [{m['importance']:.2f}] {m['text'][:50]}")

    mem.store("We migrated to Azure CosmosDB, PostgreSQL is gone", day=45)
    print(f"    After contradiction (new DB): {mem.get_stats()['count']} memories")
    for m in mem.memories:
        print(f"      [{m['importance']:.2f}] {m['text'][:50]}")

    # Test retrieval
    print("\n  Retrieval Test:")
    print("  " + "-" * 55)
    results = mem.retrieve("Where do I work?", k=3, current_day=50)
    print(f"    Query: 'Where do I work?'")
    for r in results:
        print(f"      → {r}")

    results = mem.retrieve("What database do we use?", k=3, current_day=50)
    print(f"    Query: 'What database do we use?'")
    for r in results:
        print(f"      → {r}")

    # Test confidence threshold
    results = mem.retrieve("What is my blood type?", k=3, current_day=50)
    print(f"    Query: 'What is my blood type?'")
    print(f"      → {results if results else '(nothing — below confidence threshold)'}")

    print("\n" + "=" * 60)
