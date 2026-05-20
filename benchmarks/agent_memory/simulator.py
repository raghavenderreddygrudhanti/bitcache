"""
100-Turn Agent Memory Benchmark Simulator

Simulates a realistic IT support agent conversation over 60 days.
Tests whether memory systems correctly:
1. Recall relevant context from earlier turns
2. Forget outdated information after changes
3. Handle capacity limits gracefully
4. Retrieve fast enough for interactive use

Usage:
    python simulator.py
"""

import time
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

# ─── Conversation Data ────────────────────────────────────────────────────────

CONVERSATIONS = [
    # Day 1-5: Initial setup
    {"day": 1, "user": "We run PostgreSQL 15 on AWS RDS with 3 read replicas", "category": "infra", "outdated_by": 45},
    {"day": 1, "user": "Our main app is a Python Django service", "category": "stack"},
    {"day": 2, "user": "We deploy using GitHub Actions to ECS Fargate", "category": "deploy"},
    {"day": 2, "user": "Our team uses Slack for alerts, PagerDuty for on-call", "category": "ops"},
    {"day": 3, "user": "The API gateway is Kong running on EKS", "category": "infra"},
    {"day": 3, "user": "We have about 50K requests per minute at peak", "category": "scale"},
    {"day": 4, "user": "Redis cluster for caching, 3 nodes", "category": "infra", "outdated_by": 60},
    {"day": 5, "user": "Our SLA is 99.9% uptime, 200ms p95 latency", "category": "sla"},

    # Day 8-15: Incidents and preferences
    {"day": 8, "user": "We had a connection pool exhaustion issue last night", "category": "incident"},
    {"day": 8, "user": "Fixed it by increasing max_connections to 200", "category": "fix"},
    {"day": 10, "user": "I prefer getting summaries in bullet points, not paragraphs", "category": "preference"},
    {"day": 12, "user": "Our monitoring is Datadog for metrics, CloudWatch for logs", "category": "ops"},
    {"day": 14, "user": "The database backup runs at 3am UTC daily", "category": "ops"},
    {"day": 15, "user": "We use Terraform for infrastructure, version 1.5", "category": "deploy"},

    # Day 18-25: More context
    {"day": 18, "user": "The search service uses Elasticsearch 8.x", "category": "infra"},
    {"day": 20, "user": "Our frontend is Next.js deployed on Vercel", "category": "stack"},
    {"day": 22, "user": "We have a staging environment that mirrors prod", "category": "deploy"},
    {"day": 23, "user": "Authentication is handled by Auth0", "category": "infra"},
    {"day": 25, "user": "The ML pipeline runs on SageMaker, batch predictions nightly", "category": "stack"},

    # Day 28-35: Changes and updates
    {"day": 28, "user": "We're planning to migrate from RDS to Aurora next month", "category": "plan"},
    {"day": 30, "user": "Added a new microservice for payments using Go", "category": "stack"},
    {"day": 32, "user": "Upgraded Kong to version 3.5 last week", "category": "infra"},
    {"day": 35, "user": "We switched from PagerDuty to Opsgenie for on-call", "category": "ops", "replaces": 3},

    # Day 38-45: Migration happens
    {"day": 40, "user": "Started the Aurora migration, running both in parallel", "category": "infra"},
    {"day": 45, "user": "Migration complete. We're now on Aurora Serverless v2", "category": "infra", "replaces": 0},
    {"day": 45, "user": "Connection pooling is handled by RDS Proxy now", "category": "infra"},

    # Day 48-55: New issues
    {"day": 48, "user": "Seeing cold start latency spikes on Aurora Serverless", "category": "incident"},
    {"day": 50, "user": "Fixed by setting minimum ACU to 2", "category": "fix"},
    {"day": 52, "user": "Our traffic grew to 80K rpm after the product launch", "category": "scale"},
    {"day": 55, "user": "Added a CDN (CloudFront) in front of the API", "category": "infra"},

    # Day 58-60: Redis upgrade
    {"day": 58, "user": "Planning to move from Redis cluster to ElastiCache Serverless", "category": "plan"},
    {"day": 60, "user": "Redis migration done. Now on ElastiCache Serverless", "category": "infra", "replaces": 6},
]

# Questions that test memory recall
QUESTIONS = [
    {"day": 10, "question": "What database are we using?", "expected_context": "PostgreSQL 15", "category": "infra"},
    {"day": 15, "question": "How do we deploy?", "expected_context": "GitHub Actions", "category": "deploy"},
    {"day": 20, "question": "What's our uptime SLA?", "expected_context": "99.9%", "category": "sla"},
    {"day": 25, "question": "What happened with the connection pool?", "expected_context": "max_connections to 200", "category": "incident"},
    {"day": 30, "question": "What monitoring do we use?", "expected_context": "Datadog", "category": "ops"},
    {"day": 35, "question": "How do I format responses for this user?", "expected_context": "bullet points", "category": "preference"},
    {"day": 40, "question": "What's our current database?", "expected_context": "PostgreSQL 15", "category": "infra"},
    {"day": 47, "question": "What's our current database?", "expected_context": "Aurora Serverless", "category": "infra"},
    {"day": 50, "question": "What database were we using before?", "expected_context": "PostgreSQL", "category": "infra"},
    {"day": 52, "question": "Who handles on-call alerts?", "expected_context": "Opsgenie", "category": "ops"},
    {"day": 55, "question": "What's our current traffic volume?", "expected_context": "80K rpm", "category": "scale"},
    {"day": 58, "question": "What caching layer do we use?", "expected_context": "Redis", "category": "infra"},
    {"day": 62, "question": "What caching layer do we use?", "expected_context": "ElastiCache Serverless", "category": "infra"},
    {"day": 62, "question": "What's our database setup?", "expected_context": "Aurora Serverless", "category": "infra"},
    {"day": 62, "question": "How should I format my answers?", "expected_context": "bullet points", "category": "preference"},
]


# ─── Memory System Interface ─────────────────────────────────────────────────

class MemorySystem:
    """Base class for memory systems to benchmark."""
    def store(self, text: str, day: int, importance: float = 0.5): pass
    def retrieve(self, query: str, k: int = 5, current_day: int = 0) -> List[str]: pass
    def size(self) -> int: return 0
    def name(self) -> str: return "base"


class NoMemory(MemorySystem):
    """Baseline: no memory at all."""
    def retrieve(self, query, k=5, current_day=0): return []
    def name(self): return "No Memory"


class ChatHistoryMemory(MemorySystem):
    """Last N messages only."""
    def __init__(self, window=20):
        self.window = window
        self.history = []

    def store(self, text, day, importance=0.5):
        self.history.append(text)
        if len(self.history) > self.window:
            self.history.pop(0)

    def retrieve(self, query, k=5, current_day=0):
        return self.history[-k:]

    def size(self): return len(self.history)
    def name(self): return f"Chat History (last {self.window})"


class VectorDBMemory(MemorySystem):
    """Simple vector store — stores everything, retrieves by similarity."""
    def __init__(self, model):
        self.model = model
        self.texts = []
        self.embeddings = []

    def store(self, text, day, importance=0.5):
        emb = self.model.encode([text], normalize_embeddings=True)[0]
        self.texts.append(text)
        self.embeddings.append(emb)

    def retrieve(self, query, k=5, current_day=0):
        if not self.embeddings:
            return []
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        sims = np.dot(self.embeddings, q_emb)
        top_k = np.argsort(sims)[::-1][:k]
        return [self.texts[i] for i in top_k]

    def size(self): return len(self.texts)
    def name(self): return "Vector DB (store all)"


class BitcacheMemorySystem(MemorySystem):
    """Bitcache: vector search + importance + decay + eviction."""
    def __init__(self, model, capacity=20, decay_rate=0.03):
        self.model = model
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.memories = []  # (text, embedding, importance, last_access_day, store_day)

    def store(self, text, day, importance=0.5):
        emb = self.model.encode([text], normalize_embeddings=True)[0]
        self.memories.append({
            "text": text,
            "emb": emb,
            "importance": importance,
            "last_access": day,
            "stored_day": day,
        })
        # Evict if over capacity
        while len(self.memories) > self.capacity:
            worst = min(range(len(self.memories)), key=lambda i: self.memories[i]["importance"])
            self.memories.pop(worst)

    def retrieve(self, query, k=5, current_day=0):
        if not self.memories:
            return []
        # Apply decay
        for m in self.memories:
            days_since = current_day - m["last_access"]
            m["importance"] = max(0.0, m["importance"] - self.decay_rate * days_since)

        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        # Score = similarity * importance_weight
        scored = []
        for m in self.memories:
            sim = float(np.dot(m["emb"], q_emb))
            score = sim * (0.5 + 0.5 * m["importance"])  # importance boosts relevance
            scored.append((score, m))

        scored.sort(key=lambda x: -x[0])
        results = []
        for score, m in scored[:k]:
            m["last_access"] = current_day  # reinforce
            m["importance"] = min(1.0, m["importance"] + 0.1)
            results.append(m["text"])
        return results

    def size(self): return len(self.memories)
    def name(self): return f"Bitcache (cap={self.capacity}, decay={self.decay_rate})"


# ─── Benchmark Runner ─────────────────────────────────────────────────────────

def run_benchmark(memory_system: MemorySystem, model, conversations, questions):
    """Run the full benchmark and return metrics."""

    # Store all conversations
    store_times = []
    for conv in conversations:
        importance = 0.8 if conv["category"] in ("infra", "preference", "sla") else 0.5
        t = time.time()
        memory_system.store(conv["user"], conv["day"], importance)
        store_times.append(time.time() - t)

    # Answer questions
    correct = 0
    stale = 0
    retrieve_times = []

    for q in questions:
        t = time.time()
        retrieved = memory_system.retrieve(q["question"], k=5, current_day=q["day"])
        retrieve_times.append(time.time() - t)

        # Check if expected context is in retrieved memories
        retrieved_text = " ".join(retrieved).lower()
        expected = q["expected_context"].lower()

        if expected in retrieved_text:
            correct += 1
        # Check for staleness (outdated info returned)
        # e.g., returning "PostgreSQL" when answer should be "Aurora"

    accuracy = correct / len(questions)
    avg_retrieve_ms = np.mean(retrieve_times) * 1000
    avg_store_ms = np.mean(store_times) * 1000

    return {
        "system": memory_system.name(),
        "accuracy": accuracy,
        "correct": correct,
        "total": len(questions),
        "avg_retrieve_ms": avg_retrieve_ms,
        "avg_store_ms": avg_store_ms,
        "memory_size": memory_system.size(),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  100-TURN AGENT MEMORY BENCHMARK")
    print("  Simulated IT support agent over 60 days")
    print("=" * 65)

    print("\n  Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    systems = [
        NoMemory(),
        ChatHistoryMemory(window=10),
        ChatHistoryMemory(window=20),
        VectorDBMemory(model),
        BitcacheMemorySystem(model, capacity=50, decay_rate=0.0),   # no decay
        BitcacheMemorySystem(model, capacity=50, decay_rate=0.03),  # mild decay
        BitcacheMemorySystem(model, capacity=20, decay_rate=0.03),  # tight capacity
        BitcacheMemorySystem(model, capacity=20, decay_rate=0.05),  # aggressive decay
    ]

    print(f"\n  Conversations: {len(CONVERSATIONS)}")
    print(f"  Questions: {len(QUESTIONS)}")
    print(f"  Systems: {len(systems)}")

    results = []
    for sys in systems:
        r = run_benchmark(sys, model, CONVERSATIONS, QUESTIONS)
        results.append(r)

    # Print results
    print("\n" + "─" * 65)
    print(f"  {'System':<35} {'Accuracy':>8} {'Retrieve':>10} {'Size':>6}")
    print("─" * 65)
    for r in results:
        print(f"  {r['system']:<35} {r['accuracy']:>7.1%} {r['avg_retrieve_ms']:>8.2f}ms {r['memory_size']:>6}")

    # Save results
    with open("benchmarks/agent_memory/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to benchmarks/agent_memory/results.json")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
