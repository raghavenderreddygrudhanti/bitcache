"""Paper 3: End-to-end agent memory workload benchmark.

Measures latency for each layer of the memory architecture:
- Insert (with embedding)
- Semantic retrieval
- Graph expansion
- Reinforcement
- Decay
- Eviction

Run: python experiments/paper3_memory_systems/eval_agent_workload.py
"""

import json
import os
import time

import numpy as np

DIM = 384
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def embed_fake(text):
    np.random.seed(abs(hash(text)) % 2**32)
    v = np.random.randn(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def main():
    from bitcache import AgentMemory
    from bitcache.graph_memory import GraphMemory

    print("Paper 3: Agent Memory Workload Benchmark")
    print("=" * 50)

    results = {}

    # Insert benchmark
    mem = AgentMemory(dim=DIM, capacity=1000, decay_rate=0.1, reinforce_amount=0.15)
    events = [f"operational event {i}: system alert on server-{i%10}" for i in range(100)]

    t0 = time.time()
    for i, evt in enumerate(events):
        vec = embed_fake(evt)
        mem.save_memory(vec, content=evt, importance=np.random.uniform(0.3, 0.9))
    insert_time = (time.time() - t0) * 1000
    results["insert_100_memories_ms"] = round(insert_time, 2)
    print(f"  Insert 100 memories: {insert_time:.2f}ms")

    # Retrieval benchmark
    query = embed_fake("database connection timeout")
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        mem.retrieve_memory(query, k=5)
        latencies.append((time.perf_counter() - t0) * 1000)
    avg_retrieval = np.mean(latencies)
    results["retrieval_avg_ms"] = round(avg_retrieval, 3)
    results["retrieval_p95_ms"] = round(np.percentile(latencies, 95), 3)
    print(f"  Retrieval (k=5): avg={avg_retrieval:.3f}ms, p95={np.percentile(latencies, 95):.3f}ms")

    # Graph expansion benchmark
    gm = GraphMemory(dim=DIM)
    for i in range(20):
        gm.add_entity(f"sys-{i}", embed_fake(f"system {i}"), name=f"System {i}", entity_type="system")
    for i in range(19):
        gm.add_relation(f"sys-{i}", "connects_to", f"sys-{i+1}")

    t0 = time.perf_counter()
    for _ in range(100):
        gm.get_relations("sys-5")
        gm.get_incoming_relations("sys-5")
    graph_time = (time.perf_counter() - t0) * 1000 / 100
    results["graph_expansion_avg_ms"] = round(graph_time, 4)
    print(f"  Graph expansion: {graph_time:.4f}ms")

    # Path finding
    t0 = time.perf_counter()
    for _ in range(100):
        gm.get_path("sys-0", "sys-10")
    path_time = (time.perf_counter() - t0) * 1000 / 100
    results["path_finding_avg_ms"] = round(path_time, 4)
    print(f"  Path finding (10 hops): {path_time:.4f}ms")

    # Decay benchmark
    now = time.time()
    for state in mem._memory_state.values():
        state["last_accessed"] = now - 5 * 86400
    t0 = time.perf_counter()
    mem._apply_decay()
    decay_time = (time.perf_counter() - t0) * 1000
    results["decay_100_memories_ms"] = round(decay_time, 3)
    results["mean_importance_after_5d"] = round(mem.get_stats()["mean_importance"], 4)
    print(f"  Decay (100 memories, 5 days): {decay_time:.3f}ms")
    print(f"  Mean importance after decay: {mem.get_stats()['mean_importance']:.4f}")

    # Eviction benchmark
    mem2 = AgentMemory(dim=DIM, capacity=50)
    t0 = time.time()
    for i in range(100):
        vec = embed_fake(f"memory {i}")
        mem2.save_memory(vec, content=f"memory {i}", importance=i * 0.01)
    eviction_time = (time.time() - t0) * 1000
    results["insert_100_with_eviction_ms"] = round(eviction_time, 2)
    results["remaining_after_eviction"] = len(mem2)
    print(f"  Insert 100 into capacity=50: {eviction_time:.2f}ms, remaining={len(mem2)}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "agent_workload.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/agent_workload.json")


if __name__ == "__main__":
    main()
