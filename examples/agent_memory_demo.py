"""End-to-end agent memory demo.

Shows the full bitcache workflow:
1. Insert memories with importance scores
2. Retrieve by semantic similarity
3. Expand context via graph relations
4. Reinforce accessed memories
5. Decay unused memories over time
6. Evict low-importance memories at capacity

Run: python examples/agent_memory_demo.py
Requires: pip install sentence-transformers
"""

import numpy as np
import time

# Use sentence-transformers for real embeddings
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    DIM = 384
    def embed(text):
        return model.encode(text).astype(np.float32)
except ImportError:
    print("sentence-transformers not installed. Using random vectors.")
    DIM = 384
    def embed(text):
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(DIM).astype(np.float32)

from bitcache import AgentMemory
from bitcache.graph_memory import GraphMemory


def main():
    print("=" * 60)
    print("bitcache Agent Memory Demo")
    print("=" * 60)

    # --- Agent Memory ---
    print("\n[1] Creating agent memory (capacity=20, decay_rate=0.1)")
    mem = AgentMemory(dim=DIM, capacity=20, decay_rate=0.1, reinforce_amount=0.15)

    # Insert operational knowledge
    memories = [
        ("Database connection pool exhausted on prod-db-01", 0.9),
        ("Restart connection pool via admin console port 8080", 0.7),
        ("prod-db-01 was migrated to new hardware last Tuesday", 0.5),
        ("API gateway depends on prod-db-01 for auth tokens", 0.6),
        ("Customer reported slow response times on checkout page", 0.8),
        ("Log rotation runs at 2am daily on all servers", 0.3),
    ]

    print("\n[2] Inserting memories:")
    for content, importance in memories:
        vec = embed(content)
        mid = mem.save_memory(vec, content=content, importance=importance)
        print(f"  [{importance:.1f}] {content[:50]}...")

    # Retrieve
    print("\n[3] Retrieving: 'database connection problem'")
    query = embed("database connection problem")
    results = mem.retrieve_memory(query, k=3)
    for r in results:
        print(f"  [{r['importance']:.2f}] {r['content'][:60]}")

    # Show reinforcement
    print("\n[4] After retrieval — accessed memories reinforced:")
    print(f"  Stats: {mem.get_stats()}")

    # --- Graph Memory ---
    print("\n[5] Building system dependency graph:")
    gm = GraphMemory(dim=DIM)

    systems = {
        "prod-db-01": "Production Database",
        "api-gateway": "API Gateway",
        "checkout-service": "Checkout Service",
        "log-server": "Log Server",
    }
    for sys_id, name in systems.items():
        gm.add_entity(sys_id, embed(f"system: {name}"), name=name, entity_type="system")

    gm.add_relation("api-gateway", "depends_on", "prod-db-01")
    gm.add_relation("checkout-service", "depends_on", "api-gateway")
    gm.add_relation("checkout-service", "logs_to", "log-server")

    print(f"  Entities: {gm.num_entities}, Relations: {gm.num_relations}")

    # Graph expansion
    print("\n[6] Graph expansion from prod-db-01:")
    print(f"  Incoming: {[f'{r[\"source_name\"]} --{r[\"relation\"]}->' for r in gm.get_incoming_relations('prod-db-01')]}")

    path = gm.get_path("checkout-service", "prod-db-01")
    if path:
        print(f"  Path checkout-service → prod-db-01: {[p['entity'] for p in path]}")

    # Decay simulation
    print("\n[7] Simulating 7 days of inactivity:")
    now = time.time()
    for state in mem._memory_state.values():
        state["last_accessed"] = now - 7 * 86400
    mem._apply_decay()
    print(f"  Stats after decay: {mem.get_stats()}")

    print("\n" + "=" * 60)
    print("Demo complete. All layers working together.")
    print("=" * 60)


if __name__ == "__main__":
    main()
