"""Generate real sentence-transformer embeddings for Rust benchmarks.

Downloads sentences from a standard dataset, embeds them with all-MiniLM-L6-v2,
and saves as numpy arrays for the Rust benchmark to load.

Output:
  data/real_embeddings.npy  — (99000, 384) float32 database vectors
  data/real_queries.npy     — (1000, 384) float32 query vectors
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer

def generate_sentences(n: int, seed: int = 42) -> list[str]:
    """Generate diverse sentences by combining templates with topics."""
    rng = np.random.default_rng(seed)

    topics = [
        "machine learning", "database optimization", "cloud computing",
        "neural networks", "distributed systems", "natural language processing",
        "computer vision", "reinforcement learning", "data engineering",
        "microservices", "kubernetes", "docker containers",
        "API design", "authentication", "caching strategies",
        "load balancing", "message queues", "event sourcing",
        "graph databases", "vector search", "embedding models",
        "transformer architecture", "attention mechanism", "fine-tuning",
        "transfer learning", "model compression", "quantization",
        "binary search", "hash tables", "B-trees",
        "network protocols", "TCP/IP", "HTTP/2",
        "encryption", "TLS certificates", "OAuth",
        "CI/CD pipelines", "testing strategies", "code review",
        "agile methodology", "sprint planning", "retrospectives",
        "user experience", "accessibility", "responsive design",
        "mobile development", "iOS", "Android",
        "web frameworks", "React", "Vue.js",
        "Python programming", "Rust language", "Go concurrency",
    ]

    templates = [
        "How to implement {} in production systems",
        "Best practices for {} at scale",
        "Understanding the fundamentals of {}",
        "Common mistakes when working with {}",
        "Advanced techniques in {}",
        "A beginner's guide to {}",
        "Performance optimization for {}",
        "Security considerations in {}",
        "Comparing different approaches to {}",
        "The future of {} in enterprise",
        "Debugging issues with {}",
        "Monitoring and observability for {}",
        "Cost optimization strategies for {}",
        "Team collaboration around {}",
        "Documentation best practices for {}",
        "Migration strategies for {}",
        "Scaling {} to millions of users",
        "Real-world case studies in {}",
        "Open source tools for {}",
        "Interview questions about {}",
    ]

    sentences = []
    for i in range(n):
        topic = topics[rng.integers(len(topics))]
        template = templates[rng.integers(len(templates))]
        # Add some variation
        suffix = f" (variant {i % 100})" if i > len(topics) * len(templates) else ""
        sentences.append(template.format(topic) + suffix)

    return sentences


def main():
    print("=" * 60)
    print("  Generating Real Sentence-Transformer Embeddings")
    print("=" * 60)

    n_db = 99_000
    n_queries = 1_000
    n_total = n_db + n_queries

    # Generate sentences
    print(f"\n  Generating {n_total} sentences...")
    sentences = generate_sentences(n_total, seed=42)
    print(f"  Sample: '{sentences[0]}'")
    print(f"  Sample: '{sentences[100]}'")

    # Load model
    print(f"\n  Loading all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Embed
    print(f"  Embedding {n_total} sentences (batch_size=512)...")
    embeddings = model.encode(sentences, batch_size=512, show_progress_bar=True,
                              normalize_embeddings=True)
    embeddings = embeddings.astype(np.float32)
    print(f"  Shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    # Split into db and queries
    db_vectors = embeddings[:n_db]
    query_vectors = embeddings[n_db:]

    # Save
    os.makedirs("data", exist_ok=True)
    np.save("data/real_embeddings.npy", db_vectors)
    np.save("data/real_queries.npy", query_vectors)

    print(f"\n  Saved:")
    print(f"    data/real_embeddings.npy  ({db_vectors.shape})")
    print(f"    data/real_queries.npy     ({query_vectors.shape})")

    # Quick sanity check
    print(f"\n  Sanity check:")
    print(f"    Norm of first vector: {np.linalg.norm(db_vectors[0]):.4f} (should be ~1.0)")
    print(f"    Mean inner product (same topic): ", end="")
    # Vectors 0 and 1000 should be from similar topics
    sims = db_vectors[:100] @ db_vectors[:100].T
    np.fill_diagonal(sims, 0)
    print(f"{sims.mean():.4f}")

    print("\n  Done! Run: cargo run --release --bin real_embeddings_bench")


if __name__ == "__main__":
    main()
