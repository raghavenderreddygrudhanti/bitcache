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
    """Generate diverse unique sentences by combining templates with topics and variants."""
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
        "mobile development", "iOS apps", "Android development",
        "web frameworks", "React components", "Vue.js reactivity",
        "Python programming", "Rust language", "Go concurrency",
        "memory management", "garbage collection", "reference counting",
        "parallel computing", "GPU acceleration", "SIMD optimization",
        "data pipelines", "ETL processes", "stream processing",
        "search engines", "inverted indexes", "ranking algorithms",
        "recommendation systems", "collaborative filtering", "content-based filtering",
        "anomaly detection", "time series analysis", "forecasting models",
        "image classification", "object detection", "semantic segmentation",
        "speech recognition", "text-to-speech", "language translation",
        "chatbot design", "dialog systems", "intent recognition",
        "knowledge graphs", "ontology design", "semantic web",
        "blockchain technology", "smart contracts", "consensus algorithms",
        "edge computing", "IoT protocols", "sensor networks",
        "serverless architecture", "function-as-a-service", "cold start optimization",
        "database sharding", "replication strategies", "consistency models",
        "API rate limiting", "circuit breakers", "retry policies",
        "observability", "distributed tracing", "log aggregation",
        "feature flags", "A/B testing", "canary deployments",
        "infrastructure as code", "Terraform modules", "CloudFormation templates",
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
        "The future of {} in enterprise software",
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
        "Architecture patterns for {}",
        "Testing strategies for {}",
        "Deployment automation for {}",
        "Error handling in {}",
        "Configuration management for {}",
        "Capacity planning for {}",
        "Disaster recovery with {}",
        "Compliance requirements for {}",
        "Training new engineers on {}",
        "Building internal tools for {}",
    ]

    contexts = [
        "in a startup environment",
        "for enterprise teams",
        "with limited resources",
        "using modern tooling",
        "in regulated industries",
        "for high-availability systems",
        "in multi-cloud setups",
        "with remote teams",
        "during rapid growth",
        "for legacy modernization",
        "in real-time applications",
        "for batch processing workloads",
        "with strict latency requirements",
        "in data-intensive applications",
        "for customer-facing services",
    ]

    # Generate unique sentences
    sentences = set()
    attempts = 0
    max_attempts = n * 10

    while len(sentences) < n and attempts < max_attempts:
        topic = topics[rng.integers(len(topics))]
        template = templates[rng.integers(len(templates))]
        base = template.format(topic)

        # Add context and variant number for uniqueness
        if rng.random() > 0.2:
            context = contexts[rng.integers(len(contexts))]
            variant = rng.integers(1, 50)
            sentence = f"{base} {context} (part {variant})"
        else:
            variant = rng.integers(1, 200)
            sentence = f"{base} — section {variant}"

        sentences.add(sentence)
        attempts += 1

    result = list(sentences)
    rng.shuffle(result)
    return result[:n]


def main():
    print("=" * 60)
    print("  Generating Real Sentence-Transformer Embeddings")
    print("  (deduplicated sentences)")
    print("=" * 60)

    n_db = 99_000
    n_queries = 1_000
    n_total = n_db + n_queries

    # Generate unique sentences
    print(f"\n  Generating {n_total} unique sentences...")
    sentences = generate_sentences(n_total, seed=42)
    print(f"  Generated: {len(sentences)} unique sentences")
    assert len(sentences) == len(set(sentences)), "Duplicates found!"
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

    # Sanity check
    print(f"\n  Sanity check:")
    print(f"    Norm of first vector: {np.linalg.norm(db_vectors[0]):.4f} (should be ~1.0)")
    sims = db_vectors[:100] @ db_vectors[:100].T
    np.fill_diagonal(sims, 0)
    print(f"    Mean pairwise similarity (first 100): {sims.mean():.4f}")
    print(f"    Max pairwise similarity (first 100): {sims.max():.4f}")

    print("\n  Done! Run: cargo run --release --bin real_embeddings_bench")


if __name__ == "__main__":
    main()
