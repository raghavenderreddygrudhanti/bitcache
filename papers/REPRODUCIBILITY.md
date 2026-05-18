# Reproducibility Guide

All experiments in the three papers can be reproduced with the following steps.

## Environment

```
Python: 3.10+
OS: macOS or Linux (arm64 or x86_64)
RAM: 16 GB minimum (32 GB for scale experiments)
GPU: Not required
```

## Dependencies

```bash
pip install numpy faiss-cpu sentence-transformers matplotlib scikit-learn
pip install -e .  # install bitcache in editable mode
```

## Dataset Generation

```bash
# Real sentence-transformer embeddings (100K, dim=384)
python -c "
from sentence_transformers import SentenceTransformer
import numpy as np, random, os
random.seed(42)
model = SentenceTransformer('all-MiniLM-L6-v2')
topics = ['machine learning', 'database systems', 'cloud computing', 'NLP',
          'computer vision', 'cybersecurity', 'web development', 'distributed systems',
          'data engineering', 'mobile development', 'blockchain', 'robotics',
          'quantum computing', 'devops', 'AI ethics', 'networking',
          'operating systems', 'software testing', 'game development', 'bioinformatics',
          'recommendation systems', 'search engines', 'compiler design', 'embedded systems',
          'signal processing', 'cryptography', 'parallel computing', 'information retrieval',
          'HCI', 'software architecture', 'API design', 'microservices',
          'containerization', 'serverless', 'edge computing', 'IoT',
          'data visualization', 'time series', 'anomaly detection', 'reinforcement learning']
verbs = ['introduction to', 'advanced', 'practical guide to', 'understanding',
         'best practices for', 'common mistakes in', 'future of', 'scaling',
         'debugging', 'optimizing', 'testing', 'deploying', 'monitoring',
         'comparing', 'building', 'designing', 'implementing', 'evaluating',
         'troubleshooting', 'migrating', 'securing', 'automating', 'benchmarking',
         'profiling', 'refactoring', 'documenting', 'maintaining']
contexts = ['in production', 'for startups', 'at scale', 'for enterprise',
            'with Python', 'with Rust', 'on AWS', 'on Kubernetes',
            'for beginners', 'for experts', 'in 2024', 'with open source',
            'using Docker', 'with CI/CD', 'for real-time systems', 'for batch processing']
sentences = []
while len(sentences) < 100000:
    sentences.append(f'{random.choice(verbs)} {random.choice(topics)} {random.choice(contexts)}')
embeddings = model.encode(sentences[:100000], batch_size=512).astype(np.float32)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
os.makedirs('benchmarks/data', exist_ok=True)
np.save('benchmarks/data/minilm_100k.npy', embeddings)
"
```

## Running Benchmarks

```bash
# Paper 1: Recall-vs-rf curve
python benchmarks/eval_rf_curve.py

# Paper 1: Scale experiments
python benchmarks/eval_scale.py

# Paper 1: 14-method comparison
python benchmarks/eval_all_dbs.py

# Paper 2: Realistic embeddings + FAISS baselines
python benchmarks/eval_realistic.py

# All results saved to benchmarks/results/ as JSON
```

## Running Tests

```bash
pytest tests/ -v
# Expected: 75 tests passing (68 Gen1 + 7 Gen3)
```

## Terminology

Consistent across all papers:
- **recall@10**: fraction of true top-10 present in predicted top-10
- **rf**: rerank factor (number of candidates = rf × k)
- **partition hit rate**: fraction of true top-k in probed partitions
- **latency**: average per-query time in milliseconds
- **QPS**: queries per second (1000 / avg_latency_ms)

## Citation

If referencing this work:
```
Grudhanti, R. R. (2026). bitcache: Tunable Staged Retrieval for
Persistent AI Memory Systems. https://github.com/raghavenderreddygrudhanti/bitcache
```
