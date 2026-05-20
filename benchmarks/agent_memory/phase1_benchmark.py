"""
Phase 1 Benchmark Suite: Contradiction, Forgetting, Confidence, Ablation
=========================================================================

Tests the 4 most critical memory quality dimensions:
1. Contradiction Robustness — can it detect varied phrasings of the same change?
2. Intelligent Forgetting — does it keep critical, drop trivial?
3. Confidence Calibration — does it say "I don't know" correctly?
4. Ablation Studies — does each feature contribute value?

No LLM calls. All scoring via string matching + embedding similarity.
"""

import time
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
sys.path.insert(0, '.')
from benchmarks.agent_memory.bitcache_smart import BitcacheSmart

model = SentenceTransformer("all-MiniLM-L6-v2")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: CONTRADICTION ROBUSTNESS
# Can the system detect semantically varied contradictions?
# ═══════════════════════════════════════════════════════════════════════════════

def test_contradiction_robustness():
    print("\n" + "─" * 60)
    print("  TEST 1: CONTRADICTION ROBUSTNESS")
    print("─" * 60)

    # Pairs: (original fact, contradiction phrasing)
    contradiction_pairs = [
        # Direct replacement
        ("We use Redis for caching", "We switched from Redis to Memcached"),
        ("Our database is PostgreSQL", "We migrated from PostgreSQL to Aurora"),
        ("Alerting via PagerDuty", "Moved alerting from PagerDuty to Opsgenie"),
        ("I work at Google", "I left Google and joined Microsoft"),
        # Indirect / paraphrased
        ("We use Redis for caching", "Redis was deprecated, now using DragonflyDB"),
        ("Our database is PostgreSQL", "PostgreSQL is decommissioned"),
        ("I work at Google", "Started my new role at Amazon last week"),
        # Subtle
        ("We use Redis for caching", "Cache layer migrated away from Redis"),
        ("Our database is PostgreSQL", "The old Postgres instance was shut down"),
        ("Deploy via Jenkins", "Jenkins replaced by GitHub Actions"),
        # Very different phrasing
        ("We use Redis for caching", "No longer using Redis"),
        ("I work at Google", "My new employer is Stripe"),
        ("Monitoring with Datadog", "Datadog contract ended, switched to Grafana"),
    ]

    mem = BitcacheSmart(model, capacity=50, decay_rate=0.01, confidence_threshold=0.2)

    detected = 0
    missed = 0
    false_positives = 0

    for original, contradiction in contradiction_pairs:
        # Store original
        mem_fresh = BitcacheSmart(model, capacity=50, decay_rate=0.01, confidence_threshold=0.2)
        mem_fresh.store(original, day=1)
        mem_fresh.store("I prefer bullet-point answers", day=1)  # unrelated control
        mem_fresh.store("My daughter is in school", day=1)  # unrelated control

        # Store contradiction
        mem_fresh.store(contradiction, day=30)

        # Check: did the original get its importance zeroed?
        original_survived = False
        for m in mem_fresh.memories:
            if m['text'] == original and m['importance'] > 0.1:
                original_survived = True

        if not original_survived:
            detected += 1
        else:
            missed += 1

        # Check: did unrelated memories survive? (no false positives)
        for m in mem_fresh.memories:
            if "bullet-point" in m['text'] and m['importance'] < 0.1:
                false_positives += 1
            if "daughter" in m['text'] and m['importance'] < 0.1:
                false_positives += 1

    total = len(contradiction_pairs)
    precision = detected / max(detected + false_positives, 1)
    recall = detected / total

    print(f"  Detected: {detected}/{total} ({recall*100:.0f}%)")
    print(f"  Missed: {missed}/{total}")
    print(f"  False positives: {false_positives}")
    print(f"  Precision: {precision*100:.0f}%")
    print(f"  Recall: {recall*100:.0f}%")

    return {"precision": precision, "recall": recall, "detected": detected, "total": total}


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: INTELLIGENT FORGETTING
# Does it keep critical memories and drop trivial ones?
# ═══════════════════════════════════════════════════════════════════════════════

def test_intelligent_forgetting():
    print("\n" + "─" * 60)
    print("  TEST 2: INTELLIGENT FORGETTING")
    print("─" * 60)

    # Memory types with expected survival
    memories = [
        # Critical (MUST survive)
        {"text": "I am allergic to peanuts", "type": "critical", "should_survive": True},
        {"text": "My name is Arjun", "type": "critical", "should_survive": True},
        {"text": "Our SLA is 99.9% uptime", "type": "critical", "should_survive": True},
        {"text": "I prefer bullet-point answers always", "type": "critical", "should_survive": True},
        {"text": "Emergency contact: wife Priya at 555-0123", "type": "critical", "should_survive": True},
        # Important (should mostly survive)
        {"text": "We use PostgreSQL for our main database", "type": "important", "should_survive": True},
        {"text": "Deploy via GitHub Actions to ECS", "type": "important", "should_survive": True},
        {"text": "Monitoring with Datadog", "type": "important", "should_survive": True},
        # Trivial (should be evicted)
        {"text": "Had pizza for lunch today", "type": "trivial", "should_survive": False},
        {"text": "Weather is sunny today", "type": "trivial", "should_survive": False},
        {"text": "Grabbed coffee from Starbucks", "type": "trivial", "should_survive": False},
        {"text": "The elevator was slow today", "type": "trivial", "should_survive": False},
        {"text": "Boring meeting about office supplies", "type": "trivial", "should_survive": False},
        {"text": "Nothing special happened today", "type": "trivial", "should_survive": False},
        {"text": "Just finished a boring call", "type": "trivial", "should_survive": False},
        {"text": "Had sushi for lunch, it was decent", "type": "trivial", "should_survive": False},
    ]

    # Capacity = 8 (forces eviction of ~half)
    mem = BitcacheSmart(model, capacity=8, decay_rate=0.01, confidence_threshold=0.2)

    for i, m in enumerate(memories):
        mem.store(m["text"], day=i + 1)

    # Check what survived
    survived_texts = [m['text'] for m in mem.memories]

    critical_survived = 0
    critical_total = 0
    trivial_evicted = 0
    trivial_total = 0

    for m in memories:
        if m["type"] == "critical":
            critical_total += 1
            if m["text"] in survived_texts:
                critical_survived += 1
        elif m["type"] == "trivial":
            trivial_total += 1
            if m["text"] not in survived_texts:
                trivial_evicted += 1

    critical_rate = critical_survived / max(critical_total, 1)
    trivial_rate = trivial_evicted / max(trivial_total, 1)

    print(f"  Critical survival: {critical_survived}/{critical_total} ({critical_rate*100:.0f}%)")
    print(f"  Trivial eviction: {trivial_evicted}/{trivial_total} ({trivial_rate*100:.0f}%)")
    print(f"  Memory size: {len(mem.memories)} (capacity=8)")
    print(f"  Surviving memories:")
    for m in mem.memories:
        print(f"    [{m['importance']:.2f}] {m['text'][:50]}")

    return {"critical_survival": critical_rate, "trivial_eviction": trivial_rate}


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: CONFIDENCE CALIBRATION
# Does it say "I don't know" when it should?
# ═══════════════════════════════════════════════════════════════════════════════

def test_confidence_calibration():
    print("\n" + "─" * 60)
    print("  TEST 3: CONFIDENCE CALIBRATION")
    print("─" * 60)

    mem = BitcacheSmart(model, capacity=50, decay_rate=0.01, confidence_threshold=0.30)

    # Store some known facts
    known_facts = [
        "I am allergic to peanuts",
        "We use PostgreSQL for our database",
        "I work at Google as a DevOps engineer",
        "I prefer bullet-point answers",
        "Our monitoring is Datadog",
    ]
    for i, fact in enumerate(known_facts):
        mem.store(fact, day=i + 1)

    # Questions the system SHOULD answer (known)
    known_queries = [
        ("Am I allergic to anything?", "peanuts"),
        ("What database do we use?", "PostgreSQL"),
        ("Where do I work?", "Google"),
        ("How should you format answers?", "bullet"),
    ]

    # Questions the system should NOT answer (unknown)
    unknown_queries = [
        "What is my blood type?",
        "What is my shoe size?",
        "What car do I drive?",
        "What is my favorite movie?",
        "What is my phone number?",
        "What programming language do I prefer?",
    ]

    # Test known queries
    known_correct = 0
    for query, expected in known_queries:
        ret = mem.retrieve(query, k=3, current_day=10)
        ret_text = " ".join(ret).lower()
        if expected.lower() in ret_text:
            known_correct += 1

    # Test unknown queries
    unknown_rejected = 0
    for query in unknown_queries:
        ret = mem.retrieve(query, k=3, current_day=10)
        if not ret:  # correctly returned nothing
            unknown_rejected += 1

    known_accuracy = known_correct / len(known_queries)
    rejection_accuracy = unknown_rejected / len(unknown_queries)
    hallucination_rate = 1.0 - rejection_accuracy

    print(f"  Known queries answered: {known_correct}/{len(known_queries)} ({known_accuracy*100:.0f}%)")
    print(f"  Unknown queries rejected: {unknown_rejected}/{len(unknown_queries)} ({rejection_accuracy*100:.0f}%)")
    print(f"  Hallucination rate: {hallucination_rate*100:.0f}%")

    return {
        "known_accuracy": known_accuracy,
        "rejection_accuracy": rejection_accuracy,
        "hallucination_rate": hallucination_rate,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: ABLATION STUDIES
# Does each feature contribute value?
# ═══════════════════════════════════════════════════════════════════════════════

def test_ablation():
    print("\n" + "─" * 60)
    print("  TEST 4: ABLATION STUDIES")
    print("─" * 60)

    # Standard test data
    memories_data = [
        (1, "I am allergic to peanuts"),
        (2, "We use PostgreSQL for our database"),
        (3, "Had pizza for lunch today"),
        (4, "Weather is sunny"),
        (5, "I prefer bullet-point answers"),
        (10, "Grabbed coffee from Starbucks"),
        (15, "The elevator was slow"),
        (20, "Our monitoring is Datadog"),
        (30, "We migrated from PostgreSQL to Aurora"),
        (40, "Had sushi for lunch, decent"),
    ]

    questions = [
        ("What database?", "Aurora", "current"),
        ("Am I allergic?", "peanuts", "preference"),
        ("What did I eat?", "FORGET", "trivial"),
        ("What is my blood type?", "UNKNOWN", "confidence"),
    ]

    def run_config(capacity, decay, contradiction_thresh, confidence_thresh, name):
        mem = BitcacheSmart(model, capacity=capacity, decay_rate=decay,
                          confidence_threshold=confidence_thresh)
        # Override contradiction threshold
        mem.contradiction_threshold = contradiction_thresh

        for day, text in memories_data:
            mem.store(text, day=day)

        correct = 0
        for query, expected, qtype in questions:
            ret = mem.retrieve(query, k=3, current_day=60)
            ret_text = " ".join(ret).lower()

            if expected == "FORGET":
                if not any(w in ret_text for w in ["pizza", "sushi", "coffee", "elevator", "sunny"]):
                    correct += 1
            elif expected == "UNKNOWN":
                if not ret:
                    correct += 1
            else:
                if expected.lower() in ret_text:
                    correct += 1

        return correct, len(questions)

    configs = [
        # Full system
        (6, 0.01, 0.50, 0.25, "Full Bitcache"),
        # No decay (decay=0)
        (6, 0.0, 0.50, 0.25, "No decay"),
        # No contradiction detection (threshold=1.0, never triggers)
        (6, 0.01, 1.0, 0.25, "No contradiction"),
        # No confidence threshold (threshold=0, always returns)
        (6, 0.01, 0.50, 0.0, "No confidence"),
        # No eviction (capacity=100, never evicts)
        (100, 0.01, 0.50, 0.25, "No eviction"),
        # Nothing (plain vector search)
        (100, 0.0, 1.0, 0.0, "Plain vector DB"),
    ]

    print(f"  {'Config':<25} {'Score':>8} {'Result':>10}")
    print(f"  {'-'*45}")

    results = {}
    for cap, decay, contra, conf, name in configs:
        correct, total = run_config(cap, decay, contra, conf, name)
        pct = correct / total * 100
        results[name] = pct
        print(f"  {name:<25} {correct}/{total}    ({pct:.0f}%)")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  PHASE 1 BENCHMARK SUITE")
    print("  Contradiction | Forgetting | Confidence | Ablation")
    print("=" * 60)

    r1 = test_contradiction_robustness()
    r2 = test_intelligent_forgetting()
    r3 = test_confidence_calibration()
    r4 = test_ablation()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Contradiction detection:  {r1['recall']*100:.0f}% recall, {r1['precision']*100:.0f}% precision")
    print(f"  Intelligent forgetting:   {r2['critical_survival']*100:.0f}% critical kept, {r2['trivial_eviction']*100:.0f}% trivial dropped")
    print(f"  Confidence calibration:   {r3['known_accuracy']*100:.0f}% known correct, {r3['rejection_accuracy']*100:.0f}% unknown rejected")
    print(f"  Ablation (full system):   {r4.get('Full Bitcache', 0):.0f}% vs {r4.get('Plain vector DB', 0):.0f}% (plain)")
    print("=" * 60)
