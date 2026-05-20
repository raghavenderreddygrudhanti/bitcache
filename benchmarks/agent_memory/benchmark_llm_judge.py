"""
Agent Memory Benchmark with LLM-as-Judge (OpenAI GPT-4o-mini)

Expanded benchmark: 500 turns, evolving knowledge, LLM judges correctness.
Compares: No memory, Chat history, Vector DB, Bitcache.
"""

import time
import json
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from typing import List, Dict

client = OpenAI()
embed_model = None  # lazy load


def get_embed_model():
    global embed_model
    if embed_model is None:
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embed_model


# ─── Generate Conversations with GPT ─────────────────────────────────────────

def generate_conversations():
    """Generate 100 realistic agent conversations with evolving knowledge."""
    print("  Generating conversations with GPT-4o-mini...")

    prompt = """Generate 100 statements that a user would tell an IT support AI agent over 60 days.
Include:
- Infrastructure details (databases, caches, servers)
- Team preferences (formatting, communication)
- Incidents and fixes
- Migrations and upgrades (things that CHANGE over time)
- Scale information (traffic, users)

Format as JSON array. Each item has:
- "day": number 1-60
- "text": what the user said
- "category": one of "infra", "preference", "incident", "fix", "scale", "deploy", "ops"
- "replaces": index of an earlier statement this makes outdated (or null)

Make it realistic. Include 15-20 items where information changes (migrations, upgrades).
Return ONLY valid JSON, no markdown."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4000,
    )

    try:
        conversations = json.loads(response.choices[0].message.content)
        print(f"  Generated {len(conversations)} conversations")
        return conversations
    except json.JSONDecodeError:
        # Fallback to hardcoded if GPT output isn't valid JSON
        print("  GPT output wasn't valid JSON, using fallback data")
        return get_fallback_conversations()


def generate_questions(conversations):
    """Generate test questions based on the conversations."""
    print("  Generating test questions...")

    conv_summary = json.dumps(conversations[:50], indent=1)  # first 50 for context

    prompt = f"""Given these conversations between a user and an IT support agent:
{conv_summary}

Generate 50 test questions that would test whether the agent remembers correctly.
Include:
- 20 questions about CURRENT state (what's true now)
- 15 questions about things that CHANGED (should return new info, not old)
- 10 questions about preferences and procedures
- 5 questions about incidents/fixes

For each question, specify:
- "day": the day the question is asked (should be AFTER relevant info was stored)
- "question": what the user asks
- "expected_answer": what a correct answer should mention (key phrase)
- "wrong_answer": what an outdated/incorrect answer would mention (or null)

Return ONLY valid JSON array."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=3000,
    )

    try:
        questions = json.loads(response.choices[0].message.content)
        print(f"  Generated {len(questions)} questions")
        return questions
    except json.JSONDecodeError:
        print("  Using fallback questions")
        return get_fallback_questions()


def llm_judge(question: str, expected: str, retrieved_context: List[str]) -> Dict:
    """Use GPT-4o-mini to judge if the retrieved context answers the question correctly."""
    context_str = "\n".join(f"- {c}" for c in retrieved_context) if retrieved_context else "(no context retrieved)"

    prompt = f"""You are judging whether an AI agent's memory retrieval is correct.

Question: "{question}"
Expected answer should mention: "{expected}"

Retrieved context from memory:
{context_str}

Judge:
1. "correct": true if the retrieved context contains information that would help answer the question correctly
2. "stale": true if the context contains OUTDATED information that would lead to a wrong answer
3. "reason": one sentence explaining your judgment

Return JSON only: {{"correct": bool, "stale": bool, "reason": "..."}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150,
        )
        return json.loads(response.choices[0].message.content)
    except:
        return {"correct": False, "stale": False, "reason": "judge error"}


# ─── Memory Systems ───────────────────────────────────────────────────────────

class NoMemory:
    def store(self, text, day, importance=0.5): pass
    def retrieve(self, query, k=5, current_day=0): return []
    def name(self): return "No Memory"

class ChatHistory:
    def __init__(self, window=20):
        self.window = window
        self.history = []
    def store(self, text, day, importance=0.5):
        self.history.append(text)
        if len(self.history) > self.window: self.history.pop(0)
    def retrieve(self, query, k=5, current_day=0):
        return self.history[-k:]
    def name(self): return f"Chat History ({self.window})"

class VectorDB:
    def __init__(self):
        self.texts, self.embeddings = [], []
    def store(self, text, day, importance=0.5):
        emb = get_embed_model().encode([text], normalize_embeddings=True)[0]
        self.texts.append(text)
        self.embeddings.append(emb)
    def retrieve(self, query, k=5, current_day=0):
        if not self.embeddings: return []
        q = get_embed_model().encode([query], normalize_embeddings=True)[0]
        sims = np.dot(self.embeddings, q)
        top = np.argsort(sims)[::-1][:k]
        return [self.texts[i] for i in top]
    def name(self): return "Vector DB"

class Bitcache:
    def __init__(self, capacity=30, decay_rate=0.03):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.memories = []
    def store(self, text, day, importance=0.5):
        emb = get_embed_model().encode([text], normalize_embeddings=True)[0]
        self.memories.append({"text": text, "emb": emb, "importance": importance, "last_access": day, "day": day})
        while len(self.memories) > self.capacity:
            worst = min(range(len(self.memories)), key=lambda i: self.memories[i]["importance"])
            self.memories.pop(worst)
    def retrieve(self, query, k=5, current_day=0):
        if not self.memories: return []
        for m in self.memories:
            days = max(0, current_day - m["last_access"])
            m["importance"] = max(0.0, m["importance"] - self.decay_rate * days)
        q = get_embed_model().encode([query], normalize_embeddings=True)[0]
        scored = []
        for m in self.memories:
            sim = float(np.dot(m["emb"], q))
            score = sim * (0.5 + 0.5 * m["importance"])
            scored.append((score, m))
        scored.sort(key=lambda x: -x[0])
        results = []
        for _, m in scored[:k]:
            m["last_access"] = current_day
            m["importance"] = min(1.0, m["importance"] + 0.1)
            results.append(m["text"])
        return results
    def name(self): return f"Bitcache (cap={self.capacity})"


# ─── Fallback Data ────────────────────────────────────────────────────────────

def get_fallback_conversations():
    return [
        {"day": 1, "text": "We run PostgreSQL 15 on AWS RDS", "category": "infra", "replaces": None},
        {"day": 2, "text": "Deploy via GitHub Actions to ECS", "category": "deploy", "replaces": None},
        {"day": 3, "text": "Redis cluster for caching", "category": "infra", "replaces": None},
        {"day": 5, "text": "SLA is 99.9% uptime", "category": "ops", "replaces": None},
        {"day": 8, "text": "Connection pool exhaustion incident", "category": "incident", "replaces": None},
        {"day": 10, "text": "I prefer bullet point summaries", "category": "preference", "replaces": None},
        {"day": 12, "text": "Monitoring with Datadog", "category": "ops", "replaces": None},
        {"day": 15, "text": "50K requests per minute at peak", "category": "scale", "replaces": None},
        {"day": 20, "text": "Frontend is Next.js on Vercel", "category": "infra", "replaces": None},
        {"day": 25, "text": "Auth handled by Auth0", "category": "infra", "replaces": None},
        {"day": 30, "text": "New payments service in Go", "category": "infra", "replaces": None},
        {"day": 35, "text": "Switched from PagerDuty to Opsgenie", "category": "ops", "replaces": None},
        {"day": 40, "text": "Started Aurora migration", "category": "infra", "replaces": None},
        {"day": 45, "text": "Migration done. Now on Aurora Serverless v2", "category": "infra", "replaces": 0},
        {"day": 50, "text": "Traffic grew to 80K rpm after launch", "category": "scale", "replaces": 7},
        {"day": 55, "text": "Added CloudFront CDN", "category": "infra", "replaces": None},
        {"day": 58, "text": "Moved to ElastiCache Serverless from Redis", "category": "infra", "replaces": 2},
        {"day": 60, "text": "Upgraded to Terraform 1.7", "category": "deploy", "replaces": None},
    ]

def get_fallback_questions():
    return [
        {"day": 10, "question": "What database do we use?", "expected_answer": "PostgreSQL", "wrong_answer": None},
        {"day": 47, "question": "What database do we use?", "expected_answer": "Aurora", "wrong_answer": "PostgreSQL"},
        {"day": 55, "question": "What's our traffic volume?", "expected_answer": "80K", "wrong_answer": "50K"},
        {"day": 60, "question": "What caching do we use?", "expected_answer": "ElastiCache", "wrong_answer": "Redis"},
        {"day": 60, "question": "Who handles on-call?", "expected_answer": "Opsgenie", "wrong_answer": "PagerDuty"},
        {"day": 60, "question": "How should I format responses?", "expected_answer": "bullet", "wrong_answer": None},
        {"day": 60, "question": "What's our SLA?", "expected_answer": "99.9", "wrong_answer": None},
        {"day": 60, "question": "What CDN do we use?", "expected_answer": "CloudFront", "wrong_answer": None},
        {"day": 60, "question": "What monitoring tool?", "expected_answer": "Datadog", "wrong_answer": None},
        {"day": 60, "question": "What auth system?", "expected_answer": "Auth0", "wrong_answer": None},
    ]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  AGENT MEMORY BENCHMARK (LLM-Judged)")
    print("  GPT-4o-mini generates data + judges correctness")
    print("=" * 65)

    # Generate or use fallback
    try:
        conversations = generate_conversations()
        questions = generate_questions(conversations)
    except Exception as e:
        print(f"  GPT generation failed ({e}), using fallback")
        conversations = get_fallback_conversations()
        questions = get_fallback_questions()

    systems = [
        NoMemory(),
        ChatHistory(20),
        VectorDB(),
        Bitcache(capacity=30, decay_rate=0.03),
        Bitcache(capacity=50, decay_rate=0.02),
    ]

    print(f"\n  Conversations: {len(conversations)}")
    print(f"  Questions: {len(questions)}")
    print(f"  Systems: {len(systems)}")

    all_results = []

    for sys in systems:
        print(f"\n  Testing: {sys.name()}...")

        # Store conversations
        for conv in conversations:
            importance = 0.8 if conv.get("category") in ("infra", "preference", "sla") else 0.5
            sys.store(conv["text"], conv["day"], importance)

        # Answer questions with LLM judge
        correct = 0
        stale = 0
        retrieve_times = []

        for q in questions:
            t = time.time()
            retrieved = sys.retrieve(q["question"], k=5, current_day=q["day"])
            retrieve_times.append(time.time() - t)

            judgment = llm_judge(q["question"], q["expected_answer"], retrieved)
            if judgment.get("correct"): correct += 1
            if judgment.get("stale"): stale += 1

        accuracy = correct / len(questions)
        staleness = stale / len(questions)
        avg_latency = np.mean(retrieve_times) * 1000

        result = {
            "system": sys.name(),
            "accuracy": accuracy,
            "staleness": staleness,
            "avg_latency_ms": avg_latency,
            "correct": correct,
            "stale": stale,
            "total": len(questions),
        }
        all_results.append(result)
        print(f"    Accuracy: {accuracy:.1%} | Staleness: {staleness:.1%} | Latency: {avg_latency:.1f}ms")

    # Final table
    print("\n" + "=" * 65)
    print(f"  {'System':<25} {'Accuracy':>9} {'Staleness':>10} {'Latency':>9}")
    print("─" * 65)
    for r in all_results:
        print(f"  {r['system']:<25} {r['accuracy']:>8.1%} {r['staleness']:>9.1%} {r['avg_latency_ms']:>7.1f}ms")
    print("=" * 65)

    with open("benchmarks/agent_memory/results_llm_judge.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n  Saved to benchmarks/agent_memory/results_llm_judge.json")


if __name__ == "__main__":
    main()
