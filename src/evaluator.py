import json
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.retriever import HybridRetriever, get_embedder
from src.generator import VerifiableGenerator
from src.schemas import VerifiableRAGResponse

logger = logging.getLogger("evaluator")

class RAGEvaluator:
    """Rigorous evaluation suite assessing Groundedness, Retrieval Recall, Answer Similarity, and Abstention."""

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: VerifiableGenerator,
        embedder_model: str = "BAAI/bge-small-en-v1.5",
    ):
        self.retriever = retriever
        self.generator = generator
        self.embedder = get_embedder(embedder_model)

    def evaluate_benchmark(self, dataset_path: str, top_k: int = 5) -> Dict[str, Any]:
        """Runs full benchmark suite and computes statistical RAG metrics."""
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {dataset_path}")

        with open(path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)

        results = []
        latencies = []
        
        in_domain_total = 0
        in_domain_correct = 0
        adversarial_total = 0
        adversarial_blocked = 0
        
        faithfulness_scores = []
        context_recall_scores = []
        semantic_similarities = []

        print(f"\n🚀 Running Enterprise Evaluation Suite on {len(test_cases)} test cases...")
        print("=" * 70)

        for idx, item in enumerate(test_cases, 1):
            query = item["query"]
            should_abstain = item.get("should_abstain", False)
            expected_answer = item.get("expected_answer", "")
            concepts = item.get("ground_truth_concepts", [])

            t0 = time.time()
            chunks = self.retriever.search(query, top_k=top_k)
            response: VerifiableRAGResponse = self.generator.generate(query, chunks)
            elapsed = time.time() - t0
            latencies.append(elapsed)

            # 1. Retrieval Context Recall
            retrieved_text = " ".join([c.text.lower() for c in chunks])
            if concepts:
                matched_concepts = sum(1 for c in concepts if c.lower() in retrieved_text)
                recall = matched_concepts / len(concepts)
            else:
                recall = 1.0 if should_abstain else 0.0
            context_recall_scores.append(recall)

            # 2. Guardrail Abstention Performance
            if should_abstain:
                adversarial_total += 1
                blocked = response.abstained
                if blocked:
                    adversarial_blocked += 1
                    status_str = "✅ BLOCKED (Correct Abstention)"
                else:
                    status_str = "❌ FAILED (Hallucinated outside knowledge)"
            else:
                in_domain_total += 1
                if not response.abstained and response.citation_verified:
                    in_domain_correct += 1
                    status_str = "✅ VERIFIED ANSWER"
                else:
                    status_str = f"❌ FAILED ({response.abstention_reason})"

            # 3. Faithfulness & Semantic Similarity for In-Domain Queries
            if not should_abstain and not response.abstained and response.claims:
                # Entailment score of claims
                ent_scores = [c.entailment_score for c in response.claims]
                mean_ent = float(np.mean(ent_scores)) if ent_scores else 0.0
                faithfulness_scores.append(mean_ent)

                # Cosine similarity to expected ground-truth
                if expected_answer:
                    emb_pred = self.embedder.encode([response.final_answer], normalize_embeddings=True)[0]
                    emb_true = self.embedder.encode([expected_answer], normalize_embeddings=True)[0]
                    sim = float(np.dot(emb_pred, emb_true))
                    semantic_similarities.append(sim)

            print(f"[{idx:02d}/{len(test_cases):02d}] {status_str} | Latency: {elapsed:.2f}s")
            print(f"     Q: '{query}'")

            results.append({
                "id": item.get("id", str(idx)),
                "query": query,
                "should_abstain": should_abstain,
                "abstained": response.abstained,
                "abstention_reason": response.abstention_reason,
                "citation_verified": response.citation_verified,
                "latency_sec": elapsed,
                "context_recall": recall,
                "claims_count": len(response.claims),
            })

        # Calculate Aggregate Statistics
        adv_block_rate = (adversarial_blocked / adversarial_total * 100.0) if adversarial_total else 100.0
        in_domain_acc = (in_domain_correct / in_domain_total * 100.0) if in_domain_total else 100.0
        avg_faithfulness = (float(np.mean(faithfulness_scores)) * 100.0) if faithfulness_scores else 0.0
        avg_context_recall = (float(np.mean(context_recall_scores)) * 100.0) if context_recall_scores else 0.0
        avg_semantic_sim = (float(np.mean(semantic_similarities)) * 100.0) if semantic_similarities else 0.0

        p50_lat = float(np.percentile(latencies, 50))
        p90_lat = float(np.percentile(latencies, 90))
        p99_lat = float(np.percentile(latencies, 99))

        report = {
            "summary": {
                "total_test_cases": len(test_cases),
                "in_domain_accuracy_pct": round(in_domain_acc, 2),
                "adversarial_block_rate_pct": round(adv_block_rate, 2),
                "mean_faithfulness_pct": round(avg_faithfulness, 2),
                "mean_context_recall_pct": round(avg_context_recall, 2),
                "mean_semantic_similarity_pct": round(avg_semantic_sim, 2),
                "latency_p50_sec": round(p50_lat, 2),
                "latency_p90_sec": round(p90_lat, 2),
                "latency_p99_sec": round(p99_lat, 2),
            },
            "test_cases": results,
        }

        self._print_report(report["summary"])
        return report

    def _print_report(self, summary: Dict[str, Any]):
        print("\n" + "=" * 70)
        print("📊 ENTERPRISE VERIFIABLE RAG BENCHMARK REPORT")
        print("=" * 70)
        print(f"Total Test Cases Evaluated     : {summary['total_test_cases']}")
        print(f"Hallucination Block Rate (Adv) : {summary['adversarial_block_rate_pct']}%")
        print(f"In-Domain Verification Rate    : {summary['in_domain_accuracy_pct']}%")
        print(f"Context Recall (Ground Truth)  : {summary['mean_context_recall_pct']}%")
        print(f"NLI Faithfulness Entailment    : {summary['mean_faithfulness_pct']}%")
        print(f"Semantic Answer Similarity     : {summary['mean_semantic_similarity_pct']}%")
        print("-" * 70)
        print(f"Latency P50: {summary['latency_p50_sec']}s | P90: {summary['latency_p90_sec']}s | P99: {summary['latency_p99_sec']}s")
        print("=" * 70 + "\n")
