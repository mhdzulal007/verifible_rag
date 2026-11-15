import json
import os
from pathlib import Path
from src.ingest import DocumentIngestor
from src.retriever import HybridRetriever
from src.generator import VerifiableGenerator
from src.evaluator import RAGEvaluator

def main():
    print("=" * 70)
    print("⚡ ENTERPRISE VERIFIABLE RAG BENCHMARK RUNNER")
    print("=" * 70)

    # 1. Ingest baseline test documents
    data_dir = "data/raw_pdfs"
    if not os.path.exists(data_dir) or not list(Path(data_dir).glob("*.pdf")):
        print(f"❌ Error: No PDFs found in {data_dir}. Please place evaluation PDFs there.")
        return

    print("📄 Ingesting evaluation PDFs...")
    ingestor = DocumentIngestor()
    chunks = ingestor.ingest_directory(data_dir)
    print(f"✅ Ingestion complete: {len(chunks)} chunks created.")

    # 2. Initialize Retriever & Generator
    print("🔍 Initializing Hybrid Retriever...")
    retriever = HybridRetriever(chunks=chunks, collection_name="benchmark_eval")
    
    print("🧠 Initializing Verifiable Generator (Gemini + DeBERTa NLI Guardrail)...")
    generator = VerifiableGenerator(model_name="gemini-3.6-flash")

    # 3. Run Benchmark
    benchmark_file = "data/evaluation_benchmark.json"
    evaluator = RAGEvaluator(retriever=retriever, generator=generator)
    report = evaluator.evaluate_benchmark(benchmark_file, top_k=5)

    # 4. Save Report
    output_path = Path("data/evaluation_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"📁 Benchmark report successfully written to: {output_path.resolve()}")

if __name__ == "__main__":
    main()
