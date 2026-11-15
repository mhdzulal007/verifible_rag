import pytest
from src.schemas import DocumentChunk
from src.retriever import HybridRetriever

@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id="chunk-1",
            text="Large accelerated filers must file Form 10-K within 60 days after fiscal year end.",
            source_file="filing_rules.pdf",
            page_number=1,
            section_header="Deadlines",
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            text="Accelerated filers have a deadline of 75 days to submit Form 10-K to the SEC.",
            source_file="filing_rules.pdf",
            page_number=2,
            section_header="Deadlines",
        ),
        DocumentChunk(
            chunk_id="chunk-3",
            text="Non-accelerated filers must file within 90 days after fiscal year end.",
            source_file="filing_rules.pdf",
            page_number=3,
            section_header="Deadlines",
        ),
    ]

def test_hybrid_retriever_initialization_and_search(sample_chunks):
    retriever = HybridRetriever(chunks=sample_chunks, collection_name="test_retrieval")
    results = retriever.search("How many days does a large accelerated filer have?", top_k=2)

    assert len(results) > 0
    assert results[0].chunk_id == "chunk-1"
    assert "60 days" in results[0].text
    retriever.close()

def test_hybrid_retriever_empty_handling():
    retriever = HybridRetriever(chunks=[], collection_name="test_empty")
    results = retriever.search("Any query", top_k=5)
    assert results == []
    retriever.close()
