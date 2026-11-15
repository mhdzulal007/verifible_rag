import pytest
from src.schemas import (
    DocumentChunk,
    Citation,
    VerifiedClaim,
    VerifiableRAGResponse,
    VerificationStatus,
)

def test_document_chunk_creation():
    chunk = DocumentChunk(
        chunk_id="chunk-001",
        text="Form 10-K is an annual report required by the SEC.",
        source_file="sec_guidelines.pdf",
        page_number=3,
        section_header="General Rules",
        bbox=[10.0, 20.0, 100.0, 50.0],
    )
    assert chunk.chunk_id == "chunk-001"
    assert chunk.page_number == 3
    assert chunk.bbox == [10.0, 20.0, 100.0, 50.0]

def test_verified_claim_defaults():
    citation = Citation(
        chunk_id="chunk-001",
        source_file="sec_guidelines.pdf",
        page_number=3,
        verbatim_quote="Form 10-K is an annual report required by the SEC.",
    )
    claim = VerifiedClaim(
        statement="Form 10-K must be submitted annually to the SEC.",
        citation=citation,
        status=VerificationStatus.VERIFIED,
        entailment_score=0.95,
        contradiction_score=0.01,
        neutral_score=0.04,
    )
    assert claim.status == VerificationStatus.VERIFIED
    assert claim.entailment_score == 0.95
    assert claim.citation.chunk_id == "chunk-001"

def test_verifiable_rag_response_serialization():
    resp = VerifiableRAGResponse(
        final_answer="Form 10-K is filed annually.",
        claims=[],
        abstained=False,
        citation_verified=True,
        latency_ms=120.5,
    )
    data = resp.model_dump()
    assert data["final_answer"] == "Form 10-K is filed annually."
    assert data["citation_verified"] is True
    assert data["latency_ms"] == 120.5
