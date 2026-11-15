import pytest
from src.schemas import (
    DocumentChunk,
    Citation,
    VerifiedClaim,
    VerifiableRAGResponse,
    VerificationStatus,
)
from src.generator import VerifiableGenerator

@pytest.fixture
def sample_chunk():
    return DocumentChunk(
        chunk_id="chunk-101",
        text="A large accelerated filer must file Form 10-K within 60 days after fiscal year end.",
        source_file="sec_rules.pdf",
        page_number=4,
    )

@pytest.fixture
def generator():
    return VerifiableGenerator()

def test_verification_valid_entailment(generator, sample_chunk):
    claim = VerifiedClaim(
        statement="Large accelerated filers are required to submit Form 10-K within 60 days.",
        citation=Citation(
            chunk_id="chunk-101",
            source_file="sec_rules.pdf",
            page_number=4,
            verbatim_quote="A large accelerated filer must file Form 10-K within 60 days after fiscal year end.",
        ),
    )
    resp = VerifiableRAGResponse(
        final_answer="Large accelerated filers have 60 days to file.",
        claims=[claim],
        abstained=False,
    )

    verified_resp = generator._verify_claims(resp, [sample_chunk])
    assert verified_resp.citation_verified is True
    assert verified_resp.abstained is False
    assert verified_resp.claims[0].status == VerificationStatus.VERIFIED
    assert verified_resp.claims[0].entailment_score >= 0.60

def test_verification_blocks_ungrounded_quote(generator, sample_chunk):
    # Quote that doesn't exist in the chunk
    claim = VerifiedClaim(
        statement="Companies must pay a $50,000 late fee.",
        citation=Citation(
            chunk_id="chunk-101",
            source_file="sec_rules.pdf",
            page_number=4,
            verbatim_quote="Companies must pay a $50,000 late fee to the regulatory commission.",
        ),
    )
    resp = VerifiableRAGResponse(
        final_answer="Companies must pay a $50,000 fee.",
        claims=[claim],
        abstained=False,
    )

    verified_resp = generator._verify_claims(resp, [sample_chunk])
    assert verified_resp.citation_verified is False
    assert verified_resp.abstained is True
    assert verified_resp.claims[0].status == VerificationStatus.QUOTE_NOT_FOUND

def test_verification_blocks_contradiction(generator, sample_chunk):
    claim = VerifiedClaim(
        statement="Large accelerated filers have 120 days to file Form 10-K.",
        citation=Citation(
            chunk_id="chunk-101",
            source_file="sec_rules.pdf",
            page_number=4,
            verbatim_quote="A large accelerated filer must file Form 10-K within 60 days after fiscal year end.",
        ),
    )
    resp = VerifiableRAGResponse(
        final_answer="They have 120 days to file.",
        claims=[claim],
        abstained=False,
    )

    verified_resp = generator._verify_claims(resp, [sample_chunk])
    assert verified_resp.citation_verified is False
    assert verified_resp.abstained is True
    assert verified_resp.claims[0].status == VerificationStatus.CONTRADICTION

def test_verification_blocks_neutral_hallucination(generator):
    chunk = DocumentChunk(
        chunk_id="chunk-202",
        text="The company filed its annual report on Monday morning.",
        source_file="company_filing.pdf",
        page_number=1,
    )
    # The statement is neutral relative to the premise (neither entailed nor contradicted)
    claim = VerifiedClaim(
        statement="The report was signed by the general counsel.",
        citation=Citation(
            chunk_id="chunk-202",
            source_file="company_filing.pdf",
            page_number=1,
            verbatim_quote="The company filed its annual report on Monday morning.",
        ),
    )
    resp = VerifiableRAGResponse(
        final_answer="The general counsel signed the report.",
        claims=[claim],
        abstained=False,
    )

    verified_resp = generator._verify_claims(resp, [chunk])
    assert verified_resp.citation_verified is False
    assert verified_resp.abstained is True
    assert verified_resp.claims[0].status == VerificationStatus.UNGROUNDED_NEUTRAL
