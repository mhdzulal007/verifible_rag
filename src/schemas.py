from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    CONTRADICTION = "CONTRADICTION"
    UNGROUNDED_NEUTRAL = "UNGROUNDED_NEUTRAL"
    QUOTE_NOT_FOUND = "QUOTE_NOT_FOUND"
    ABSTAINED = "ABSTAINED"

class DocumentChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique deterministic or generated identifier for the chunk")
    text: str = Field(..., description="Contextualized text content of the chunk")
    source_file: str = Field(..., description="Original filename of the document")
    page_number: int = Field(default=1, description="Page number in source document (1-indexed)")
    section_header: str = Field(default="General", description="Section header or title if detected")
    bbox: Optional[List[float]] = Field(default=None, description="Bounding box [x0, y0, x1, y1] if available")

class Citation(BaseModel):
    chunk_id: str = Field(..., description="Exact ID of the retrieved chunk cited")
    source_file: str = Field(..., description="Source document filename")
    page_number: int = Field(..., description="Page number in source document")
    verbatim_quote: str = Field(..., description="Exact substring copied verbatim from the source chunk")

class VerifiedClaim(BaseModel):
    statement: str = Field(..., description="Atomic factual claim answering part of the query")
    citation: Citation = Field(..., description="Direct citation supporting the statement")
    status: VerificationStatus = Field(default=VerificationStatus.VERIFIED, description="Verification verdict")
    entailment_score: float = Field(default=0.0, description="NLI probability that quote entails claim")
    contradiction_score: float = Field(default=0.0, description="NLI probability that claim contradicts quote")
    neutral_score: float = Field(default=0.0, description="NLI probability that claim is unsupported/neutral")

class VerifiableRAGResponse(BaseModel):
    final_answer: str = Field(..., description="Synthesized verified answer or explicit abstention message")
    claims: List[VerifiedClaim] = Field(default_factory=list, description="List of atomic claims with verified citations")
    abstained: bool = Field(default=False, description="True if retrieval lacked sufficient evidence or verification failed")
    abstention_reason: Optional[str] = Field(default=None, description="Detailed diagnostic reason if abstained")
    citation_verified: bool = Field(default=False, description="True only if all claims passed physical & NLI verification")
    latency_ms: Optional[float] = Field(default=None, description="Total pipeline latency in milliseconds")
    model: str = Field(default="gemini-3.6-flash", description="Underlying generator model name")
