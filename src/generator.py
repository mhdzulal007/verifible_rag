import os
import re
import json
import time
import logging
import threading
import numpy as np
from typing import List, Dict, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import CrossEncoder
from rapidfuzz import fuzz
from src.schemas import (
    DocumentChunk,
    VerifiableRAGResponse,
    VerifiedClaim,
    Citation,
    VerificationStatus,
)

load_dotenv()
logger = logging.getLogger("generator")

# Global singleton cache for NLI model
_NLI_CACHE: Dict[str, CrossEncoder] = {}
_NLI_LOCK = threading.Lock()

def get_nli_model(model_name: str = "cross-encoder/nli-deberta-v3-small") -> CrossEncoder:
    with _NLI_LOCK:
        if model_name not in _NLI_CACHE:
            logger.info(f"Loading NLI semantic verification model: {model_name}")
            _NLI_CACHE[model_name] = CrossEncoder(model_name)
        return _NLI_CACHE[model_name]

SYSTEM_PROMPT = """You are a strictly grounded technical research assistant.
Answer the user's question using ONLY the provided context chunks.

CRITICAL RULES:
1. Every factual statement in your answer must be backed by a VerifiedClaim object.
2. In each claim citation, you MUST include:
   - chunk_id: The exact chunk_id from the context header where you found the information.
   - source_file: The file name from the context header.
   - page_number: The page number from the context header.
   - verbatim_quote: An EXACT, word-for-word substring copied directly from that chunk.
3. If the context does not contain sufficient facts to answer the question, you MUST set abstained=true, provide an abstention_reason, and set final_answer="I don't know based on the provided documents."
4. Never assume, extrapolate, or use outside knowledge.
"""

class VerifiableGenerator:
    """Production-grade generator with dual-stage physical and semantic NLI verification."""

    def __init__(
        self,
        model_name: str = "gemini-3.6-flash",
        nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
        entailment_threshold: float = 0.60,
        contradiction_threshold: float = 0.20,
        verbatim_fuzzy_threshold: float = 88.0,
    ):
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.verbatim_fuzzy_threshold = verbatim_fuzzy_threshold

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY is not set in environment or .env file.")

        self.client = genai.Client(api_key=api_key)
        self.nli_model = get_nli_model(nli_model_name)

    def generate(self, query: str, retrieved_chunks: List[DocumentChunk]) -> VerifiableRAGResponse:
        """Generates an answer using Gemini and executes dual-stage verification."""
        start_time = time.time()

        if not retrieved_chunks:
            return VerifiableRAGResponse(
                final_answer="I don't know based on the provided documents.",
                claims=[],
                abstained=True,
                abstention_reason="No relevant context chunks were retrieved.",
                citation_verified=True,
                latency_ms=(time.time() - start_time) * 1000,
                model=self.model_name,
            )

        # 1. Format Context with explicit chunk identifiers
        context_blocks = []
        for c in retrieved_chunks:
            context_blocks.append(
                f"[chunk_id={c.chunk_id} | file={c.source_file} | page={c.page_number} | section={c.section_header}]\n{c.text}"
            )
        context_str = "\n\n".join(context_blocks)

        prompt = f"Context:\n{context_str}\n\nQuestion: {query}"

        # 2. Resilient LLM Inference with Retry Backoff
        raw_response = self._call_gemini_with_retry(prompt)
        if raw_response is None:
            return VerifiableRAGResponse(
                final_answer="I don't know based on the provided documents.",
                claims=[],
                abstained=True,
                abstention_reason="LLM inference failed or connection timed out after retries.",
                citation_verified=False,
                latency_ms=(time.time() - start_time) * 1000,
                model=self.model_name,
            )

        # 3. Parse JSON safely
        response = self._parse_llm_json(raw_response)
        response.model = self.model_name

        # 4. If LLM explicitly abstained, honor it immediately
        if response.abstained:
            response.citation_verified = True
            response.latency_ms = (time.time() - start_time) * 1000
            return response

        # 5. Dual-Stage Verification Guardrail
        verified_response = self._verify_claims(response, retrieved_chunks)
        verified_response.latency_ms = (time.time() - start_time) * 1000
        return verified_response

    def _call_gemini_with_retry(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Calls Gemini API with exponential backoff on transient errors."""
        for attempt in range(1, max_retries + 1):
            try:
                result = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=VerifiableRAGResponse,
                    ),
                )
                if result and result.text:
                    return result.text
            except Exception as e:
                logger.warning(f"Gemini API attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(1.5 * attempt)
        return None

    def _parse_llm_json(self, raw_text: str) -> VerifiableRAGResponse:
        """Parses LLM JSON text defensively, stripping markdown formatting if present."""
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n", "", text)
            text = re.sub(r"\n```$", "", text)
        try:
            data = json.loads(text)
            return VerifiableRAGResponse(**data)
        except Exception as e:
            logger.error(f"Failed to parse LLM structured output as JSON: {e}")
            return VerifiableRAGResponse(
                final_answer="I don't know based on the provided documents.",
                claims=[],
                abstained=True,
                abstention_reason=f"LLM JSON schema parsing error: {e}",
                citation_verified=False,
            )

    def _verify_claims(self, response: VerifiableRAGResponse, chunks: List[DocumentChunk]) -> VerifiableRAGResponse:
        """Executes physical verbatim substring checks and NLI semantic entailment verification."""
        chunk_lookup = {c.chunk_id: c for c in chunks}

        # If answer is not empty but no claims were provided
        if not response.claims:
            if not response.abstained and len(response.final_answer.strip()) > 30:
                response.abstained = True
                response.abstention_reason = "Answer contained factual assertions with zero verifiable citations."
                response.final_answer = "I don't know based on the provided documents."
                response.citation_verified = False
            return response

        all_claims_verified = True
        verification_failures = []

        for claim in response.claims:
            chunk = chunk_lookup.get(claim.citation.chunk_id)

            # --- STAGE 1: PHYSICAL GROUNDING (Verbatim Substring Check) ---
            if chunk is None:
                claim.status = VerificationStatus.QUOTE_NOT_FOUND
                all_claims_verified = False
                verification_failures.append(f"Chunk ID '{claim.citation.chunk_id}' was not found in retrieved set.")
                continue

            quote = claim.citation.verbatim_quote.strip()
            source_text = chunk.text.strip()

            # Normalized substring or fuzzy match to absorb OCR hyphenation and whitespace
            exact_match = quote.lower() in source_text.lower()
            fuzzy_ratio = fuzz.partial_ratio(quote.lower(), source_text.lower())

            if not exact_match and fuzzy_ratio < self.verbatim_fuzzy_threshold:
                claim.status = VerificationStatus.QUOTE_NOT_FOUND
                all_claims_verified = False
                verification_failures.append(
                    f"Citation quote '{quote[:50]}...' was not found in source chunk (similarity {fuzzy_ratio:.1f}%)."
                )
                continue

            # --- STAGE 2: SEMANTIC NLI ENTAILMENT VERIFICATION ---
            premise = quote
            hypothesis = claim.statement.strip()

            try:
                # Returns logits [contradiction, entailment, neutral]
                scores = self.nli_model.predict([(premise, hypothesis)], show_progress_bar=False)
                logits = scores[0]
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)

                prob_contradiction = float(probs[0])
                prob_entailment = float(probs[1])
                prob_neutral = float(probs[2])

                claim.contradiction_score = prob_contradiction
                claim.entailment_score = prob_entailment
                claim.neutral_score = prob_neutral

                # Strict Guardrail Thresholding
                if prob_contradiction >= self.contradiction_threshold:
                    claim.status = VerificationStatus.CONTRADICTION
                    all_claims_verified = False
                    verification_failures.append(
                        f"Claim contradicted citation quote: P(Contradiction)={prob_contradiction:.2f} >= {self.contradiction_threshold}"
                    )
                elif prob_entailment < self.entailment_threshold:
                    claim.status = VerificationStatus.UNGROUNDED_NEUTRAL
                    all_claims_verified = False
                    verification_failures.append(
                        f"Claim not entailed by citation quote: P(Entailment)={prob_entailment:.2f} < {self.entailment_threshold} (Neutral={prob_neutral:.2f})"
                    )
                else:
                    claim.status = VerificationStatus.VERIFIED

            except Exception as e:
                logger.error(f"NLI model inference error: {e}")
                claim.status = VerificationStatus.UNGROUNDED_NEUTRAL
                all_claims_verified = False
                verification_failures.append(f"NLI model evaluation failed: {e}")

        # If any claim failed either stage, quarantine the entire response
        if not all_claims_verified:
            response.abstained = True
            response.citation_verified = False
            response.abstention_reason = "Semantic Verification Failed: " + "; ".join(verification_failures[:2])
            response.final_answer = "I don't know based on the provided documents."
        else:
            response.citation_verified = True
            response.abstained = False

        return response
