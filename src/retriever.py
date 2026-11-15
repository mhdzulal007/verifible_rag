import os
import threading
import logging
import numpy as np
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.schemas import DocumentChunk

logger = logging.getLogger("retriever")

# Global model cache to avoid re-allocating weights in RAM/VRAM on each session
_MODEL_CACHE: Dict[str, object] = {}
_MODEL_LOCK = threading.Lock()

def get_embedder(model_name: str = "BAAI/bge-small-en-v1.5") -> SentenceTransformer:
    with _MODEL_LOCK:
        if model_name not in _MODEL_CACHE:
            logger.info(f"Loading embedding model: {model_name}")
            _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        return _MODEL_CACHE[model_name]

def get_reranker(model_name: str = "BAAI/bge-reranker-base") -> CrossEncoder:
    with _MODEL_LOCK:
        if model_name not in _MODEL_CACHE:
            logger.info(f"Loading cross-encoder reranker: {model_name}")
            _MODEL_CACHE[model_name] = CrossEncoder(model_name)
        return _MODEL_CACHE[model_name]

class HybridRetriever:
    """Thread-safe hybrid retriever combining Dense vector search, Sparse BM25, and Cross-Encoder reranking."""

    def __init__(
        self,
        chunks: List[DocumentChunk],
        collection_name: str = "tech_docs",
        client: Optional[QdrantClient] = None,
        embedder_model: str = "BAAI/bge-small-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-base",
    ):
        self.chunks = chunks
        self.chunk_map: Dict[str, DocumentChunk] = {c.chunk_id: c for c in chunks}
        self.collection = collection_name
        self._lock = threading.Lock()

        # Shared singleton models
        self.embedder = get_embedder(embedder_model)
        self.reranker = get_reranker(reranker_model)

        # Initialize Qdrant Client (Networked URL if set, otherwise memory-safe client)
        if client is not None:
            self.qdrant = client
        else:
            qdrant_url = os.getenv("QDRANT_URL")
            if qdrant_url:
                logger.info(f"Connecting to networked Qdrant instance: {qdrant_url}")
                self.qdrant = QdrantClient(url=qdrant_url)
            else:
                logger.info("Initializing in-memory isolated Qdrant vector store")
                self.qdrant = QdrantClient(":memory:")

        # Initialize Sparse BM25
        if self.chunks:
            tokenized_corpus = [c.text.lower().split() for c in self.chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self._index_dense_vectors()
        else:
            self.bm25 = None
            logger.warning("HybridRetriever initialized with zero document chunks.")

    def _index_dense_vectors(self):
        """Indexes dense embeddings into Qdrant safely."""
        with self._lock:
            # Recreate collection to prevent stale vector artifacts
            try:
                if self.qdrant.collection_exists(self.collection):
                    self.qdrant.delete_collection(self.collection)
            except Exception:
                pass

            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

            texts = [c.text for c in self.chunks]
            embeddings = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

            points = [
                PointStruct(
                    id=idx,
                    vector=emb.tolist(),
                    payload={
                        "chunk_id": c.chunk_id,
                        "source_file": c.source_file,
                        "page_number": c.page_number,
                        "section_header": c.section_header,
                    },
                )
                for idx, (c, emb) in enumerate(zip(self.chunks, embeddings))
            ]
            self.qdrant.upsert(collection_name=self.collection, points=points)
            logger.info(f"Indexed {len(points)} points into Qdrant collection '{self.collection}'.")

    def search(self, query: str, candidate_k: int = 20, top_k: int = 5) -> List[DocumentChunk]:
        """Executes Hybrid Dense + Sparse search with Reciprocal Rank Fusion and Cross-Encoder Reranking."""
        if not self.chunks or self.bm25 is None:
            return []

        clean_query = query.strip()
        if not clean_query:
            return []

        effective_k = min(candidate_k, len(self.chunks))

        # A. Dense Vector Retrieval
        q_vec = self.embedder.encode([clean_query], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
        try:
            dense_hits = self.qdrant.query_points(
                collection_name=self.collection,
                query=q_vec,
                limit=effective_k,
            ).points
            dense_ids = [h.payload["chunk_id"] for h in dense_hits]
        except Exception as e:
            logger.error(f"Dense retrieval error: {e}")
            dense_ids = []

        # B. Sparse BM25 Keyword Retrieval
        tokenized_query = clean_query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:effective_k]
        sparse_ids = [self.chunks[i].chunk_id for i in top_bm25_idx if bm25_scores[i] > 0]

        # C. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        k_const = 60.0

        for rank, cid in enumerate(dense_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_const + rank + 1))

        for rank, cid in enumerate(sparse_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_const + rank + 1))

        fused_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:effective_k]
        candidates = [self.chunk_map[cid] for cid in fused_ids if cid in self.chunk_map]

        if not candidates:
            return []

        # D. Cross-Encoder Reranking
        pairs = [[clean_query, c.text] for c in candidates]
        try:
            rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)
            ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
            return [c for c, _ in ranked[:top_k]]
        except Exception as e:
            logger.error(f"Reranking error, falling back to RRF ordering: {e}")
            return candidates[:top_k]

    def close(self):
        """Cleanly releases database handles if needed."""
        try:
            self.qdrant.close()
        except Exception:
            pass
