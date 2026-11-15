import os
import uuid
import logging
from pathlib import Path
from typing import List, Optional
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from src.schemas import DocumentChunk

logger = logging.getLogger("ingest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class DocumentIngestor:
    """Production-grade document ingestion pipeline powered by Docling."""

    def __init__(self, tokenizer_model: str = "BAAI/bge-small-en-v1.5", max_tokens: int = 512):
        self.converter = DocumentConverter()
        self.chunker = HybridChunker(tokenizer=tokenizer_model, max_tokens=max_tokens)

    def ingest_directory(self, pdf_dir: str) -> List[DocumentChunk]:
        """Processes all PDFs in a given directory with isolated fault tolerance."""
        dir_path = Path(pdf_dir)
        if not dir_path.exists():
            logger.warning(f"Ingestion directory does not exist: {pdf_dir}")
            return []

        pdf_paths = sorted(list(dir_path.glob("*.pdf")))
        if not pdf_paths:
            logger.warning(f"No PDF documents found in: {pdf_dir}")
            return []

        all_chunks: List[DocumentChunk] = []
        for pdf_path in pdf_paths:
            chunks = self.ingest_file(str(pdf_path))
            all_chunks.extend(chunks)

        logger.info(f"Ingestion complete for {pdf_dir}. Generated {len(all_chunks)} total chunks.")
        return all_chunks

    def ingest_file(self, file_path: str) -> List[DocumentChunk]:
        """Converts and chunks an individual PDF document with safe metadata extraction."""
        path = Path(file_path)
        logger.info(f"Parsing document layout: {path.name}...")
        chunks: List[DocumentChunk] = []

        try:
            result = self.converter.convert(str(path))
            doc = result.document

            for idx, chunk in enumerate(self.chunker.chunk(doc)):
                # Safe page number extraction
                page_no = 1
                bbox_coords = None
                if hasattr(chunk, "meta") and chunk.meta:
                    if getattr(chunk.meta, "doc_items", None):
                        for item in chunk.meta.doc_items:
                            if getattr(item, "prov", None) and len(item.prov) > 0:
                                prov = item.prov[0]
                                if hasattr(prov, "page_no"):
                                    page_no = prov.page_no
                                if hasattr(prov, "bbox") and prov.bbox:
                                    try:
                                        # Docling BoundingBox
                                        bbox_coords = [
                                            float(prov.bbox.l),
                                            float(prov.bbox.t),
                                            float(prov.bbox.r),
                                            float(prov.bbox.b)
                                        ]
                                    except Exception:
                                        pass
                                break

                # Safe section heading extraction
                heading = "General"
                if hasattr(chunk, "meta") and chunk.meta and getattr(chunk.meta, "headings", None):
                    if len(chunk.meta.headings) > 0 and chunk.meta.headings[0]:
                        heading = str(chunk.meta.headings[0])

                # Contextualized text preserves table headers and parent hierarchy
                try:
                    chunk_text = self.chunker.contextualize(chunk)
                except Exception:
                    chunk_text = getattr(chunk, "text", str(chunk))

                chunk_id = f"{path.stem[:6]}-p{page_no}-{idx}-{uuid.uuid4().hex[:6]}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text.strip(),
                        source_file=path.name,
                        page_number=page_no,
                        section_header=heading,
                        bbox=bbox_coords,
                    )
                )
            logger.info(f"Successfully processed {path.name}: {len(chunks)} chunks.")
        except Exception as e:
            logger.error(f"Failed to process document {path.name}: {str(e)}", exc_info=True)

        return chunks

# Functional backward-compatibility helper
_default_ingestor: Optional[DocumentIngestor] = None

def ingest_pdfs(pdf_dir: str) -> List[DocumentChunk]:
    global _default_ingestor
    if _default_ingestor is None:
        _default_ingestor = DocumentIngestor()
    return _default_ingestor.ingest_directory(pdf_dir)
