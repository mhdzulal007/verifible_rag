import os
import time
import uuid
import shutil
import logging
import threading
from pathlib import Path
from typing import Dict, Optional, Any
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.ingest import DocumentIngestor
from src.retriever import HybridRetriever
from src.generator import VerifiableGenerator
from src.schemas import VerifiableRAGResponse

load_dotenv()
logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

BASE_UPLOAD_DIR = Path("data/uploads")
BASE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Verifiable RAG Enterprise API",
    version="2.0.0",
    description="Zero-hallucination document intelligence engine with NLI semantic verification.",
)

# Enable CORS for web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str = Field(..., description="User query to answer from ingested documents")
    session_id: str = Field(default="default", description="Tenant or session isolation identifier")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top chunks to present to generator")

class JobStatus(BaseModel):
    job_id: str
    session_id: str
    status: str
    progress: int = 0
    message: str = ""
    created_at: float
    updated_at: float

class SessionManager:
    """Thread-safe state manager for isolated sessions, retrievers, and background jobs."""

    def __init__(self):
        self._lock = threading.Lock()
        self.retrievers: Dict[str, HybridRetriever] = {}
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.ingestor = DocumentIngestor()
        self.generator = VerifiableGenerator(model_name="gemini-3.6-flash")

    def create_job(self, session_id: str) -> str:
        job_id = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            self.jobs[job_id] = {
                "job_id": job_id,
                "session_id": session_id,
                "status": "QUEUED",
                "progress": 5,
                "message": "Job queued for ingestion.",
                "created_at": now,
                "updated_at": now,
            }
        return job_id

    def update_job(self, job_id: str, status: str, progress: int, message: str):
        with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id]["status"] = status
                self.jobs[job_id]["progress"] = progress
                self.jobs[job_id]["message"] = message
                self.jobs[job_id]["updated_at"] = time.time()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.jobs.get(job_id)

    def set_retriever(self, session_id: str, retriever: HybridRetriever):
        with self._lock:
            old = self.retrievers.get(session_id)
            if old:
                old.close()
            self.retrievers[session_id] = retriever

    def get_retriever(self, session_id: str) -> Optional[HybridRetriever]:
        with self._lock:
            return self.retrievers.get(session_id)

session_manager = SessionManager()

def run_document_ingestion_task(job_id: str, session_id: str, session_dir: Path):
    """Executes asynchronous document conversion, chunking, and index construction safely."""
    try:
        session_manager.update_job(job_id, "PROCESSING", 20, "Parsing layout and extracting semantic elements...")
        chunks = session_manager.ingestor.ingest_directory(str(session_dir))

        if not chunks:
            session_manager.update_job(job_id, "FAILED", 100, "No valid text or chunks extracted from document(s).")
            return

        session_manager.update_job(job_id, "INDEXING", 60, "Generating embeddings and building hybrid search index...")
        collection_name = f"session_{session_id.replace('-', '_')}"
        retriever = HybridRetriever(chunks=chunks, collection_name=collection_name)

        session_manager.set_retriever(session_id, retriever)
        session_manager.update_job(job_id, "COMPLETED", 100, f"Index ready! Total chunks: {len(chunks)}.")
        logger.info(f"Ingestion job {job_id} for session {session_id} successfully completed.")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        session_manager.update_job(job_id, "FAILED", 100, f"Ingestion error: {str(e)}")

# --- API ENDPOINTS ---

@app.post("/api/v1/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    """Uploads a PDF into a tenant-isolated session and triggers async ingestion."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    effective_session = session_id.strip() if session_id else str(uuid.uuid4())
    session_dir = BASE_UPLOAD_DIR / effective_session
    session_dir.mkdir(parents=True, exist_ok=True)

    dest_path = session_dir / file.filename
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist uploaded file: {e}")

    job_id = session_manager.create_job(effective_session)
    background_tasks.add_task(run_document_ingestion_task, job_id, effective_session, session_dir)

    return {
        "status": "ACCEPTED",
        "job_id": job_id,
        "session_id": effective_session,
        "filename": file.filename,
        "message": "File received. Asynchronous indexing in progress.",
    }

@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Polls status and progress of an ingestion job."""
    job = session_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return JobStatus(**job)

@app.post("/api/v1/chat", response_model=VerifiableRAGResponse)
async def chat(request: ChatRequest):
    """Performs hybrid retrieval and dual-stage verifiable generation."""
    retriever = session_manager.get_retriever(request.session_id)
    if not retriever:
        # Fallback to default session if exists
        retriever = session_manager.get_retriever("default")

    if not retriever:
        raise HTTPException(
            status_code=400,
            detail=f"No search index found for session '{request.session_id}'. Please upload a document first.",
        )

    try:
        chunks = retriever.search(request.query, top_k=request.top_k)
        response = session_manager.generator.generate(request.query, chunks)
        return response
    except Exception as e:
        logger.error(f"Chat generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    """System health check and diagnostic information."""
    return {
        "status": "HEALTHY",
        "timestamp": time.time(),
        "active_sessions": len(session_manager.retrievers),
        "total_jobs": len(session_manager.jobs),
    }

# --- BACKWARD COMPATIBILITY ALIASES ---
@app.post("/upload")
async def compat_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    return await upload_document(background_tasks=background_tasks, file=file, session_id="default")

@app.get("/status/{job_id}")
async def compat_status(job_id: str):
    job = await get_job_status(job_id)
    return {"job_id": job.job_id, "status": "Completed" if job.status == "COMPLETED" else ("Failed: " + job.message if job.status == "FAILED" else job.message)}

@app.post("/chat")
async def compat_chat(request: ChatRequest):
    res = await chat(request)
    return res.model_dump()

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Verifiable RAG Enterprise API on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
