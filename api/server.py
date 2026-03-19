"""FastAPI backend for AiWo Visual Document QA.

Endpoints:
  POST /query       — text query → retrieval + VLM answer
  GET  /health      — health check
  GET  /stats       — index statistics

Frontend sends text (from Whisper STT) to /query,
backend returns answer + retrieved page images.
"""

import os
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.retriever import DocumentRetriever
from generation.vlm_generator import VLMGenerator

# ── Config ──
INDEX_DIR = os.environ.get("INDEX_DIR", "data/index")
IMAGE_DIR = os.environ.get("IMAGE_DIR", "data/images")
MODEL_NAME = os.environ.get("MODEL_NAME", "nanovdr/NanoVDR-S-Multi")
VLM_DEPLOYMENT = os.environ.get("VLM_DEPLOYMENT", "gpt-4o")

# ── App ──
app = FastAPI(
    title="AiWo Visual Document QA",
    description="Visual document retrieval + VLM-powered QA for equipment manuals",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve page images as static files
if os.path.exists(IMAGE_DIR):
    app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# ── Models (lazy init) ──
retriever: DocumentRetriever = None
generator: VLMGenerator = None


@app.on_event("startup")
async def startup():
    global retriever, generator
    retriever = DocumentRetriever(model_name=MODEL_NAME, index_dir=INDEX_DIR)
    generator = VLMGenerator(deployment=VLM_DEPLOYMENT)


# ── Schemas ──
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    generate_answer: bool = True
    max_pages_for_vlm: int = 3


class RetrievedPage(BaseModel):
    page_id: str
    score: float
    image_url: str
    source: str
    page_number: int


class QueryResponse(BaseModel):
    query: str
    answer: str | None
    retrieved_pages: list[RetrievedPage]
    retrieval_latency_ms: float
    total_latency_ms: float


# ── Endpoints ──

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/stats")
async def stats():
    if retriever is None:
        raise HTTPException(503, "Retriever not loaded")
    return {
        "index_size": len(retriever.page_metadata),
        "embedding_dim": retriever.doc_embeddings.shape[1],
        "model": MODEL_NAME,
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if retriever is None:
        raise HTTPException(503, "Retriever not loaded")

    t0 = time.time()

    # Retrieve relevant pages
    pages, retrieval_latency = retriever.retrieve(req.query, top_k=req.top_k)

    # Build response pages with image URLs
    retrieved_pages = []
    for p in pages:
        img_filename = p.get("image_path", "")
        image_url = f"/images/{img_filename}" if img_filename else ""
        retrieved_pages.append(RetrievedPage(
            page_id=p["page_id"],
            score=p["score"],
            image_url=image_url,
            source=p.get("source", ""),
            page_number=p.get("page_number", 0),
        ))

    # Generate answer with VLM (optional)
    answer = None
    if req.generate_answer and generator is not None:
        try:
            answer = generator.generate(
                query=req.query,
                retrieved_pages=pages,
                image_dir=IMAGE_DIR,
                max_pages=req.max_pages_for_vlm,
            )
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"

    total_latency = time.time() - t0

    return QueryResponse(
        query=req.query,
        answer=answer,
        retrieved_pages=retrieved_pages,
        retrieval_latency_ms=retrieval_latency * 1000,
        total_latency_ms=total_latency * 1000,
    )
