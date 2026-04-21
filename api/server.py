"""FastAPI backend for AiWo Visual Document QA.

Endpoints:
  POST /query        — legacy: text query → retrieval + VLM answer
  POST /chat         — TUNI-compatible: {messages, provider, model} → {result}
  GET  /v1/models    — TUNI provider discovery
  GET  /health       — health check
  GET  /stats        — index statistics

The answer-generation backend (Azure OpenAI GPT-4o / Anthropic Claude / local
Qwen2.5-VL) is selected at startup via the VLM_BACKEND env var — see
`generation/factory.py`.

Conversation state design (/chat):
  - The endpoint itself is stateless.
  - Each turn runs an LLM router ("RETRIEVE" vs "FOLLOWUP") over the recent
    transcript to decide whether to hit the retriever again.
  - RETRIEVE turns: fresh retrieval + append a visible markdown footer
    listing the pages (links to /aiwo-images/*.png). That footer doubles as
    the page-memory for the next turn.
  - FOLLOWUP turns: parse page filenames out of the most-recent assistant
    message's footer and reuse them, skipping retrieval.
"""

import json
import os
import re
import sys
import time
from urllib.parse import quote, unquote

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Auto-load .env from the repo root so operators don't have to `export` secrets
# into their shell (which leaks into bash history + `ps`). Pre-existing process
# env vars win, so shell overrides still work for ad-hoc debugging.
try:
    from dotenv import load_dotenv
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(dotenv_path=os.path.join(_REPO_ROOT, ".env"), override=False)
except ImportError:
    pass  # python-dotenv optional; falls back to shell env

from retrieval.retriever import DocumentRetriever
from generation.factory import build_generator

# ── Config ──
INDEX_DIR = os.environ.get("INDEX_DIR", "data/index")
IMAGE_DIR = os.environ.get("IMAGE_DIR", "data/images")
MODEL_NAME = os.environ.get("MODEL_NAME", "nanovdr/NanoVDR-S-Multi")
RAG_MODEL_ID = "aiwo-rag"
# URL prefix the frontend uses to reach the backend's /images/ via a Next.js
# rewrite — keeping it frontend-origin avoids CORS + SSH-tunnel-port issues.
PAGE_URL_PREFIX = os.environ.get("PAGE_URL_PREFIX", "/aiwo-images")

# ── App ──
app = FastAPI(
    title="AiWo Visual Document QA",
    description="Visual document retrieval + VLM-powered QA for equipment manuals",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(IMAGE_DIR):
    app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# ── State (lazy init) ──
retriever: DocumentRetriever = None
generator = None
generator_backend: str = "unknown"


@app.on_event("startup")
async def startup():
    global retriever, generator, generator_backend
    retriever = DocumentRetriever(model_name=MODEL_NAME, index_dir=INDEX_DIR)
    generator, generator_backend = build_generator()
    print(f"Generation backend: {generator_backend}")


# ── Schemas (legacy /query) ──
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


# ── Schemas (TUNI /chat) ──
class ChatMessageIn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessageIn]
    provider: str | None = None
    model: str | None = None
    top_k: int = 5
    max_pages_for_vlm: int = 3
    # Escape hatches for debugging / future UI toggles
    force_retrieval: bool | None = None


class ChatResultMessage(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    result: ChatResultMessage
    retrieved_pages: list[RetrievedPage] = []
    retrieval_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    retrieved: bool = True
    router_decision: str = ""


# ── Helpers ──
def _retrieved_to_pages(pages: list[dict]) -> list[RetrievedPage]:
    out: list[RetrievedPage] = []
    for p in pages:
        img_filename = p.get("image_path", "")
        image_url = f"{PAGE_URL_PREFIX}/{quote(img_filename)}" if img_filename else ""
        out.append(RetrievedPage(
            page_id=p["page_id"],
            score=float(p.get("score", 0.0)),
            image_url=image_url,
            source=p.get("source", ""),
            page_number=int(p.get("page_number", 0)),
        ))
    return out


def _last_user_message(messages: list[ChatMessageIn]) -> str:
    for m in reversed(messages):
        if m.role == "user" and m.content.strip():
            return m.content
    return ""


# ── Router: RETRIEVE vs FOLLOWUP ──
ROUTER_PROMPT = """You are a query classifier for a field operator's equipment-manual assistant.

Decide whether the operator's LATEST user message requires looking up NEW information \
from the equipment manual, or whether it is a clarification / follow-up / chit-chat \
about content ALREADY DISCUSSED in this conversation.

Prefer FOLLOWUP whenever the user is referring to something already said earlier, \
including: "step N", "that / this / it", "again", "explain", "more detail", \
"thanks", simple acknowledgements, or asking about the same component just discussed.

Prefer RETRIEVE only when the user asks about a DIFFERENT component, task, or \
specification that the previous assistant message did not already address.

Examples (the prior turn was about replacing the hydraulic filter):
- "Thanks!"                               → FOLLOWUP
- "Can you explain step 4 again?"         → FOLLOWUP
- "What does 'green silicone end' mean?"  → FOLLOWUP
- "How often should this be done?"        → RETRIEVE   (spec not yet stated)
- "What about the air filter instead?"    → RETRIEVE   (different component)
- "How do I change the engine oil?"       → RETRIEVE   (different task)

Conversation so far (oldest first):
{transcript}

Respond with EXACTLY ONE word: RETRIEVE or FOLLOWUP.
"""


def _transcript(messages: list[ChatMessageIn], max_chars_per_msg: int = 400) -> str:
    lines = []
    for m in messages[-8:]:
        body = m.content.strip().replace("\n", " ")
        if len(body) > max_chars_per_msg:
            body = body[:max_chars_per_msg] + "…"
        lines.append(f"{m.role}: {body}")
    return "\n".join(lines)


def _decide_retrieval(messages: list[ChatMessageIn]) -> tuple[bool, str]:
    """Return (needs_retrieval, router_tag).

    First user turn always retrieves. Otherwise ask the VLM; default to
    RETRIEVE if the classification fails / is ambiguous.
    """
    user_msgs = [m for m in messages if m.role == "user"]
    if len(user_msgs) <= 1:
        return True, "first-turn"

    # If the last assistant message had no page footer, we have nothing to
    # reuse — don't bother calling the router.
    last_asst = next((m for m in reversed(messages) if m.role == "assistant"), None)
    if last_asst is None or not _extract_prior_pages(last_asst.content):
        return True, "no-prior-pages"

    try:
        prompt = ROUTER_PROMPT.format(transcript=_transcript(messages))
        raw = generator.generate_text(prompt, max_new_tokens=8) or ""
    except Exception as e:
        print(f"router: falling back to RETRIEVE ({e})")
        return True, "router-error"

    tag = raw.strip().upper()
    if "FOLLOWUP" in tag:
        return False, f"router:{tag[:32]}"
    return True, f"router:{tag[:32]}"


# ── Page memory: encode & decode via the assistant markdown footer ──
# Match markdown links into the frontend's page namespace:
#   [p325](/aiwo-images/OM_...pdf_p0325.png)
PAGE_LINK_RE = re.compile(
    r"\]\(" + re.escape(PAGE_URL_PREFIX) + r"/([^)\s]+?\.png)\)"
)


def _extract_prior_pages(assistant_content: str) -> list[dict]:
    """Parse image filenames out of the previous assistant message's footer.

    The footer stores URL-encoded paths (required for markdown to render
    filenames with spaces), so we URL-decode each capture before handing
    it back to the generator as an on-disk filename.
    """
    pages: list[dict] = []
    seen: set[str] = set()
    for encoded in PAGE_LINK_RE.findall(assistant_content):
        img_filename = unquote(encoded)
        if img_filename in seen:
            continue
        seen.add(img_filename)
        page_id = img_filename.removesuffix(".png")
        m = re.search(r"_p(\d+)$", page_id)
        page_number = int(m.group(1)) if m else 0
        source = page_id.split(".pdf")[0] + ".pdf" if ".pdf" in page_id else ""
        pages.append({
            "page_id": page_id,
            "image_path": img_filename,
            "score": 0.0,
            "source": source,
            "page_number": page_number,
        })
    return pages


def _render_page_footer(pages: list[dict], max_shown: int) -> str:
    """Render a visible markdown footer — user-readable AND machine-readable
    (we parse the /aiwo-images/ links back on follow-up turns).

    Format: a horizontal strip of clickable thumbnails under a separator.
    Each thumbnail is a linked image `[![label](url)](url)` so CSS in the
    frontend can size them (see `.markdownanswer img`).
    """
    if not pages:
        return ""
    thumbs = []
    labels = []
    for p in pages[:max_shown]:
        img_filename = p.get("image_path", "")
        if not img_filename:
            continue
        page_id = p.get("page_id", img_filename.removesuffix(".png"))
        m = re.search(r"_p(\d+)$", page_id)
        label = f"p{int(m.group(1))}" if m else page_id
        score = float(p.get("score", 0.0))
        # URL-encode so filenames with spaces don't break markdown parsing
        # and our own regex on the follow-up turn.
        url = f"{PAGE_URL_PREFIX}/{quote(img_filename)}"
        thumbs.append(f"[![{label}]({url})]({url})")
        labels.append(
            f"[{label} ({score:.2f})]({url})" if score > 0 else f"[{label}]({url})"
        )
    if not thumbs:
        return ""
    return (
        "\n\n---\n📄 **Retrieved manual pages:** "
        + " · ".join(labels)
        + "\n\n"
        + " ".join(thumbs)
    )


# ── Endpoints ──

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "generator_backend": generator_backend,
    }


@app.get("/stats")
async def stats():
    if retriever is None:
        raise HTTPException(503, "Retriever not loaded")
    return {
        "index_size": len(retriever.page_metadata),
        "embedding_dim": retriever.doc_embeddings.shape[1],
        "model": MODEL_NAME,
        "generator_backend": generator_backend,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "models": [
            {
                "id": RAG_MODEL_ID,
                "name": "AiWo RAG (Ponsse manuals)",
                "provider": "aiwo-rag",
            }
        ]
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if retriever is None:
        raise HTTPException(503, "Retriever not loaded")

    t0 = time.time()
    pages, retrieval_latency = retriever.retrieve(req.query, top_k=req.top_k)

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
        retrieved_pages=_retrieved_to_pages(pages),
        retrieval_latency_ms=retrieval_latency * 1000,
        total_latency_ms=total_latency * 1000,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """TUNI-frontend-compatible multi-turn chat endpoint.

    Pipeline:
      1. Router chooses RETRIEVE or FOLLOWUP.
      2. RETRIEVE: fresh retrieval on the last user message.
         FOLLOWUP: reuse pages parsed from the previous assistant footer.
      3. VLM.generate_chat with the full history + chosen pages.
      4. Append a markdown page footer on RETRIEVE turns (also our
         page-memory for the next turn).
    """
    if retriever is None:
        raise HTTPException(503, "Retriever not loaded")
    if generator is None:
        raise HTTPException(503, "Generator not loaded")

    if not req.messages or req.messages[-1].role != "user" or not req.messages[-1].content.strip():
        raise HTTPException(400, "Last message must be a non-empty user turn")

    t0 = time.time()

    # ── Decide retrieval
    if req.force_retrieval is True:
        needs_retrieval, router_tag = True, "forced"
    elif req.force_retrieval is False:
        needs_retrieval, router_tag = False, "forced-skip"
    else:
        needs_retrieval, router_tag = _decide_retrieval(req.messages)

    # ── Retrieve or reuse
    retrieval_latency = 0.0
    if needs_retrieval:
        query_text = req.messages[-1].content
        pages, retrieval_latency = retriever.retrieve(query_text, top_k=req.top_k)
    else:
        last_asst = next((m for m in reversed(req.messages) if m.role == "assistant"), None)
        pages = _extract_prior_pages(last_asst.content) if last_asst else []
        if not pages:
            # Safety net — router said FOLLOWUP but we have nothing to reuse.
            needs_retrieval = True
            router_tag = "fallback-retrieve"
            pages, retrieval_latency = retriever.retrieve(
                req.messages[-1].content, top_k=req.top_k
            )

    # ── Generate with full history
    history = [{"role": m.role, "content": m.content} for m in req.messages]
    try:
        answer = generator.generate_chat(
            messages=history,
            retrieved_pages=pages,
            image_dir=IMAGE_DIR,
            max_pages=req.max_pages_for_vlm,
        )
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    # ── Footer only when we actually retrieved (keeps history clean on FOLLOWUP)
    if needs_retrieval:
        answer = answer + _render_page_footer(pages, max_shown=req.max_pages_for_vlm)

    total_latency = time.time() - t0

    return ChatResponse(
        result=ChatResultMessage(role="assistant", content=answer),
        retrieved_pages=_retrieved_to_pages(pages[: req.max_pages_for_vlm]),
        retrieval_latency_ms=retrieval_latency * 1000,
        total_latency_ms=total_latency * 1000,
        retrieved=needs_retrieval,
        router_decision=router_tag,
    )
