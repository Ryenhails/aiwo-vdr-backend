# AiWo Visual Document QA — Frontend Integration Guide

This backend turns operator questions into grounded answers by (a) retrieving
the most relevant page images from a pre-indexed equipment manual and (b)
running a vision-capable LLM over those pages plus the prior conversation.
The entire contract with a frontend is three HTTP endpoints.

This guide is written for a frontend developer who wants to call the backend
from their own web app. It covers the API surface, the multi-turn protocol,
data preparation, and everything you need to stand the system up on a fresh
host.

---

## 1. System overview

```
┌─────────────────┐       POST /chat        ┌────────────────────────────┐
│   Frontend      │ ─────────────────────▶  │      AiWo RAG backend       │
│   (any Web app) │                         │      (FastAPI, :8765)       │
│                 │  { messages, model }    │                             │
│                 │ ◀─────────────────────  │  ┌───────────────────────┐ │
│                 │  { result,              │  │ Retriever             │ │
│                 │    retrieved_pages,     │  │  NanoVDR-S-Multi /    │ │
│                 │    retrieved,           │  │  (query text → top-k) │ │
│                 │    router_decision }    │  └───────────────────────┘ │
│                 │                         │  ┌───────────────────────┐ │
│  Thumbnails via │   GET  /images/*.png    │  │ LLM router            │ │
│  GET /aiwo-images/*  ────────────────▶   │  │  RETRIEVE / FOLLOWUP  │ │
│  (Next.js       │                         │  └───────────────────────┘ │
│   rewrite)      │                         │  ┌───────────────────────┐ │
└─────────────────┘                         │  │ VLM generator         │ │
                                            │  │  local Qwen2.5-VL /   │ │
                                            │  │  Claude / Azure GPT-4o│ │
                                            │  └───────────────────────┘ │
                                            └────────────────────────────┘
```

Key properties:

- **Stateless.** Every `/chat` request carries the full conversation. The
  backend never stores sessions — if you want a new conversation, just send
  an empty history.
- **Grounded.** Answers are produced from images of actual manual pages, not
  from the LLM's parametric memory.
- **Self-routing.** On the 2nd+ turn the backend asks the LLM itself whether
  the new question needs a fresh retrieval (`RETRIEVE`) or can be answered
  from existing context (`FOLLOWUP`). Follow-ups take ~0 ms of retrieval.
- **Interchangeable generator.** The VLM backend is one of `local` (on-prem
  GPU), `claude` (Anthropic API), or `azure` (Azure OpenAI GPT-4o). Picked at
  server startup via the `VLM_BACKEND` env var; same API for the frontend.

---

## 2. Quick start (5 min)

Assumes the backend host already has the data prepared (see §7) and CUDA if
you want `VLM_BACKEND=local`.

```bash
# 1. Clone + env
git clone <backend repo URL>
cd aiwo-vdr-backend

conda create -p ./.conda python=3.11 -y
conda activate ./.conda
pip install -r requirements.txt

# 2. Pick a generation backend — see §9 for the full menu
export VLM_BACKEND=local                # uses GPU
# or: export VLM_BACKEND=claude ANTHROPIC_API_KEY=sk-ant-...
# or: export VLM_BACKEND=azure AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_API_KEY=... VLM_DEPLOYMENT=gpt-4o

# 3. (First time only) build the index — see §7
python scripts/index_documents.py --pdf_dir data/manuals/ \
    --output_dir data/index/ --image_dir data/images/

# 4. Serve
uvicorn api.server:app --host 0.0.0.0 --port 8765
```

Smoke test:
```bash
curl http://127.0.0.1:8765/health
curl -X POST http://127.0.0.1:8765/chat -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"How do I replace the hydraulic filter?"}]}'
```

---

## 3. API reference

### `POST /chat` — main entry point

Request:
```jsonc
{
  "messages": [
    { "role": "user",      "content": "How do I replace the hydraulic filter?" },
    { "role": "assistant", "content": "<previous answer incl. page footer>" },
    { "role": "user",      "content": "What's the torque on step 4?" }
  ],

  // Optional — ignored by the backend, kept for TUNI frontend compatibility
  "provider":  "aiwo-rag",
  "model":     "aiwo-rag",

  // Optional — override retrieval size
  "top_k":             5,      // how many pages to retrieve (default 5)
  "max_pages_for_vlm": 3,      // how many to show the VLM + render as thumbnails (default 3)

  // Optional debug override — skip the router
  "force_retrieval": true      // always retrieve
  // "force_retrieval": false  // never retrieve, must have pages in history
}
```

Response:
```jsonc
{
  "result": {
    "role": "assistant",
    "content": "1. Ensure pressure is 0 bar.\n2. ...\n\n---\n📄 **Retrieved manual pages:** [p332 (0.53)](/aiwo-images/OM_...p0332.png) · ...\n\n[![p332](/aiwo-images/OM_...p0332.png)](/aiwo-images/OM_...p0332.png) ..."
  },
  "retrieved_pages": [
    { "page_id": "OM_PONSSE..._p0332", "score": 0.53,
      "image_url": "/aiwo-images/OM_PONSSE%20..._p0332.png",
      "source": "OM_PONSSE SCORPION KING 8W_A011333_ENG.pdf",
      "page_number": 332 }
  ],
  "retrieval_latency_ms": 48.1,
  "total_latency_ms":    3900.7,
  "retrieved":       true,                  // whether a retrieval happened this turn
  "router_decision": "router:RETRIEVE"      // first-turn | no-prior-pages | router:RETRIEVE | router:FOLLOWUP | forced | forced-skip | router-error | fallback-retrieve
}
```

Notes:
- `result.role` is always `"assistant"`. `result.content` is markdown — render
  with `react-markdown` + `remark-gfm` (or equivalent).
- On `retrieved: true`, the backend appends a markdown footer to
  `result.content` containing the page thumbnails / links (§5). On
  `retrieved: false` it doesn't — the history stays clean.
- `retrieved_pages` is a parallel structured list of the same pages the VLM
  saw this turn (top-`max_pages_for_vlm`). Use this if you prefer to render
  pages outside the message bubble.

### `GET /health`
```json
{ "status": "ok", "model": "nanovdr/NanoVDR-S-Multi", "generator_backend": "local" }
```

### `GET /stats`
```json
{ "index_size": 491, "embedding_dim": 2048,
  "model": "nanovdr/NanoVDR-S-Multi", "generator_backend": "local" }
```

### `GET /v1/models` — provider-discovery (optional)
```json
{ "models": [ { "id": "aiwo-rag", "name": "AiWo RAG (Ponsse manuals)", "provider": "aiwo-rag" } ] }
```
Only useful if your frontend needs to auto-populate a model dropdown.

### `GET /images/<filename>` — raw page PNG
Direct access to any page image. Filenames come from
`page_metadata.json` (see §7). The TUNI frontend uses `/aiwo-images/*` (§6)
instead of hitting this directly.

### `POST /query` — legacy retrieval+generate (single-shot)
Takes `{ query, top_k, generate_answer, max_pages_for_vlm }` and returns
`{ query, answer, retrieved_pages, retrieval_latency_ms, total_latency_ms }`.
Kept for backward compatibility; new frontends should use `/chat`.

---

## 4. Minimal integration example

A bare-bones fetch from any frontend:

```typescript
async function ask(history: {role: "user" | "assistant", content: string}[]) {
  const res = await fetch("http://your-backend:8765/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: history }),
  });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.result.content;                 // markdown string
}

// Usage — build history as you chat:
const history = [];
history.push({ role: "user", content: "How do I replace the hydraulic filter?" });
const a1 = await ask(history);
history.push({ role: "assistant", content: a1 });  // include the footer verbatim — the backend parses it on the next turn

history.push({ role: "user", content: "thanks!" });
const a2 = await ask(history);                    // router decides FOLLOWUP, no retrieval
```

The full assistant content — footer and all — **must go back to the backend
unchanged** in the next turn's history. That's how the backend remembers what
pages were retrieved (§5).

---

## 5. Multi-turn protocol (page-memory)

The backend is stateless, so the "memory of which pages we're discussing"
lives inside the assistant messages as a markdown footer. Flow:

```
Turn 1 user     →  backend retrieves pages p332/p325/p329
                   backend generates an answer
                   backend appends footer linking those three pages
Turn 1 assistant returned to client:
   "1. Ensure pressure is 0 bar.
    2. ...
    ---
    📄 Retrieved manual pages: [p332 (0.53)](/aiwo-images/..._p0332.png) · ...
    [![p332](/aiwo-images/..._p0332.png)](/aiwo-images/..._p0332.png) ..."

Turn 2 user: "thanks!"
   (client sends the full history, including the full assistant content above)
   → backend's router says FOLLOWUP (no new retrieval)
   → backend parses the /aiwo-images/ links out of the previous assistant
     message and reuses the same pages
   → the VLM sees the whole conversation + those pages and answers
   → no footer is appended (history stays clean)
```

The router's behaviour on turn 2+:

- First user message in history        →  always RETRIEVE
- Last assistant has no page footer    →  RETRIEVE (nothing to reuse)
- Otherwise Qwen is asked to classify  →  RETRIEVE or FOLLOWUP

You can override this for debugging with `"force_retrieval": true|false`.

The router's latest decision and whether retrieval fired are echoed back in
`retrieved` and `router_decision`.

---

## 6. Page images and URL conventions

Manual page PNGs are served from the backend at `GET /images/<filename>`.
The backend emits URLs with the prefix `/aiwo-images/` (configurable via the
`PAGE_URL_PREFIX` env var — default `/aiwo-images`). The prefix exists so
that the frontend can proxy those requests through its own origin without
exposing the backend URL to the browser. E.g. in the TUNI Next.js frontend:

```js
// next.config.js
async rewrites() {
  return [
    { source: "/aiwo-images/:path*",
      destination: `${process.env.AIWO_RAG_BASE_URL}/images/:path*` },
  ];
}
```

If your frontend is plain React/Vue served from a different origin, either
(a) set up an equivalent reverse proxy, or (b) set `PAGE_URL_PREFIX` on the
backend to an absolute URL and let the browser fetch the PNGs directly.
The backend already sends `Access-Control-Allow-Origin: *`.

**Filenames can contain spaces** (`OM_PONSSE SCORPION KING 8W_...`). The
backend URL-encodes paths when rendering the footer (`%20` for spaces), so
follow-up turns round-trip correctly. Don't strip the encoding in your
frontend or the backend's regex will fail to match on the next turn.

---

## 7. Data preparation

Everything except the raw PDF lives under `data/`:

```
data/
  manuals/    Your input PDFs (source-of-truth, committed).
  images/     One PNG per page (auto-generated; 200 DPI by default).
  index/      doc_embeddings.npy (float16, shape=[pages, 2048])
              page_metadata.json (per-page: page_id, image_path, source, page_number)
```

### 7.1  Indexing a new manual

```bash
# Requires the full env (GPU for Qwen3-VL-Embedding-2B).
# Runs offline — no external services needed.
python scripts/index_documents.py \
    --pdf_dir    data/manuals/ \
    --output_dir data/index/ \
    --image_dir  data/images/ \
    --dpi 200 --batch_size 4
```

What it does:

1. For each PDF in `--pdf_dir`, rasterizes every page at `--dpi` with
   PyMuPDF → PNG → `data/images/<pdf>_<pNNNN>.png`.
2. Loads `Qwen/Qwen3-VL-Embedding-2B` (teacher) via `sentence-transformers`
   and encodes every page image → 2048-D vector.
3. Saves `data/index/doc_embeddings.npy` and `page_metadata.json`.

At runtime the backend loads `nanovdr/NanoVDR-S-Multi` (student, same
embedding dim) for query-side encoding — their embedding spaces are aligned
by distillation.

### 7.2  What to put in Git

| Path | Size (Ponsse example) | Recommendation |
|---|---|---|
| `data/manuals/*.pdf`          | 36 MB for one book | **Commit.** Source-of-truth, regenerates everything else. |
| `data/index/doc_embeddings.npy` + `page_metadata.json` | ~2 MB | **Commit.** Tiny, and building it needs a GPU — worth not making every collaborator re-run the teacher. |
| `data/images/*.png`           | 188 MB (491 pages @ 200 DPI) | **Skip.** Re-rasterizing from the PDF is CPU-only and takes ~20 s; the repo stays lean. See §7.3. |

The default `.gitignore` ignores all three subdirectories. To commit the
manual + index but not the images, add to `.gitignore`:

```
# override: commit manual + index, skip large images
!data/manuals/
!data/index/
```

### 7.3  Regenerating images only (no GPU)

A collaborator who cloned with the committed PDF + index but without images
can rasterize with PyMuPDF alone:

```python
# scripts/rasterize_only.py (≈10 LOC)
import fitz, os, json
meta = json.load(open("data/index/page_metadata.json"))
by_pdf = {}
for m in meta: by_pdf.setdefault(m["source"], []).append(m)
os.makedirs("data/images", exist_ok=True)
for pdf_name, pages in by_pdf.items():
    doc = fitz.open(f"data/manuals/{pdf_name}")
    for m in pages:
        pix = doc[m["page_number"]].get_pixmap(matrix=fitz.Matrix(200/72, 200/72))
        pix.save(f"data/images/{m['image_path']}")
```

---

## 8. Environment variables

| Var | Required | Default | Meaning |
|---|---|---|---|
| `VLM_BACKEND`            | yes | `azure`   | `azure` / `claude` / `local` |
| `MODEL_NAME`             | no  | `nanovdr/NanoVDR-S-Multi` | Retriever SBERT model |
| `INDEX_DIR`              | no  | `data/index`              | Pre-built index path |
| `IMAGE_DIR`              | no  | `data/images`             | Page PNG directory |
| `PAGE_URL_PREFIX`        | no  | `/aiwo-images`            | URL prefix in footer/image_url |
| **`azure` backend**      |     |                           |                                |
| `AZURE_OPENAI_ENDPOINT`  | yes | —                         | `https://<resource>.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY`   | yes | —                         |                                |
| `VLM_DEPLOYMENT`         | no  | `gpt-4o`                  | Azure deployment name          |
| **`claude` backend**     |     |                           |                                |
| `ANTHROPIC_API_KEY`      | yes | —                         |                                |
| `ANTHROPIC_MODEL`        | no  | `claude-opus-4-6`         |                                |
| **`local` backend**      |     |                           |                                |
| `LOCAL_VLM_MODEL`        | no  | `Qwen/Qwen2.5-VL-7B-Instruct` | HF model name               |
| `LOCAL_VLM_DEVICE`       | no  | `cuda`                    |                                |

---

## 9. Generation backends

All three implement the same contract (`generate_chat`, `generate_text`) so
the frontend sees the same API regardless. Differences:

| Backend | Startup cost | Per-call latency (3 pages, 1024 tok) | VRAM | Requires |
|---|---|---|---|---|
| `local` (Qwen2.5-VL-7B-Instruct, fp16) | ~30 s warm-up                | 2–5 s | ~17 GB | a GPU with ≥20 GB (H100/H200 slice, A100) |
| `claude` (Anthropic API)               | 0                            | 2–6 s | 0      | network + API key                         |
| `azure`  (GPT-4o vision)               | 0                            | 2–5 s | 0      | Azure OpenAI resource                     |

The system prompt and multi-turn shape are identical; output style can still
differ (e.g. Claude tends to be a bit more verbose).

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'sentence_transformers.base'` at startup | ST < 5.2 | `pip install 'sentence-transformers>=5.4'` |
| `qwen_vl_utils` import error `No module named 'torchvision'` | torchvision missing on a CUDA-only env | `pip install torchvision` |
| Teacher load prints "Some weights of Qwen3VLModel were not initialized…" | You used `transformers.AutoModel` instead of `SentenceTransformer` | Use the shipped `scripts/index_documents.py`, not the raw Transformers API |
| Retrieval returns random-looking pages | The embedding index was built with random weights (above bug) | Rebuild: `rm -rf data/index data/images && python scripts/index_documents.py ...` |
| Thumbnails render as literal markdown text | Filename has spaces + footer path not URL-encoded | Upgrade to latest backend (`PAGE_LINK_RE` + `quote()` in `_render_page_footer`) |
| `"thanks!"` always triggers retrieval | `_extract_prior_pages` returning empty — either the footer is being stripped by the frontend before re-send, or URL-encoding is wrong | The frontend must send the assistant's content back **verbatim** in the next turn |
| VLM answers ignore the retrieved pages | Either `IMAGE_DIR` doesn't exist at backend startup, or `page.image_path` points at a file that isn't there (e.g. images not regenerated after re-clone) | Check `ls "$IMAGE_DIR" \| head`; re-rasterize (§7.3) |
| 429 rate-limit from Claude/Azure under load | Paid plan quota | Fall back to `VLM_BACKEND=local` during demos |

---

## 11. File layout for reference

```
aiwo-vdr-backend/
├── api/
│   └── server.py                     FastAPI entry, /chat + /query + router + page-memory
├── retrieval/
│   └── retriever.py                  NanoVDR SBERT + cosine top-k
├── generation/
│   ├── factory.py                    picks a generator from VLM_BACKEND
│   ├── vlm_generator.py              Azure OpenAI GPT-4o (vision)
│   ├── claude_generator.py           Anthropic Claude (vision)
│   └── local_vlm_generator.py        Qwen2.5-VL-7B-Instruct (local GPU)
├── scripts/
│   ├── index_documents.py            PDF → PNG → embeddings index
│   └── smoke_test.py                 stdlib-only client for /health/chat
├── data/
│   ├── manuals/   ◀ PDF sources
│   ├── images/    ◀ page PNGs (auto)
│   └── index/     ◀ doc_embeddings.npy + page_metadata.json (auto)
├── requirements.txt
├── .env.example
├── CHANGELOG.md
└── INTEGRATION_GUIDE.md              (this file)
```
