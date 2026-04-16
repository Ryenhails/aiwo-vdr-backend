# AiWo Visual Document QA

A visual-retrieval-augmented-generation (V-RAG) assistant for industrial
forestry-machine operators. Shows the operator's question, pulls the most
relevant pages from the equipment's PDF manual, and lets a vision-capable
LLM answer grounded in those actual pages.

This repository is a **single integrated deployment** containing both:

- `./` — FastAPI backend (retrieval + VLM generation).
- `frontend/` — Next.js chat UI (fork of TUNI's multi-provider chat UI,
  extended with an `aiwo-rag` provider).

A developer who just wants to use the backend from a different frontend
should read [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md) — it is the
single-document reference for the HTTP API, the multi-turn protocol, data
prep, env vars, and troubleshooting.

---

## Architecture

```
┌──────────────────────┐   POST /api/chat    ┌─────────────────────────────┐
│  frontend/  (Next 14)│ ──────────────────▶ │  Next.js /api/chat          │
│  chat UI + mic TTS   │                     │  (validates, rate-limits,   │
│  "AiWo RAG" option   │                     │   dispatches to provider)   │
│  in the dropdown     │ ◀────────────────── │                             │
└──────────────────────┘   { result }        └────────────┬────────────────┘
     ▲    │                                               │ createAiwoRagProvider
     │    │ /aiwo-images/*.png  (Next.js rewrite)         ▼
     │    └─────────────────────────────────┐   POST http://backend/chat
     │                                      │
     │                                      ▼
┌────┴─────────────────────────────────────────────────────────────────────┐
│  Backend (FastAPI, port 8765)                                            │
│                                                                          │
│   POST /chat ───▶  LLM router (RETRIEVE / FOLLOWUP)                      │
│                ├─▶ retrieval (NanoVDR-S-Multi + cosine top-k)            │
│                └─▶ generator  (VLM_BACKEND=local | claude | azure)       │
│                                                                          │
│   GET /images/*.png  — raw page PNGs from data/images/                   │
└──────────────────────────────────────────────────────────────────────────┘
```

Highlights:

- **Stateless multi-turn** — the full conversation travels in every
  request; page-memory is embedded in the assistant's markdown footer.
- **Self-routing** — the VLM decides whether a given turn needs a fresh
  retrieval or can answer from prior context. Follow-ups skip the retriever.
- **Three interchangeable VLMs** — local Qwen2.5-VL-7B-Instruct (on-prem
  GPU), Anthropic Claude, or Azure OpenAI GPT-4o. Same API from the
  frontend's side.
- **Thumbnails** — retrieved pages render as clickable thumbnails under the
  assistant's answer; click to open full-size.

---

## Quick start

### 0. Prereqs

- Python 3.11 + a CUDA GPU (≥20 GB VRAM) if you want `VLM_BACKEND=local`.
  Otherwise any machine that can run `pip install` will do — set
  `VLM_BACKEND=azure` or `VLM_BACKEND=claude` and supply creds.
- Node.js ≥ 18 for the frontend.

### 1. Backend

```bash
# Create an env
conda create -p ./.conda-aiwo-rag python=3.11 -y
conda activate ./.conda-aiwo-rag
pip install -r requirements.txt

# Pick a generation backend
export VLM_BACKEND=local                                              # on-prem GPU
# export VLM_BACKEND=claude   ANTHROPIC_API_KEY=sk-ant-...             # Anthropic
# export VLM_BACKEND=azure    AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_API_KEY=...

# (first-time) regenerate page PNGs — the index itself is committed (§ Data)
python - <<'PY'
import fitz, os, json
meta = json.load(open("data/index/page_metadata.json"))
pages_by_pdf = {}
for m in meta:
    pages_by_pdf.setdefault(m["source"], []).append(m)
os.makedirs("data/images", exist_ok=True)
for pdf_name, pages in pages_by_pdf.items():
    doc = fitz.open(f"data/manuals/{pdf_name}")
    for m in pages:
        pix = doc[m["page_number"]].get_pixmap(matrix=fitz.Matrix(200/72, 200/72))
        pix.save(f"data/images/{m['image_path']}")
print(f"rasterised {len(meta)} pages")
PY

# Serve
uvicorn api.server:app --host 0.0.0.0 --port 8765
```

Health check:
```bash
curl http://127.0.0.1:8765/health
```

### 2. Frontend

```bash
cd frontend
cp .env.local.example .env.local        # edit AIWO_RAG_BASE_URL if backend isn't local
npm install
npm run dev                              # → http://localhost:3000
```

Open `http://localhost:3000`, pick **AiWo RAG (Ponsse, local Qwen2.5-VL)**
from the dropdown, ask a question. Retrieved pages appear as thumbnails
below the answer; click to open full-size.

### 3. Indexing a new manual (requires GPU)

If you add a different PDF under `data/manuals/`, regenerate the index:

```bash
python scripts/index_documents.py \
    --pdf_dir data/manuals/ \
    --output_dir data/index/ \
    --image_dir data/images/ \
    --dpi 200 --batch_size 4
```

Details and what-to-commit guidance: see
[`INTEGRATION_GUIDE.md §7`](INTEGRATION_GUIDE.md#7-data-preparation).

---

## Data

What is committed:

| Path | Size | Why |
|---|---|---|
| `data/manuals/*.pdf` | ~36 MB (one Ponsse manual) | source-of-truth; everything else regenerates from it |
| `data/index/doc_embeddings.npy` + `page_metadata.json` | ~2 MB | rebuilding needs GPU + Qwen3-VL-Embedding-2B (teacher) |

What is **not** committed:

| Path | Size | Why skipped |
|---|---|---|
| `data/images/*.png` | ~188 MB (491 pages @ 200 DPI) | regenerates from PDF in ~20 s on CPU via PyMuPDF (script in [Quick start §1](#1-backend)) |

---

## Env vars (full table in `INTEGRATION_GUIDE.md §8`)

Minimum to run:

| Var | Required when | Example |
|---|---|---|
| `VLM_BACKEND`            | always           | `local` / `claude` / `azure` |
| `ANTHROPIC_API_KEY`      | `VLM_BACKEND=claude` | `sk-ant-...` |
| `AZURE_OPENAI_ENDPOINT`  | `VLM_BACKEND=azure`  | `https://<resource>.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY`   | `VLM_BACKEND=azure`  |  |

Frontend:

| Var | Required when | Example |
|---|---|---|
| `AIWO_RAG_BASE_URL` | always (enables the `aiwo-rag` provider) | `http://127.0.0.1:8765` |

---

## Project layout

```
aiwo-vdr-backend/
├── api/server.py                 FastAPI entry — /chat, /query, router, page-memory
├── retrieval/retriever.py        NanoVDR-S-Multi SBERT retriever
├── generation/
│   ├── factory.py                VLM_BACKEND dispatcher
│   ├── vlm_generator.py          Azure OpenAI GPT-4o (vision)
│   ├── claude_generator.py       Anthropic Claude (vision)
│   └── local_vlm_generator.py    Qwen2.5-VL-7B-Instruct (local GPU)
├── scripts/
│   ├── index_documents.py        PDF → PNG → embeddings index (GPU)
│   └── smoke_test.py             stdlib-only client for /health and /chat
├── data/
│   ├── manuals/                  (committed)  source PDFs
│   └── index/                    (committed)  doc_embeddings.npy + page_metadata.json
├── frontend/                     Next.js 14 chat UI (see §Frontend)
│   ├── pages/                    index.tsx, api/chat.ts, api/providers.ts
│   ├── lib/providers.ts          includes the aiwo-rag provider
│   ├── next.config.js            rewrites /aiwo-images/* to the backend
│   └── styles/Home.module.css    thumbnail sizing
├── requirements.txt
├── .env.example
├── README.md                     (this file — overview + quick start)
├── INTEGRATION_GUIDE.md          (deep API reference for frontend devs)
└── CHANGELOG.md
```

---

## Attribution

The `frontend/` directory is derived from TUNI's
[`aiwo-visual-document-qa`](https://gitlab.tuni.fi/pirg/aiwo-visual-document-qa)
chat UI, licensed MIT, extended with an `aiwo-rag` provider that calls
this backend. The retrieval stack is based on NanoVDR; the teacher encoder
is Qwen3-VL-Embedding-2B.

This project has received funding from the **Business Finland**
co-innovation programme under grant agreement No. 69/31/2025, supported by
the [AiWo](https://aifieldwork.aalto.fi/events/) project (2025–2027).
