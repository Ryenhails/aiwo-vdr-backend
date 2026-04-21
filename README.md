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

- **Python 3.11**. Any virtual-environment tool works — the examples below
  use the stdlib `venv`; see [conda alternative](#conda-alternative) if you
  prefer conda.
- **Node.js ≥ 18** for the frontend. Check with `node -v`.
- **A CUDA GPU with ≥20 GB VRAM** only if you want `VLM_BACKEND=local`.
  For `claude` / `azure`, any machine that can reach the API will do.

### 1. Backend

```bash
# Create a virtual env and install deps
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure — the backend reads .env automatically, no `export` needed.
cp .env.example .env
$EDITOR .env          # set VLM_BACKEND and the relevant credentials

# First time only — rebuild page PNGs from the committed index (~20 s, CPU)
python scripts/rasterize_from_index.py

# Serve
uvicorn api.server:app --host 0.0.0.0 --port 8765
```

Health check:
```bash
curl http://127.0.0.1:8765/health
```

#### conda alternative

```bash
conda create -p ./.conda-aiwo-rag python=3.11 -y
conda activate ./.conda-aiwo-rag
pip install -r requirements.txt
# …then proceed with `cp .env.example .env`, the rasterize script, and uvicorn.
```

If `conda` is not on your PATH, install miniconda first
([docs.conda.io/miniconda](https://docs.conda.io/en/latest/miniconda.html))
and add it to your shell's PATH per the installer's instructions.

### 2. Frontend

```bash
cd frontend
cp .env.local.example .env.local        # edit AIWO_RAG_BASE_URL if backend isn't local
npm install
npm run dev                              # → http://localhost:3000
```

Next.js reads `.env.local` automatically — do **not** `export` secrets in
your shell.

Open `http://localhost:3000`, pick **AiWo RAG (Ponsse, local Qwen2.5-VL)**
from the dropdown, ask a question. Retrieved pages appear as thumbnails
below the answer; click to open full-size.

### 3. Indexing a new manual (requires GPU)

The bundled index covers exactly one manual (Ponsse Scorpion King 8W, 491
pages). To point the system at different PDFs, drop them into
`data/manuals/` and rebuild:

```bash
python scripts/index_documents.py \
    --pdf_dir data/manuals/ \
    --output_dir data/index/ \
    --image_dir data/images/ \
    --dpi 200 --batch_size 4
```

This rasterises every page AND encodes it with Qwen3-VL-Embedding-2B, so
after it finishes you don't need the separate `rasterize_from_index.py`.
Details and what-to-commit guidance:
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
