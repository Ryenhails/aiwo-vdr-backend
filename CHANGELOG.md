# Changelog

Changes to the AiWo Visual Document QA backend, tracked for auditability.
Dates are YYYY-MM-DD (Europe/Helsinki).

## 2026-04-16 — TUNI frontend integration (minimal-diff)

Goal: keep the current `/query` contract for backward compatibility while adding a
TUNI-frontend-compatible `/chat` endpoint, and let the answer-generation backend
be swapped between Azure OpenAI GPT-4o (original), Anthropic Claude API, and a
local Qwen2.5-VL model via a single env var.

Scope: code only. No indexing or deployment runs yet — H200 node not provisioned.

### Added
- `data/manuals/OM_PONSSE SCORPION KING 8W_A011333_ENG.pdf` — copied from
  `/scratch/elec/t412-aiwo/chenyang/aiwo_demo/en_manuals/`. Covered by existing
  `.gitignore` (`data/manuals/`), so the 36 MB blob stays out of git.
- `generation/claude_generator.py` — Anthropic Claude Messages API with vision.
  Default model `claude-opus-4-6`. Requires `ANTHROPIC_API_KEY`.
- `generation/local_vlm_generator.py` — local Qwen2.5-VL-Instruct generator
  (default `Qwen/Qwen2.5-VL-7B-Instruct`, fp16, CUDA). Loads once at startup.
- `generation/factory.py` — picks one of `{azure, claude, local}` at startup
  based on `VLM_BACKEND` env var (default `azure` to preserve old behaviour).
- `POST /chat` endpoint in `api/server.py` — accepts TUNI's
  `{messages, provider, model}` shape and returns `{result: {role, content}}`.
  Uses the last user message as the RAG query, ignores `model` for the retriever,
  and passes generation through the factory-selected generator.
- `GET /v1/models` — lists the single virtual model `aiwo-rag` so the TUNI
  provider factory has something to register.

### Changed
- `api/server.py`: generator selection moved to `generation.factory`. The
  startup hook now builds the retriever + selected generator. The existing
  `/query`, `/health`, `/stats` endpoints keep the same response shape.
- `.env.example`: documents the new `VLM_BACKEND`, `ANTHROPIC_API_KEY`,
  `ANTHROPIC_MODEL`, `LOCAL_VLM_MODEL`, `LOCAL_VLM_DEVICE` variables.
- `requirements.txt`: adds `anthropic`, `PyMuPDF`, `tqdm`, `torch`,
  `transformers`, `qwen-vl-utils`, `accelerate` (the indexing script and
  local VLM generator all need these; they were already implicit imports).

### Not changed
- `retrieval/retriever.py` — the NanoVDR path is untouched.
- `scripts/index_documents.py` — untouched; still expects GPU.
- `/query`, `/health`, `/stats` response shapes — untouched.

### Open items (need H200 to verify)
- Run `scripts/index_documents.py` on the Ponsse PDF to produce
  `data/index/doc_embeddings.npy` + `data/images/`.
- Smoke test `/chat` with each `VLM_BACKEND` value.
- Confirm `nanovdr/NanoVDR-S-Multi` exists on HuggingFace; the README points
  to an arXiv paper with a future date, so the model name may need updating.

## 2026-04-16 (later) — H200 debug session

Ran against H200 MIG 2g.35gb on gpu49. Fresh `aiwo-rag` conda env at
`/scratch/work/liuz16/.conda_envs/aiwo-rag` (Python 3.11) built from
`requirements.txt`; both ST models, Qwen2.5-VL-7B generation, and the
full `/chat` round-trip verified end to end on the Ponsse manual.

### Fixes applied
- `scripts/index_documents.py`: rewrote `encode_with_teacher`. The original
  loaded `Qwen/Qwen3-VL-Embedding-2B` via `transformers.AutoModel`, which
  silently initialises weights randomly ("Some weights of Qwen3VLModel were
  not initialized from the model checkpoint …") and would have produced
  meaningless embeddings. The HF model card's recommended path is
  `SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B").encode([{"image": img}])`,
  which wraps the custom embedder module correctly — that is what we now
  use. Same output dim (2048), verified teacher↔student alignment:
  teacher-image(red)×student-text("red square") = 0.41 vs
  teacher-image(red)×student-text("blue circle") = 0.17.
- `requirements.txt`:
  - Bumped `sentence-transformers` to `>=5.4` (was `>=3.0`). The
    Qwen3-VL-Embedding-2B `modules.json` references
    `sentence_transformers.base.modules.transformer.Transformer`, a path
    that only exists from ST 5.2 onward. Loading on 5.1 raised
    `ModuleNotFoundError: No module named 'sentence_transformers.base'`.
  - Added `torchvision>=0.26`. It's an implicit import inside
    `qwen_vl_utils.vision_process` and was missing on the fresh env,
    which crashed the local-VLM path at `from qwen_vl_utils import …`.

### Verified on gpu49 (H200 MIG 2g.35gb, 35 GB)
- Indexing: 491 pages from the Ponsse PDF → `doc_embeddings.npy (491, 2048)`
  in `data/index/`, PNGs in `data/images/` (188 MB). Teacher encoding
  completes in the GPU's sub-10 GB envelope.
- Retrieval: `GET /health`, `/stats`, `/v1/models` all respond. Query
  "How to replace the hydraulic filter?" retrieves pages p0320–p0333
  (hydraulic maintenance section) with cosine scores ~0.50–0.54.
- Generation with `VLM_BACKEND=local`: loads Qwen2.5-VL-7B-Instruct
  (8.29 B params, ~16.6 GB fp16 VRAM). End-to-end `/chat` returns a
  structured six-step hydraulic-filter replacement procedure, details
  ("place new element with green silicone end first", "refit O-ring")
  matching real manual content. First-call total latency ≈ 11.9 s
  (includes first-image preprocessing); retrieval alone ≈ 0.2 s after warmup.

### Added
- `scripts/smoke_test.py` — stdlib-only client hitting `/health`,
  `/stats`, `/v1/models`, and `/chat` (or `/query` with `--skip-generation`).

### Not yet verified
- `VLM_BACKEND=claude` — needs a real `ANTHROPIC_API_KEY`.
- `VLM_BACKEND=azure` — needs real Azure credentials.
- TUNI frontend → our backend over the `aiwo-rag` provider (requires
  running the TUNI Next.js app with `AIWO_RAG_BASE_URL` set).

## 2026-04-16 (even later) — multi-turn, router, page thumbnails

End-to-end verified with local Qwen2.5-VL on gpu49. TUNI frontend at
`http://gpu49:3000`, backend at `127.0.0.1:8765`.

### Added
- **Full multi-turn history** (backend + every generator).
  - New `BaseGenerator.generate_chat(messages, retrieved_pages, image_dir,
    max_pages)` on Azure / Claude / local generators. History turns are
    replayed as text-only; retrieved images are attached to the LAST user
    turn. Legacy `generate(query, …)` delegates to `generate_chat(...)`.
- **LLM router** (`_decide_retrieval` in `api/server.py`).
  - First user turn → always RETRIEVE.
  - No page-footer in prior assistant message → RETRIEVE (nothing to reuse).
  - Otherwise a short pure-text VLM classification (`generate_text`)
    returns `RETRIEVE` / `FOLLOWUP`. Few-shot examples in `ROUTER_PROMPT`.
  - Failure → safe default RETRIEVE.
  - `ChatRequest.force_retrieval` bypasses the router for debugging.
  - `ChatResponse.retrieved` + `.router_decision` expose what the router did.
- **Page memory via the assistant footer.**
  - On RETRIEVE turns, the backend appends a markdown footer to the
    assistant's content: a label line (`p332 (0.53)`, …) plus a row of
    linked thumbnails (`[![pNNN](/aiwo-images/…)](/aiwo-images/…)`).
  - On FOLLOWUP turns, the backend parses the `/aiwo-images/…png` paths
    out of the most recent assistant message (`PAGE_LINK_RE`) and reuses
    them — no retriever call.
  - This keeps the backend stateless: all context lives in the messages
    the frontend sends back. A fresh chat clears the memory naturally.
- **Frontend thumbnail rendering.**
  - `next.config.js`: added a `/aiwo-images/:path*` rewrite that forwards
    to `${AIWO_RAG_BASE_URL}/images/:path*`. Keeps image requests on the
    frontend origin so the user only needs one SSH tunnel (port 3000)
    and the backend's CORS isn't relevant.
  - `styles/Home.module.css`: constrained `.markdownanswer img` to
    max-width 160 × max-height 220 so inline thumbnails don't blow up the
    message bubble.

### Changed
- `api/server.py`: `/chat` rewritten around the retrieve/router/generate
  flow above. `ChatResponse` gained `retrieved` and `router_decision`.
  `PAGE_URL_PREFIX` env var lets you retarget the image namespace
  (default `/aiwo-images`). `/query` unchanged in contract; it now goes
  through `generate_chat` under the hood.
- `generation/local_vlm_generator.py`: internal `_run(...)` helper factors
  out the tokenise-generate-decode path shared by `generate_chat` and
  `generate_text`. Switched `torch_dtype=` to the modern `dtype=` kwarg.
- `generation/claude_generator.py`, `generation/vlm_generator.py`:
  added `generate_chat(...)` and `generate_text(...)` with the same
  semantics (untested — no Claude / Azure credentials on hand).

### Verified behaviour
Two-turn tests on the Ponsse PDF, VLM_BACKEND=local:
- Turn 1 "How do I replace the hydraulic filter?" →
  router=`first-turn`, retrieved=True, correct 6-step procedure with
  p332/p325/p329 footer. Thumbnails render at 160 px in the UI.
- Turn 2 "Thanks!" → router=`FOLLOWUP`, retrieved=False, latency 3 ms,
  no new footer (history stays clean).
- Turn 2 "Can you explain step 4 again?" → router=`RETRIEVE` initially;
  after adding few-shot examples to `ROUTER_PROMPT` the model correctly
  classifies follow-ups that reference "step N" / "that / it" / "again".
- Turn 2 "How often should this be done?" → router=`RETRIEVE`,
  answers with "every other scheduled maintenance, max 3600 h" from a
  fresh top-k page.
- Turn 2 "What about the air filter instead?" → router=`RETRIEVE` (correct,
  different component).

## 2026-04-16 (final) — system prompt hardening + integration guide

### Changed
- `SYSTEM_PROMPT` in all three generators rewritten to a numbered rule list:
  no invented part numbers/torques/intervals; procedures as ≤8-step numbered
  lists; reference on-page labels like "bolt (5)"; surface
  DANGER/WARNING/CAUTION and torque/pressure/temperature/interval values;
  say so + name the likely manual section when the answer isn't on the pages;
  mirror the operator's language; keep it tight.

### Added
- `INTEGRATION_GUIDE.md` — single-document onboarding for a new frontend
  developer. Architecture diagram, full API reference (`/chat`, `/query`,
  `/health`, `/stats`, `/v1/models`, `/images/*`), a minimal integration
  example, the multi-turn page-memory protocol, URL-prefix conventions,
  data-prep workflow (what to commit vs. regenerate), the three generation
  backends side by side, env-var table, troubleshooting.
