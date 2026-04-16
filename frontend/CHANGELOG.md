# Changelog

Aalto-side modifications to the TUNI `aiwo-visual-document-qa` frontend.

## 2026-04-16 — AiWo RAG provider (minimal-diff)

Goal: surface the Aalto-hosted visual RAG backend
(`aiwo-vdr-backend`, /chat endpoint) as a selectable provider in the
existing dropdown, alongside Azure Anthropic. No UI changes.

### Added
- `lib/providers.ts`:
  - New `createAiwoRagProvider()` factory. Registered only when
    `AIWO_RAG_BASE_URL` env var is set. POSTs
    `{messages, provider: "aiwo-rag", model}` to `${AIWO_RAG_BASE_URL}/chat`
    and reads `data.result.content`.
  - Registered in `providerFactories` between Anthropic and OpenAI so it
    shows up when configured but doesn't shadow the primary Azure path.

### Changed
- `pages/api/chat.ts`:
  - Added `"aiwo-rag"` to `ALLOWED_PROVIDER_NAMES` so the server-side
    validator accepts requests targeting the new provider.

### Env vars (new)
- `AIWO_RAG_BASE_URL` — e.g. `http://aalto-host:8000`. Required to activate.
- `AIWO_RAG_MODEL_ID` (optional, default `aiwo-rag`).
- `AIWO_RAG_MODEL_NAME` (optional, default `AiWo RAG (Ponsse)`).

### Not changed
- `pages/index.tsx` — the UI reads `/api/providers` and auto-lists whatever
  the backend returns, so the new provider appears in the model dropdown
  without template changes.
- Any other file.

## 2026-04-16 (evening) — page thumbnails support

### Added
- `next.config.js`: `/aiwo-images/:path*` rewrite → `${AIWO_RAG_BASE_URL}/images/:path*`.
  Serves backend page PNGs from the frontend origin so the user only needs
  one SSH tunnel and the backend's CORS isn't in the picture.
- `styles/Home.module.css`: `.markdownanswer img` sized to 160×220 max so
  the thumbnails the backend embeds in assistant messages don't overflow
  the chat bubble.

### Changed
- `pages/index.tsx` (L370): passed a `components={{ a: … }}` override to
  `ReactMarkdown` so every link in assistant content opens in a new tab
  with `target="_blank" rel="noopener noreferrer"`. Without this, clicking
  a page thumbnail navigated the whole SPA to the image and lost the chat.
- `pages/index.tsx` `speak()`: split assistant content on the first
  `\n---\n` and only speak the part above it — the RAG page footer is for
  the eyes, not the ears. Also added an upfront `!\[…\]\(…\)` strip so any
  inline image markdown is removed before TTS.
