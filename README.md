# AiWo Visual Document QA Backend

Visual document retrieval + VLM-powered QA for equipment manuals. Uses [NanoVDR](https://arxiv.org/abs/2603.12824) for fast document retrieval and Azure OpenAI GPT-4o for answer generation.

## Architecture

```
[Frontend - TUNI]                    [Backend - Aalto]
Speech (Whisper) → Text query  ───→  POST /query
                                      ├── NanoVDR-S-Multi (69M, CPU, ~51ms)
                                      │   └── Cosine similarity vs pre-indexed pages
                                      ├── Top-k page images
                                      └── GPT-4o Vision (Azure OpenAI)
                                           └── Answer based on retrieved pages
                                ←───  { answer, retrieved_pages[] }
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Index PDF manuals

```bash
# Requires GPU for Qwen3-VL-Embedding-2B (one-time indexing)
python scripts/index_documents.py \
    --pdf_dir data/manuals/ \
    --output_dir data/index/ \
    --image_dir data/images/
```

### 3. Configure Azure OpenAI

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-key"
export VLM_DEPLOYMENT="gpt-4o"  # or your deployment name
```

### 4. Start server

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

## API

### POST /query

```json
// Request
{
    "query": "How to replace the hydraulic filter?",
    "top_k": 5,
    "generate_answer": true,
    "max_pages_for_vlm": 3
}

// Response
{
    "query": "How to replace the hydraulic filter?",
    "answer": "According to the manual (page 45), to replace the hydraulic filter: 1. ...",
    "retrieved_pages": [
        {
            "page_id": "manual_p0045",
            "score": 0.89,
            "image_url": "/images/manual_p0045.png",
            "source": "operator_manual.pdf",
            "page_number": 45
        }
    ],
    "retrieval_latency_ms": 51.2,
    "total_latency_ms": 3200.5
}
```

### GET /health

```json
{"status": "ok", "model": "nanovdr/NanoVDR-S-Multi"}
```

### GET /stats

```json
{"index_size": 500, "embedding_dim": 2048, "model": "nanovdr/NanoVDR-S-Multi"}
```

## Project Structure

```
aiwo-vdr-backend/
├── api/
│   └── server.py          # FastAPI endpoints
├── retrieval/
│   └── retriever.py       # NanoVDR document retrieval
├── generation/
│   └── vlm_generator.py   # Azure OpenAI GPT-4o answer generation
├── scripts/
│   └── index_documents.py # PDF → page images → embeddings index
├── data/
│   ├── manuals/           # Input PDFs (add your manuals here)
│   ├── images/            # Extracted page images (auto-generated)
│   └── index/             # Embeddings + metadata (auto-generated)
├── requirements.txt
└── README.md
```

## Frontend Integration

The frontend (TUNI team) sends text queries via `POST /query`. The text typically comes from Whisper speech-to-text. The backend returns:
- `answer`: VLM-generated answer based on retrieved manual pages
- `retrieved_pages`: List of relevant pages with scores and image URLs
- Page images accessible via `/images/{filename}`

## Acknowledgements

This project has received funding from the **Business Finland** co-innovation programme under grant agreement No. 69/31/2025, supported by the [AiWo](https://aifieldwork.aalto.fi/events/) project (2025–2027).
