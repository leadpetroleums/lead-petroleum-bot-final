# Lead Petroleum — AI Product Assistant (local-first build)

A local-first, open-source RAG chatbot for the Lead Petroleum product catalogue. No paid APIs. Runs entirely on your own machine or server using Ollama + ChromaDB + FastAPI + Streamlit.

---

## What this does

Buyers ask natural questions about Lead Petroleum lubricants and get grounded answers citing the exact catalogue page. The system handles four kinds of questions:

- **Recommendations** — "What oil for my diesel truck with API CK-4?"
- **Comparisons** — "Compare MEGA 7000 vs GIGA 10000"
- **Exact specs** — "What's the pour point of YOTTA 15000?"
- **General queries** — "What certifications does Lead Petroleum hold?"

Every answer cites the source product and page number. Exact spec queries are answered from a hand-curated structured table (no LLM involvement, zero hallucination risk) or, if the data isn't curated yet, the bot directs the user to the catalogue page rather than guessing.

---

## Quick start (local development)

### 1. Install Python dependencies

```bash
cd project
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install Ollama and pull a model

Download Ollama from [ollama.com](https://ollama.com), then:

```bash
ollama serve                  # runs in background, or open the desktop app
ollama pull llama3.2:3b       # ~2GB download, ~4GB RAM to run
```

Alternative models if you have more RAM or a GPU:

| Model          | RAM    | Quality  | Speed (CPU)  |
|----------------|--------|----------|--------------|
| `phi3:mini`    | ~3GB   | Decent   | Fastest      |
| `llama3.2:3b`  | ~4GB   | Good     | Fast         |
| `mistral:7b`   | ~5GB   | Better   | Moderate     |
| `llama3.1:8b`  | ~6GB   | Best     | Slow on CPU  |

Set the model with `OLLAMA_MODEL` in `.env` or the environment.

### 3. Add the catalogue PDF

Drop the catalogue at `data/catalogue.pdf`.

### 4. Ingest the catalogue

```bash
python scripts/ingest.py
```

This extracts all 48 products into `data/catalogue.json` and indexes them into `data/chroma_db/`. Takes about 30 seconds.

### 5. Run the chatbot

Either the CLI:

```bash
python ui/cli.py
```

Or the web UI:

```bash
streamlit run ui/app.py
```

Or the API server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 6. Run the tests (optional)

```bash
python tests/test_queries.py
```

Tests run with a stub LLM so they work without Ollama.

---

## Architecture

```
                       ┌─────────────────────┐
                       │      User query     │
                       └──────────┬──────────┘
                                  │
                        ┌─────────▼──────────┐
                        │ Intent classifier  │  ← rule-based, no LLM
                        │  (intent.py)       │
                        └─┬─────┬──────┬─────┘
                          │     │      │
              ┌───────────┘     │      └────────────┐
              │                 │                   │
      ┌───────▼──────┐  ┌───────▼────────┐  ┌──────▼──────┐
      │ EXACT_SPEC   │  │ COMPARE /      │  │ GENERAL     │
      │ handler      │  │ RECOMMEND      │  │ handler     │
      │              │  │ handler        │  │             │
      │ Manual specs │  │                │  │             │
      │ CSV lookup   │  │ RAG + LLM      │  │ RAG + LLM   │
      │ (no LLM)     │  │                │  │             │
      └──────┬───────┘  └────────┬───────┘  └──────┬──────┘
             │                   │                 │
             └───────────────────┼─────────────────┘
                                 │
                       ┌─────────▼──────────┐
                       │ ChromaDB vector    │
                       │ search (local)     │
                       └─────────┬──────────┘
                                 │
                       ┌─────────▼──────────┐
                       │ Ollama LLM (local) │
                       │ llama3.2:3b        │
                       └─────────┬──────────┘
                                 │
                       ┌─────────▼──────────┐
                       │ Cited answer       │
                       │ with page refs     │
                       └────────────────────┘
```

### Key design decisions

**Intent routing prevents hallucinated specs.** Exact spec queries never touch the LLM — they hit a structured lookup in `data/manual_specs.csv`. If the data isn't curated yet, the bot directs users to the catalogue page rather than guessing a number.

**One chunk per product for retrieval.** Product spec sheets are self-contained. Splitting them mid-section destroys context. The 48 products fit comfortably into ChromaDB and the top-5 retrieval almost always includes the right product.

**Local embeddings.** Uses the default `all-MiniLM-L6-v2` sentence transformer that ships with ChromaDB. No external API calls for embedding.

**Streaming responses.** Tokens stream as the LLM generates them. Perceived latency is much better than waiting for a full response — especially important on CPU inference where full generation can take 5–20 seconds.

---

## Project structure

```
project/
├── data/
│   ├── catalogue.pdf          ← drop the source PDF here
│   ├── catalogue.json         ← generated by ingest.py
│   ├── manual_specs.csv       ← hand-curated exact specs (optional)
│   └── chroma_db/             ← generated vector store
├── scripts/
│   ├── ingest.py              ← PDF → JSON → vectors
│   └── update_knowledge.py    ← re-ingest on PDF change
├── api/
│   ├── intent.py              ← intent classifier (rules)
│   ├── rag.py                 ← RAG pipeline
│   ├── llm.py                 ← Ollama adapter
│   └── main.py                ← FastAPI server
├── ui/
│   ├── cli.py                 ← CLI chatbot
│   └── app.py                 ← Streamlit web UI
├── tests/
│   └── test_queries.py        ← adversarial test suite
├── requirements.txt
├── .env.example
└── README.md
```

---

## Updating the catalogue

When Lead Petroleum releases a new catalogue version:

1. Replace `data/catalogue.pdf` with the new file.
2. Run `python scripts/update_knowledge.py` (one-shot re-ingest).

Or run `python scripts/update_knowledge.py --watch` to leave it running in the background — it will auto-rebuild whenever the PDF changes.

---

## Curating exact specs (optional but recommended)

The catalogue's PDF spec tables are rendered as images by Canva, which means OCR is unreliable. The most accurate path is to hand-curate specs for the top products buyers ask about most. Edit `data/manual_specs.csv` — the format is one row per (product, viscosity grade). See the file header for the full format.

Start with 10–20 of the most-asked products; expand as you see query logs. Products without curated specs still work for recommendations and descriptions — they just can't answer exact numeric questions directly.

---

## REST API

When running via `uvicorn`, the following endpoints are available:

| Method | Path                        | Purpose                              |
|--------|-----------------------------|--------------------------------------|
| GET    | `/health`                   | Health check                          |
| POST   | `/api/chat`                 | Chat endpoint (streams tokens)        |
| GET    | `/api/products`             | List products (filter by category/API) |
| GET    | `/api/products/{id}`        | Full product details                  |
| GET    | `/api/categories`           | Category list with counts             |

Example:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What oil for diesel trucks?", "stream": false}'
```

---

## Deployment on Hostinger VPS (for production)

### Hardware sizing

CPU-only inference with `llama3.2:3b`:

| VPS plan             | Response time   | Concurrent users | Notes                     |
|----------------------|-----------------|------------------|---------------------------|
| Hostinger KVM 2 (8GB)| 8–15s           | 1–2              | Tight, OK for soft launch |
| Hostinger KVM 4 (16GB)| 5–10s         | 3–5              | **Recommended starting point** |
| Hostinger KVM 8 (32GB)| 3–7s          | 5–10             | Comfortable at scale      |

If the client later provisions a GPU VPS, response time drops to 1–2 seconds and `llama3.1:8b` becomes practical with much better answer quality.

### Deploy steps

1. SSH into the Hostinger VPS
2. Install Python 3.11+, Ollama (`curl -fsSL https://ollama.com/install.sh | sh`)
3. Clone this repo, `pip install -r requirements.txt`
4. Drop `catalogue.pdf` in `data/`, run `python scripts/ingest.py`
5. Run as a systemd service (see `deploy/petroleum-bot.service` — add as needed)
6. Put Nginx or Caddy in front with TLS
7. Point the ASP.NET frontend to the API URL

### Production checklist

- [ ] Lock down CORS origins in `api/main.py` to the production domain
- [ ] Set up log rotation for Ollama and the API
- [ ] Rate-limit `/api/chat` (suggest 10 requests/min per IP)
- [ ] Monitor RAM — if Ollama OOMs, switch to `phi3:mini`
- [ ] Back up `data/catalogue.json` and `data/manual_specs.csv`
- [ ] Schedule weekly re-ingestion if the catalogue changes often

---

## Integrating with an ASP.NET Core frontend

The API is a plain REST service. From ASP.NET:

```csharp
var client = new HttpClient();
var response = await client.PostAsJsonAsync(
    "http://your-vps:8000/api/chat",
    new { query = userQuery, stream = false }
);
var answer = await response.Content.ReadFromJsonAsync<ChatResponse>();
```

For streaming responses, consume the response as a `Stream` and forward to the browser via SignalR or Server-Sent Events. A minimal JS widget for the site footer can be added as a separate deliverable.

---

## Known limitations

- **No long-term chat memory.** Each query is answered in isolation. Multi-turn conversations are possible but not built yet — would need a session store (Redis).
- **Spec tables in the PDF are images, not text.** The exact numeric spec table on each product page is rendered by Canva as a raster image. This system extracts product descriptions, applications, performance levels, API ratings, certifications, and viscosity grades from text layers; exact numeric spec values come from the optional `manual_specs.csv` or a page-reference pointer.
- **Local model quality.** Small open-source models (3B–8B parameters) are weaker at technical grounding than frontier paid APIs. The intent router and spec-lookup structure compensate for this on the highest-risk queries (exact specs), but recommendation quality is noticeably better with GPT-4o or Claude. If accuracy becomes an issue after launch, the `llm.py` adapter can swap to a paid API without touching the rest of the system.
- **English-only right now.** Urdu/Arabic support can be added by switching to a multilingual embedding model (e.g. `paraphrase-multilingual-MiniLM-L12-v2`) and adding a translation layer in `rag.py`. About a day of work.

---

## License / ownership

This system uses only open-source components: Ollama (Apache 2), ChromaDB (Apache 2), FastAPI (MIT), Streamlit (Apache 2), Llama 3 (Meta Llama license — permits commercial use). No runtime dependency on paid APIs.
