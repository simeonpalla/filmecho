# Filmecho

An agent pipeline that produces a two-audience brief (studio + fan) for a film title, upcoming or already released. Built for the Google Cloud Agentic Cinema Hackathon, Parallel track.

Enter a title, and six agents run concurrently, backed by real-time Parallel Search and YouTube Data API calls, synthesized by Gemini through Google's Agent Development Kit (ADK), into:

- **Studio brief** — sentiment summary, competitive risk, notable news, a recommendation, and (for already-released titles) concrete lessons learned.
- **Fan pulse** — excitement level, recurring themes, a fun fact, and (for already-released titles) a "worth watching now?" verdict.
- **Cast reception** — per-actor performance notes, not just aggregate sentiment.
- **Marketing** — campaign strategies observed, what worked, what underperformed, and, for released titles, a genuine retrospective with lessons learned.

The pipeline is **release-status-aware**: it detects (grounded against today's actual date, not a guess) whether a title has already come out, and asks a fundamentally different question depending on the answer — "how's the trailer landing" for an upcoming title vs. "how did critical opinion hold up over time, and what can we learn from the campaign" for one that's already released.

## Architecture

```
Title input
    │
    ▼
Entity resolution (Parallel search + Gemini, structured output)
  → canonical title, year, director, cast, release_status,
    confidence score, disambiguation note if the title is ambiguous
    │
    ├─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
    ▼             ▼             ▼             ▼             ▼             ▼
Web Sentiment  YouTube    Competitive    News & Cast    Cast        Marketing
(Parallel)   (Discovery   (Parallel)     (Parallel,     (Parallel)  (Parallel,
              → Data)                    extractive)                retrospective
                                                                     if released)
    │             │             │             │             │             │
    └─────────────┴─────────────┴─────┬───────┴─────────────┴─────────────┘
                                       ▼
                          Sentiment Synthesis (web + YouTube)
                                       │
                                       ▼
                    Main Synthesis (sentiment + competitive + news
                                     + cast + marketing)
                                       │
                                       ▼
                    { studio_brief, fan_pulse, sources_used }
```

Every agent's output is schema-enforced via ADK's `output_schema` (Pydantic models in `agents/schemas.py`), not free text parsed with regex — a version of this pipeline that did that drifted from its intended shape in testing, so structured output is enforced at the framework level throughout.

Progress streams to the frontend over Server-Sent Events as each of the six concurrent agents genuinely finishes (true completion order, via `asyncio.as_completed`, not a fixed/guessed sequence).

## Repo structure

```
filmecho/
├── LICENSE                        # MIT — replace the placeholder name before submitting
├── README.md
├── requirements.txt
├── Dockerfile                     # Cloud Run build target
├── .dockerignore
├── .gitignore
│
├── agents/
│   ├── __init__.py                # exposes each Agent for `adk web`/`adk run` discovery
│   ├── schemas.py                 # Pydantic models — every agent's output_schema
│   ├── entity_context.py          # title → EntityResolution (confidence, release_status)
│   ├── web_sentiment_agent.py     # Parallel search, release-status-aware
│   ├── competitive_agent.py       # Parallel search, release-status-aware
│   ├── news_cast_agent.py         # Parallel search, extractive-only
│   ├── cast_agent.py              # Parallel search — per-actor performance reception
│   ├── marketing_agent.py         # Parallel search — campaign analysis + retrospective lessons
│   ├── sentiment_synthesis.py     # combines web + YouTube sentiment feeds
│   ├── main_synthesis.py          # combines everything into studio_brief/fan_pulse
│   ├── youtube_discovery_agent.py # YouTube search.list → candidate trailer video IDs
│   └── youtube_data_agent.py      # YouTube videos.list + commentThreads.list
│
├── orchestration/
│   └── pipeline.py                # stream_pipeline() (SSE source of truth) + run_pipeline()
│
└── backend/
    ├── app.py                     # FastAPI: /api/brief, /api/brief/stream, serves static/
    └── static/
        └── index.html             # entire frontend, single file, no build step
```

**Not built**: Reddit Sentiment Agent (lowest priority per original build order — the six agents above already cover trailer/critic sentiment, YouTube comments, competitive positioning, production news, cast reception, and marketing, which is a complete submission on its own).

## Setup

```bash
pip install -r requirements.txt
```

### Required environment variables (`.env`, never committed)

| Variable | Used by | Notes |
|---|---|---|
| `PARALLEL_API_KEY` | every `*_agent.py` that calls `parallel.search()` | From platform.parallel.ai |
| `YOUTUBE_API_KEY` | `youtube_discovery_agent.py`, `youtube_data_agent.py` | YouTube Data API v3 must be enabled on the GCP project |
| `GOOGLE_GENAI_USE_VERTEXAI` | Gemini calls (via ADK + `entity_context.py`) | `True` to route through Vertex AI + your GCP project's billing, instead of a separate AI Studio prepay balance |
| `GOOGLE_CLOUD_PROJECT` | same | Your GCP project ID |
| `GOOGLE_CLOUD_LOCATION` | same | e.g. `us-central1` |
| `FILMECHO_GEMINI_MODEL` | same | **Verify this against your own project before trusting it** — model availability differs between AI Studio and Vertex, and between projects/regions. `gemini-2.5-flash` is confirmed working as of this project's testing; don't assume a newer-sounding name (`gemini-3.x`) is available to you without checking. |

Local (non-Vertex) alternative: set `GEMINI_API_KEY` instead of the three `GOOGLE_*` Vertex variables, and omit `GOOGLE_GENAI_USE_VERTEXAI`. Note this uses AI Studio's separate prepay billing, which does **not** draw from Google Cloud promotional credits by default.

For deployment, these become Cloud Run `--set-secrets` (for `PARALLEL_API_KEY`/`YOUTUBE_API_KEY`, via Secret Manager) and `--set-env-vars` (for the Vertex config) — see below.

## Run locally

```bash
uvicorn backend.app:app --reload --port 8080
```

Or run the pipeline directly without the web layer:

```bash
python -m orchestration.pipeline
```

(edit the hardcoded title in `pipeline.py`'s `__main__` block first)

## Deploy to Cloud Run

```bash
gcloud run deploy filmecho \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="GOOGLE_GENAI_USE_VERTEXAI=True,GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID,GOOGLE_CLOUD_LOCATION=us-central1,FILMECHO_GEMINI_MODEL=gemini-2.5-flash" \
  --set-secrets="PARALLEL_API_KEY=parallel-api-key:latest,YOUTUBE_API_KEY=youtube-api-key:latest"
```

Requires: Vertex AI API and Secret Manager API enabled on the project, the two secrets already created in Secret Manager, and the Cloud Run service's own service account granted `roles/secretmanager.secretAccessor` on both secrets and `roles/aiplatform.user` on the project — this is a *different* identity than your local `gcloud auth` user, grant it explicitly, don't assume it inherits your permissions.

**Test the actual deployed `*.run.app` URL after deploying**, not just a local/Cloud Shell preview — in particular, confirm the SSE progress stream (`/api/brief/stream`) delivers events incrementally on the deployed URL and not all at once at the end, some managed platforms buffer streaming responses differently than local dev servers do.

## What's actually enforced vs. what to double-check yourself

**Enforced by the code, not just hoped for:**
- Every agent's output shape (Pydantic `output_schema`, framework-level validation)
- Entity resolution confidence + disambiguation for ambiguous titles (e.g., a title shared by multiple real films)
- `lessons_learned`/`worth_watching` only populate for confirmed-released titles, never guessed for something still upcoming
- Graceful degradation: any single agent failing returns `None` for that field rather than crashing the whole run

**Not independently verified — check before relying on them:**
- ADK's `output_schema` + `tools` combination on the same agent (used by every fetch agent) is documented as supported but wasn't tested against every model/region combination
- SSE streaming behavior specifically on Cloud Run's infrastructure (works locally; verify post-deploy)
- Whether the six-agent, per-title API cost (3+ Parallel searches, ~7 Gemini calls) stays comfortably within your Google Cloud credit for repeated demo runs

## Findings / learnings (useful for the Devpost submission text)

- Google's Gemini Developer API (AI Studio) and Vertex AI are billed through **entirely separate systems** — a Google Cloud promotional credit does not apply to AI Studio's prepay balance unless you first fund that balance yourself, unlike Vertex AI billing, which draws on ordinary Cloud Billing credits directly. This is easy to miss and will silently 429 you.
- `asyncio.as_completed` (vs. `asyncio.gather`) turned out to matter for more than just "how the code looks" — it's what lets the frontend show honest, real-time progress instead of a fixed timer that just guesses how long five parallel agents will take.
- Structured output (`output_schema`) should have been the default from the start rather than added after a schema-drift bug surfaced in testing — free-text-then-regex-parse is exactly the kind of thing that looks fine until an LLM decides to nest a field somewhere slightly different.
