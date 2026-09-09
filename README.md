# Filmecho

An AI "studio intelligence room" for a film title, upcoming or already released. Enter a title and get a verdict-first **Greenlight Memo**: a recommendation, a confidence score with the reasoning behind it, the biggest opportunity and biggest risk, and a live-rendered "War Room" of six studio personas actually discussing the evidence — not six independent one-liners, a real multi-turn discussion the synthesis model authors from the same data every persona is grounded in. Built for the Google Cloud Agentic Cinema Hackathon, Parallel track.

The pipeline is **release-status-aware**: it detects (grounded against today's actual date, not a guess) whether a title has already come out, and asks a fundamentally different question depending on the answer — "how's the trailer landing" for an upcoming title vs. "how did critical opinion hold up over time, and what can we learn from the campaign" for one that's already released.

## What you get

- **Greenlight Memo** — verdict (`greenlight` / `greenlight_with_changes` / `hold` / `pass` / `insufficient_data`), a confidence score with a one-sentence rationale naming which sources it's actually based on, biggest opportunity, biggest risk, and a recommended action.
- **The War Room** — a real, grounded multi-turn discussion among six personas (director, producer, marketing chief, casting executive, distribution executive, analyst), each speaking from their own upstream data, genuinely disagreeing when two sources' real data conflict — not scripted, not six isolated lines.
- **Reputation timeline** — dated milestones (announcement, casting reveals, trailer drops) pulled from both YouTube and news sources.
- **Release window advisory** — congestion read and a directional (not predictive) recommendation, checking both earlier *and* later release windows across the whole year, not just same-weekend competitors.
- **Cast reception, marketing analysis, production/cast news, franchise history** — each with its own evidence tab, sourced and linkable.
- **PDF export** — a real, laid-out document (ReportLab), not a browser print-to-PDF of the page.

## Architecture

```
Title input
    │
    ▼
Entity resolution (Parallel search + Gemini, structured output)
  → canonical title, year, director, release_status, source_type
    (sequel/reboot/spinoff/remake/original), confidence score,
    disambiguation note if the title is ambiguous
    │
    ├─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
    ▼             ▼             ▼             ▼             ▼             ▼
Web Sentiment  YouTube    Competitive    News & Cast    Cast        Marketing
(Parallel     (Discovery  (Parallel,     (Parallel,     Reception   (Parallel,
 search)      → Data,     year-wide +    extractive,    (Parallel   retrospective
              plain API,  same-window,   conflict-      search)     if released)
              no LLM)     + franchise    checked)
                          history if
                          sequel/reboot)
    │             │             │             │             │             │
    └─────────────┴─────────────┴─────┬───────┴─────────────┴─────────────┘
                                       ▼
                          Sentiment Synthesis (web + YouTube + news dates)
                                       │
                                       ▼
                    Main Synthesis (sentiment + competitive + news
                                     + cast + marketing)
                                       │
                                       ▼
                 refusal_guard → evidence_guard → memo_cache
                                       │
                                       ▼
                              { GreenlightMemo }
```

Every agent's output is schema-enforced via ADK's `output_schema` (Pydantic models in `agents/schemas.py`), not free text parsed with regex. Progress streams to the frontend over Server-Sent Events as each concurrent branch genuinely finishes (true completion order, via `asyncio.as_completed`, not a fixed/guessed sequence).

### Guardrails (deterministic code, not just prompt instructions)

An LLM instruction is a request, not an enforcement mechanism — these three modules are the actual, code-level backstops:

- **`orchestration/refusal_guard.py`** — recursively scans every agent's dumped output for refusal-shaped text ("I can't...", "I don't have...") and drops that branch to `None`, the same degrade path as an API failure, so a model refusal never silently ends up looking like a real (if odd) answer.
- **`orchestration/evidence_guard.py`** — after `main_synthesis` returns, counts how many of the 5 upstream sources actually had data. Below 3 of 5, it force-overrides the verdict to `insufficient_data` and caps confidence at 25, regardless of what the model produced — a low-confidence number next to a normal-looking verdict badge still reads as a real recommendation to someone skimming.
- **`orchestration/memo_cache.py`** — released titles are cached indefinitely (nothing about the past changes); upcoming/unclear titles are cached for a short TTL (`FILMECHO_UPCOMING_CACHE_TTL_SECONDS`, default 20 min) so the same query doesn't visibly re-roll different results a minute apart. A "Refresh this memo" action in the UI bypasses the cache on demand.

## Repo structure

```
filmecho/
├── README.md
├── requirements.txt
├── .env.example                   # copy to .env and fill in real values
├── Dockerfile                     # Cloud Run build target
├── .dockerignore
├── .gitignore
│
├── agents/
│   ├── __init__.py                # exposes each Agent for `adk web`/`adk run` discovery
│   ├── schemas.py                 # Pydantic models — every agent's output_schema
│   ├── entity_context.py          # title → EntityContext (confidence, release_status, source_type)
│   ├── web_sentiment_agent.py     # Parallel search, release-status-aware
│   ├── competitive_agent.py       # Parallel search — same-window + year-wide competition, franchise history
│   ├── news_cast_agent.py         # Parallel search — extractive, cross-checks its own results for conflicts
│   ├── cast_agent.py              # Parallel search — per-actor performance reception
│   ├── marketing_agent.py         # Parallel search — campaign analysis + retrospective lessons
│   ├── sentiment_synthesis.py     # combines web + YouTube sentiment + news dates → timeline
│   ├── main_synthesis.py          # combines everything into the Greenlight Memo + War Room discussion
│   ├── youtube_discovery_agent.py # YouTube search.list → candidate trailer video IDs (plain API, no LLM)
│   ├── youtube_data_agent.py      # YouTube videos.list + commentThreads.list (plain API, no LLM)
│   └── poster_agent.py            # TMDB poster lookup — NOT wired into the pipeline (see below)
│
├── orchestration/
│   ├── pipeline.py                # stream_pipeline() (SSE source of truth) + run_pipeline()
│   ├── memo_cache.py              # release-status-aware result cache
│   ├── refusal_guard.py           # drops any branch whose output reads as a model refusal
│   └── evidence_guard.py          # deterministic verdict/confidence override on thin evidence
│
├── backend/
│   ├── app.py                     # FastAPI: /api/brief, /api/brief/stream, /api/brief/pdf, /api/cache/stats, serves static/
│   ├── pdf_report.py              # ReportLab PDF export, built from the same payload the frontend gets
│   └── static/
│       └── index.html             # entire frontend, single file, vanilla JS, no build step
│
└── tests/                         # pytest — schemas, guardrails, cache, entity resolution degrade paths
```

**`agents/poster_agent.py` is inactive infrastructure**, not dead weight left by accident: it was built, then the frontend was deliberately reverted to using YouTube trailer thumbnails instead of TMDB posters, so `pipeline.py` no longer calls it. It's left in the repo (with its own passing test) in case posters get re-enabled later; `TMDB_API_KEY` is optional and only matters if you wire it back in yourself.

**Not built**: a Reddit sentiment agent (lowest priority in the original plan — the agents above already cover trailer/critic sentiment, YouTube comments, competitive positioning, production news, cast reception, and marketing).

## Setup

```bash
git clone <this repo>
cd filmecho
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Then fill in `.env` — see the table below and the comments in `.env.example` itself for what each variable does and where to get it.

### Environment variables

| Variable | Required? | Used by | Notes |
|---|---|---|---|
| `PARALLEL_API_KEY` | **Yes** | every `agents/*_agent.py` that calls `parallel.search()` | From platform.parallel.ai |
| `YOUTUBE_API_KEY` | **Yes** | `youtube_discovery_agent.py`, `youtube_data_agent.py` | YouTube Data API v3 must be enabled on the GCP project |
| `GOOGLE_GENAI_USE_VERTEXAI` + `GOOGLE_CLOUD_PROJECT` + `GOOGLE_CLOUD_LOCATION` | **Yes** (Option A) | Every Gemini call, via ADK | Routes through Vertex AI + your GCP project's own billing |
| `GEMINI_API_KEY` | **Yes** (Option B, instead of the three above) | Every Gemini call | AI Studio — a *separate* prepay balance, does not draw on Cloud Billing credit |
| `FILMECHO_GEMINI_MODEL` | No (default `gemini-2.5-flash`) | Every agent | **Verify against your own project before trusting it** — model availability differs between AI Studio/Vertex and between projects/regions |
| `FILMECHO_TEMPERATURE` | No (default `0.1`) | Every agent's `generate_content_config` | Lower = more consistent/grounded |
| `FILMECHO_UPCOMING_CACHE_TTL_SECONDS` | No (default `1200`) | `memo_cache.py` | Only affects upcoming/unclear titles — released titles cache indefinitely regardless |
| `TMDB_API_KEY` | No, unused by default | `poster_agent.py` | Only matters if you re-wire `fetch_poster()` back into `pipeline.py` yourself |

For deployment, the two secrets (`PARALLEL_API_KEY`, `YOUTUBE_API_KEY`) become Cloud Run `--set-secrets` via Secret Manager, and the rest become `--set-env-vars` — see [Deploy to Cloud Run](#deploy-to-cloud-run).

## Run locally

```bash
uvicorn backend.app:app --reload --port 8080
```

Then open `http://localhost:8080`.

Or run the pipeline directly without the web layer:

```bash
python -m orchestration.pipeline
```

(edit the hardcoded title in `pipeline.py`'s `__main__` block first)

## Testing

```bash
python -m pytest tests/ -v
```

Exercises the Pydantic schemas directly (rejecting invalid enum values, structural constraints), the three guardrail modules (`refusal_guard`, `evidence_guard`, `memo_cache`) in isolation, `poster_agent`, and `entity_context.resolve_entity`'s degrade paths via injected mock clients (`parallel_client`/`genai_client` params exist specifically for this). No API keys or network access required for the test suite itself — but it does require `google-adk` and the rest of `requirements.txt` actually installed, since several agent modules import `google.adk.agents.Agent` at module load time.

This does **not** test the live agent pipeline end to end — that needs real API keys and is what `python -m orchestration.pipeline` (or a real browser session against the running server) is for.

## API call volume — know this before a live demo

One full `run_pipeline()` call makes:
- **13-14 Gemini calls**: five agents use tools (`web_sentiment`, `competitive`, `news_cast`, `cast`, `marketing`), and ADK's function-calling is a two-turn round trip per tool-using agent (decide to call the tool, then produce the final answer once the tool result is back) — 2 calls × 5 agents = 10, plus entity resolution (1, or 2 if a disambiguation retry fires), plus sentiment synthesis (1), plus main synthesis (1).
- **6-7 Parallel searches**: one per tool-using agent, plus entity resolution's own search (plus its retry, if triggered). The competitive agent alone fires 2-3 queries in one `search()` call (same-window + year-wide + franchise history if applicable) — still one billed search call, just a richer objective.
- YouTube: 1 `search.list` + 1 `videos.list` + up to 5 `commentThreads.list` calls (plain REST, no LLM involved).

Check actual per-call pricing on `platform.parallel.ai`'s dashboard and Vertex AI's billing page directly rather than assume, and budget test runs accordingly if you're demoing repeatedly against a limited credit. `memo_cache.py` means repeat searches of the *same* title within its TTL don't re-spend any of this.

## Deploy to Cloud Run

```bash
gcloud run deploy filmecho \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="GOOGLE_GENAI_USE_VERTEXAI=True,GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID,GOOGLE_CLOUD_LOCATION=us-central1,FILMECHO_GEMINI_MODEL=gemini-2.5-flash" \
  --set-secrets="PARALLEL_API_KEY=parallel-api-key:latest,YOUTUBE_API_KEY=youtube-api-key:latest" \
  --memory 1Gi \
  --timeout 300
```

Requires, before this will work:
- Vertex AI API and Secret Manager API enabled on the project.
- The two secrets already created in Secret Manager (`parallel-api-key`, `youtube-api-key` — the resource names are lowercase-hyphenated; they don't need to match the uppercase env var names the code reads, `--set-secrets` is what connects them).
- The Cloud Run service's default compute service account (`{project-number}-compute@developer.gserviceaccount.com` — a *different* identity than your local `gcloud auth` user) granted `roles/secretmanager.secretAccessor` and `roles/cloudbuild.builds.builder` (the latter fixes a "could not resolve source" error during `--source .` builds).

**Recommend `--min-instances 1` for a live demo specifically** — `MemoCache` and ADK's `InMemorySessionService` both live in process memory, so a Cloud Run cold start silently empties the cache and any in-flight session state.

**Test the actual deployed `*.run.app` URL after deploying**, not just a local/Cloud Shell preview — in particular, confirm the SSE progress stream (`/api/brief/stream`) delivers events incrementally on the deployed URL and not all at once at the end; some managed platforms buffer streaming responses differently than local dev servers do. Also hard-refresh (or use a private/incognito window) when checking a redeploy — browsers can and do cache the static frontend bundle between deploys.

## What's actually enforced vs. what to double-check yourself

**Enforced by the code, not just hoped for:**
- Every agent's output shape (Pydantic `output_schema`, framework-level validation)
- Entity resolution confidence + disambiguation for ambiguous titles
- `lessons_learned`/`what_we_would_change` only populate for the correct release status, never guessed for the wrong one
- The evidence threshold on the final verdict badge (`evidence_guard.py`) — a model-produced verdict on thin evidence gets overridden in code, not just discouraged in the prompt
- Graceful degradation: any single agent failing (or refusing) returns `None` for that field rather than crashing the whole run or silently leaking refusal text

**Not independently verified — check before relying on them:**
- ADK's `output_schema` + `tools` combination on the same agent (used by every fetch agent) is documented as supported but wasn't tested against every model/region combination
- SSE streaming behavior specifically on Cloud Run's infrastructure (works locally; verify post-deploy)
- Whether the per-title API cost (6-7 Parallel searches, ~13-14 Gemini calls) stays comfortably within your Google Cloud credit for repeated demo runs

## Findings / learnings

- Google's Gemini Developer API (AI Studio) and Vertex AI are billed through **entirely separate systems** — a Google Cloud promotional credit does not apply to AI Studio's prepay balance unless you first fund that balance yourself, unlike Vertex AI billing, which draws on ordinary Cloud Billing credits directly. This is easy to miss and will silently 429 you.
- `asyncio.as_completed` (vs. `asyncio.gather`) turned out to matter for more than just "how the code looks" — it's what lets the frontend show honest, real-time progress instead of a fixed timer that just guesses how long five parallel agents will take.
- Structured output (`output_schema`) should have been the default from the start rather than added after a schema-drift bug surfaced in testing — free-text-then-regex-parse is exactly the kind of thing that looks fine until an LLM decides to nest a field somewhere slightly different.
- A prompt instruction is a request, not a guarantee: the model was explicitly told never to show a confident-looking verdict on thin evidence, and still did it in testing. `evidence_guard.py` exists because "don't do X" in a prompt is not the same as X being actually impossible.
- Five independently-researched Parallel Search branches genuinely can (and do) disagree with each other about the same real-world fact — e.g. cast reception reporting an actor as returning while production/cast news reports the opposite. That's not a bug to average away; `main_synthesis.py`'s cross-check instruction and the War Room's `disagreement`-flagged turns are built to surface it honestly instead of silently picking whichever framing sounded more confident.
