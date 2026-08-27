# Filmecho — agent pipeline (Web Sentiment + YouTube pair)

This is the first build increment per the roadmap: entity resolution, the
Parallel-backed Web Sentiment Agent, and the sequential YouTube Discovery →
Data pair, running concurrently with each other.

## Setup

```bash
pip install -r requirements.txt
```

Required environment variables:

| Variable | Used by | Notes |
|---|---|---|
| `PARALLEL_API_KEY` | entity_context.py, web_sentiment_agent.py | From platform.parallel.ai |
| `GEMINI_API_KEY` | entity_context.py, web_sentiment_agent.py (via ADK) | Gemini Developer API key. If deploying on Vertex AI instead, set `GOOGLE_GENAI_USE_VERTEXAI=True` and configure ADC instead of this key. |
| `YOUTUBE_API_KEY` | youtube_discovery_agent.py, youtube_data_agent.py | From Google Cloud Console, YouTube Data API v3 enabled |
| `FILMECHO_GEMINI_MODEL` | optional | Defaults to `gemini-3-flash`. Override if your project pins a different Gemini version. |

Never hardcode these; use Secret Manager in deployment (see roadmap Section 2).

## Run

```bash
python -m orchestration.pipeline
```

This resolves the entity context for the hardcoded demo title in
`pipeline.py`'s `__main__` block, then runs both branches and prints the
combined JSON result. Swap the title or wire this into your frontend.

## What's here vs. what's next

Built:
- `agents/entity_context.py` — title → structured facts (Parallel + Gemini)
- `agents/web_sentiment_agent.py` — the Parallel track-requirement call, wrapped as a Gemini ADK agent
- `agents/youtube_discovery_agent.py` + `agents/youtube_data_agent.py` — the sequential YouTube pair
- `orchestration/pipeline.py` — ties it together, parallelizing Web Sentiment against the YouTube pair

Not built yet (see roadmap for order): Main Synthesis Agent, Competitive
Agent, News & Cast Agent, Reddit Sentiment Agent, frontend, demo video.

## Known gaps / things to verify before Stage One submission

- `entity_context.py` calls Gemini directly via `google-genai`. If your
  Stage One reviewers specifically want to see Vertex AI / Agent Engine
  usage (not just the Gemini Developer API), confirm this satisfies "runs
  as ADK agents on Vertex AI" or switch this call to go through an ADK
  `LlmAgent` as well, for consistency with `web_sentiment_agent.py`.
- Comment sentiment analysis on YouTube comments (positive/negative
  classification) isn't implemented yet — `collect_youtube_data` returns
  raw comment text only. That's Sentiment Synthesis Agent's job per the
  roadmap, not built in this increment.
- No retry/backoff on YouTube quota errors (HTTP 403 quotaExceeded is a
  different failure mode than commentsDisabled, and isn't distinguished
  in the current error handling).
