"""Filmecho backend — thin FastAPI wrapper around orchestration.pipeline.

This is intentionally minimal per the roadmap's guidance: "Design is one
of four equally-weighted judging criteria, not the only one." The job of
this file is to expose run_pipeline() over HTTP and serve the static
frontend, nothing more.

Run locally:
    uvicorn backend.app:app --reload --port 8080

Deployed on Cloud Run, this same app is the container's entrypoint.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from orchestration.pipeline import run_pipeline

app = FastAPI(title="Filmecho")

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/api/brief")
async def get_brief(title: str):
    """Run the full pipeline for a title and return the combined result.

    Args:
        title: Bare film/show title, e.g. "Dune Part Three".

    Returns:
        The same dict orchestration.pipeline.run_pipeline returns:
        entity, web_sentiment, youtube, competitive, news,
        sentiment_synthesis, and result (studio_brief/fan_pulse/
        sources_used).
    """
    title = (title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    try:
        return await run_pipeline(title)
    except Exception as exc:  # noqa: BLE001
        # Surface a clean 500 rather than letting a bare stack trace reach
        # the frontend, this endpoint is a demo surface, not an internal
        # tool, so the error shape matters for the judges' first look too.
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc


# Serve the frontend. Mounted last so it doesn't shadow /api/*.
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
