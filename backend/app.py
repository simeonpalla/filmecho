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

import json
import os
import re

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.pdf_report import build_pdf
from orchestration.pipeline import run_pipeline, stream_pipeline

app = FastAPI(title="Filmecho")

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/api/brief")
async def get_brief(title: str, region_hint: str = ""):
    """Run the full pipeline for a title and return the combined result.

    Non-streaming; kept for programmatic/API callers that just want one
    JSON response. The frontend uses /api/brief/stream instead, so it can
    show real progress.
    """
    title = (title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    try:
        return await run_pipeline(title, region_hint=region_hint)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc


@app.get("/api/brief/stream")
async def stream_brief(title: str, region_hint: str = ""):
    """Stream pipeline progress as Server-Sent Events.

    Each event is a JSON-encoded line in the SSE `data:` field, matching
    the dicts stream_pipeline() yields: {"event": "stage", ...},
    {"event": "error", ...}, or the final {"event": "result", ...}.

    region_hint is an optional locale/timezone string the frontend
    derives client-side (Intl.DateTimeFormat + navigator.language) and
    passes through untouched — this backend does no IP geolocation or
    server-side location inference of its own.

    Validation (missing title) happens before the stream opens, so it
    still returns a normal HTTP error rather than an SSE error event.
    """
    title = (title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")

    async def event_source():
        try:
            async for event in stream_pipeline(title, region_hint=region_hint):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # noqa: BLE001
            # Last-resort guard: stream_pipeline degrades individual
            # branches internally, but if something outside that (e.g. a
            # bug in event serialization) still throws, tell the client
            # rather than silently closing the connection.
            yield f"data: {json.dumps({'event': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Some reverse proxies buffer responses by default, which would
            # defeat the purpose of streaming; this is the standard opt-out
            # header for nginx-based proxies. Harmless if the proxy in front
            # of your deployment doesn't buffer (Cloud Run's default path
            # generally doesn't), but verify progress actually streams
            # incrementally once deployed, not just locally.
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/brief/pdf")
async def brief_pdf(data: dict = Body(...)):
    """Generate a formatted A4 PDF from an already-computed brief payload.

    Takes the SAME JSON the frontend already holds after a pipeline run
    (the {"event": "result", "data": ...} payload) and renders it to a
    real, laid-out document via backend/pdf_report.py — this does NOT
    re-run the pipeline, so it costs zero additional Parallel/Gemini
    calls, it's purely a formatting step over data already fetched.
    """
    try:
        pdf_bytes = build_pdf(data)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}") from exc

    title = ((data.get("entity") or {}).get("title")) or "brief"
    safe_title = re.sub(r"[^A-Za-z0-9_-]+", "_", title).strip("_") or "brief"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="filmecho-{safe_title}.pdf"'},
    )


# Serve the frontend. Mounted last so it doesn't shadow /api/*.
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
