"""Pata gateway: FastAPI on Render. Model on DigitalOcean, rules as fallback, Sentry tracing.

    uv run uvicorn api.main:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import sentry_sdk
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from pata.client import ModelClient
from pata.csvio import read_inputs, write_results
from pata.schema import Market, TriageInput
from pata.service import triage, triage_batch

MAX_BATCH = 500
STATIC = Path(__file__).parent / "static"

if dsn := os.environ.get("SENTRY_DSN"):
    sentry_sdk.init(
        dsn=dsn,
        traces_sample_rate=1.0,
        send_default_pii=False,
        environment=os.environ.get("SENTRY_ENVIRONMENT", "development"),
        release=os.environ.get("RENDER_GIT_COMMIT"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = ModelClient.from_env()
    yield
    if app.state.client:
        await app.state.client.aclose()


app = FastAPI(title="Pata", version="0.1.0", lifespan=lifespan)


class BatchIn(BaseModel):
    orders: list[TriageInput] = Field(min_length=1, max_length=MAX_BATCH)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
async def healthz() -> dict:
    c = app.state.client
    return {"ok": True, "model": c.model if c else None, "mode": "model" if c else "rules-only"}


@app.post("/triage")
async def triage_one(inp: TriageInput) -> dict:
    return (await triage(inp, app.state.client)).as_dict()


@app.post("/triage/batch")
async def triage_many(body: BatchIn) -> dict:
    results = await triage_batch(body.orders, app.state.client)
    return {"results": [r.as_dict() for r in results]}


@app.post("/triage/csv")
async def triage_csv(
    file: Annotated[UploadFile, File()], market: Annotated[Market, Form()] = Market.AE
) -> Response:
    text = (await file.read()).decode("utf-8", errors="replace")
    try:
        rows, inputs = read_inputs(text, market)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    if not inputs:
        raise HTTPException(422, "CSV has no rows")
    if len(inputs) > MAX_BATCH:
        raise HTTPException(413, f"max {MAX_BATCH} rows per upload")
    results = await triage_batch(inputs, app.state.client)
    return Response(
        write_results(rows, results),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="triaged.csv"'},
    )
