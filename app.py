"""BERT playground — FastAPI app (thin wrapper around playground.py).

Serves the static UI and the JSON API described in API_CONTRACT.md:

    GET  /                the UI (static/index.html)
    GET  /static/...      assets
    GET  /api/meta        model + probe registry
    POST /api/tokenize    tokenizer pipeline breakdown
    POST /api/forward     all hidden states + attention for one text
    POST /api/layer       one hidden state at full 768-dim resolution
    POST /api/similarity  sentence-embedding cosines + PCA
    POST /api/probe       fine-tuned classifier head

Run it (from this directory):

    .venv/bin/python app.py                  # http://127.0.0.1:8303
    .venv/bin/uvicorn app:app --port 8303    # equivalent

Same origin as the UI — no CORS configured on purpose.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import playground

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("bert-playground")

BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Startup: load the two encoders eagerly so the first classroom click is fast.
# Probe classifiers stay lazy; /api/meta reports their availability using
# playground's cheap checks (directory / cached config — no weights loaded).
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    for spec in playground.MODELS:
        playground.get_model(spec["id"])
    emotion_found = playground.EMOTION_DIR.is_dir()
    logger.info("emotion model dir: %s (%s)",
                playground.EMOTION_DIR, "found" if emotion_found else "MISSING")
    for probe_id in playground.PROBES:
        logger.info("probe %r available=%s", probe_id,
                    playground.probe_available(probe_id))
    logger.info("ready — UI at http://127.0.0.1:8303/")
    yield


app = FastAPI(title="BERT playground", lifespan=lifespan)


# --------------------------------------------------------------------------
# Request bodies — field names per API_CONTRACT.md
# --------------------------------------------------------------------------

class TokenizeBody(BaseModel):
    model: str
    text: str


class ForwardBody(BaseModel):
    model: str
    text: str


class LayerBody(BaseModel):
    model: str
    text: str
    layer: int


class SimilarityBody(BaseModel):
    model: str
    sentences: list[str]


class ProbeBody(BaseModel):
    text: str
    probe: str


def _run(fn):
    """Execute a payload builder; ValueError means bad request -> 400.

    The message becomes {"detail": "<message>"}, the contract's error shape.
    """
    try:
        return fn()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# --------------------------------------------------------------------------
# API routes
# --------------------------------------------------------------------------

@app.get("/api/meta")
def api_meta():
    return playground.meta_payload()


@app.post("/api/tokenize")
def api_tokenize(body: TokenizeBody):
    return _run(lambda: playground.tokenize_payload(body.model, body.text))


@app.post("/api/forward")
def api_forward(body: ForwardBody):
    return _run(lambda: playground.forward_payload(body.model, body.text))


@app.post("/api/layer")
def api_layer(body: LayerBody):
    return _run(lambda: playground.layer_payload(body.model, body.text, body.layer))


@app.post("/api/similarity")
def api_similarity(body: SimilarityBody):
    return _run(lambda: playground.similarity_payload(body.model, body.sentences))


@app.post("/api/probe")
def api_probe(body: ProbeBody):
    return _run(lambda: playground.probe_payload(body.probe, body.text))


# --------------------------------------------------------------------------
# Static UI (same origin — no CORS)
# --------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index():
    index_html = BASE_DIR / "static" / "index.html"
    if not index_html.is_file():
        raise HTTPException(status_code=404,
                            detail="static/index.html not built yet")
    return FileResponse(index_html)


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8303)
