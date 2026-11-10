"""Community Cloud entry point: streamlit run app.py (Streamlit 1.67)."""

# Apply numerical CPU limits before importing Streamlit in this module.
from engine import PredictRequest, PredictResponse, get_labeler

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import streamlit as st
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route
from threadpoolctl import threadpool_limits


@asynccontextmanager
async def lifespan(_app):
    # CLI may have imported native libraries before the entry point was discovered.
    with threadpool_limits(limits=1):
        labeler = get_labeler()
        try:
            yield {"labeler": labeler, "inference_slot": asyncio.Semaphore(1)}
        finally:
            labeler._cached_probabilities.cache_clear()
            get_labeler.cache_clear()


async def predict(request):
    try:
        payload = PredictRequest.model_validate_json(await request.body())
    except ValidationError as error:
        details = json.loads(error.json(include_url=False, include_context=False, include_input=False))
        for item in details:
            item["loc"] = ["body", *item["loc"]]
        return JSONResponse({"detail": details}, status_code=422)
    # st.App exposes lifespan state through app.state, not request.state.
    async with app.state["inference_slot"]:
        result = await run_in_threadpool(
            app.state["labeler"].predict, payload.text, payload.threshold
        )
    return JSONResponse(result.model_dump())


async def health(_request):
    return JSONResponse({"status": "ok"})


OPENAPI = {
    "openapi": "3.1.0",
    "info": {"title": "IndoBERT Labeler API", "version": "1.0.0"},
    "paths": {
        "/predict": {"post": {
            "summary": "Predict report labels",
            "requestBody": {"required": True, "content": {"application/json": {
                "schema": {"$ref": "#/components/schemas/PredictRequest"},
            }}},
            "responses": {
                "200": {"description": "Prediction", "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/PredictResponse"},
                }}},
                "422": {"description": "Invalid text, threshold, or JSON body"},
            },
        }},
        "/health": {"get": {"responses": {"200": {"description": "Service ready"}}}},
    },
    "components": {"schemas": {
        "PredictRequest": PredictRequest.model_json_schema(),
        "PredictResponse": PredictResponse.model_json_schema(),
    }},
}


async def openapi(_request):
    return JSONResponse(OPENAPI)


async def docs(_request):
    return HTMLResponse("""<!doctype html><html lang="id"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>IndoBERT API</title>
<body style="max-width:760px;margin:48px auto;padding:0 20px;font:16px/1.6 system-ui">
<h1>IndoBERT Labeler API</h1><p><b>POST /predict</b> · Content-Type: application/json</p>
<pre>{"text":"jalan berlubang dan banyak sampah","threshold":0.5}</pre>
<p>Hasil berisi <code>labels</code>, <code>probabilities</code>, dan <code>threshold</code>.
Threshold opsional, bawaan 0.5. Teks harus berisi 1–20.000 karakter setelah trim;
threshold harus lebih dari 0 hingga 1. Input tidak valid: HTTP 422.</p>
<p>Di Community Cloud, gunakan prefix <code>/~/+</code> pada URL publik,
misalnya <code>https://NAMA-APP.streamlit.app/~/+/predict</code>.</p>
<p><a href="openapi.json">OpenAPI JSON</a> · <a href="health">Health</a> ·
<a href="/">Coba model di Streamlit</a></p></body></html>""")


# Public Streamlit API: HTTP routes use the same port as the dashboard.
# Community Cloud should launch this file, rather than dashboard.py directly.
app = st.App(
    Path(__file__).with_name("dashboard.py"),
    lifespan=lifespan,
    routes=[
        Route("/predict", predict, methods=["POST"]),
        Route("/health", health),
        Route("/docs", docs),
        Route("/openapi.json", openapi),
    ],
)
