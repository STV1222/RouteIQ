"""
RouteIQ FastAPI application entry point.

Run locally:
    uvicorn src.main:app --reload --port 8080

Via Docker:
    docker-compose up
"""

from __future__ import annotations

import logging
import sys

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.config import settings

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("routeiq")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RouteIQ",
    description=(
        "Multi-Agent LLM Cost Optimizer proxy. "
        "Drop-in replacement for the OpenAI API — saves 50–90% on LLM costs "
        "by routing each request to the cheapest capable model automatically."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — open by default, tighten in production via env
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error. Please try again.",
                "type": "server_error",
                "code": "internal_error",
            }
        },
    )


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup() -> None:
    logger.info("RouteIQ starting up — env=%s", settings.routeiq_env)

    if settings.routeiq_env == "development":
        # Auto-create DynamoDB Local tables for a smooth local dev experience
        try:
            from src.storage.dynamo import create_tables_local
            create_tables_local()
        except Exception as exc:
            logger.warning("Could not create DynamoDB tables (will retry on first request): %s", exc)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("RouteIQ shutting down.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.include_router(router)

# Static files + dashboard shortcut
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

@app.get("/dashboard", include_in_schema=False)
async def dashboard():
    return FileResponse(os.path.join(_STATIC_DIR, "dashboard.html"))
