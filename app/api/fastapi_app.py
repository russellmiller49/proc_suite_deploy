"""FastAPI application wiring for the Procedure Suite services.

⚠️ SOURCE OF TRUTH: This is the MAIN FastAPI application.
- Running on port 8000 via ops/devserver.sh
- Uses CodingService from app/coder/application/coding_service.py (new hexagonal architecture)
- DO NOT edit api/app.py - it's deprecated

See AI_ASSISTANT_GUIDE.md for details.
"""

# ruff: noqa: E402

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


# Prefer explicitly-exported environment variables over values in `.env`.
# Tests can opt out (and avoid accidental real network calls) by setting `PROCSUITE_SKIP_DOTENV=1`.
if not _truthy_env("PROCSUITE_SKIP_DOTENV"):
    try:
        load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Failed to load .env via python-dotenv (%s); proceeding with OS env only",
            type(e).__name__,
        )

from app.api.bootstrap import StartupBootstrap
from app.api.ml_advisor_router import router as ml_advisor_router
from app.api.registry_payload import shape_registry_payload as _shape_registry_payload
from app.api.routes.legacy_coder import router as legacy_coder_router
from app.api.routes.legacy_registry import router as legacy_registry_router
from app.api.routes.metrics import router as metrics_router
from app.api.routes.phi import router as phi_router
from app.api.routes.phi_demo_cases import router as phi_demo_router
from app.api.routes.procedure_codes import router as procedure_codes_router
from app.api.routes.process_bundle import router as process_bundle_router
from app.api.routes.qa import router as qa_router
from app.api.routes.registry_runs import router as registry_runs_router
from app.api.routes.reporting import router as reporting_router
from app.api.routes.unified_process import router as unified_process_router
from app.api.routes_registry import router as registry_extract_router
from app.api.schemas import KnowledgeMeta
from app.common.knowledge import knowledge_hash, knowledge_version


# ============================================================================
# Application Lifespan Context Manager
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan with startup/shutdown bootstrap orchestration."""
    bootstrap = StartupBootstrap(app)
    await bootstrap.startup()
    yield
    await bootstrap.shutdown()


app = FastAPI(
    title="Procedure Suite API",
    version="0.3.0",
    lifespan=lifespan,
)

# CORS (dev-friendly defaults)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev: allow all
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def _phi_redactor_headers(request: Request, call_next):
    """
    Ensure the PHI redactor UI (including /vendor/* model assets) works in
    cross-origin isolated contexts and when embedded/loaded from other origins
    during development.
    """
    resp = await call_next(request)
    path = request.url.path
    # Apply COEP headers to all /ui/ paths (PHI Redactor is now the main UI)
    if path.startswith("/ui"):
        # Required for SharedArrayBuffer in modern browsers (cross-origin isolation).
        resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        resp.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        # Allow these assets to be requested as subresources in COEP contexts.
        resp.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        # Dev convenience: make vendor assets fetchable from any origin.
        # (CORSMiddleware adds CORS headers when an Origin header is present,
        # but some contexts can still surface this as a "CORS error" without it.)
        resp.headers.setdefault("Access-Control-Allow-Origin", "*")
        resp.headers.setdefault("Access-Control-Allow-Methods", "*")
        resp.headers.setdefault("Access-Control-Allow-Headers", "*")
        # Chrome Private Network Access (PNA): when the UI is loaded from a
        # "public" secure context (e.g., an https webview) and it fetches
        # localhost resources, Chrome sends a preflight with
        # Access-Control-Request-Private-Network: true and expects this header.
        if request.headers.get("access-control-request-private-network", "").lower() == "true":
            resp.headers["Access-Control-Allow-Private-Network"] = "true"
        # Avoid stale caching during rapid iteration/debugging.
        resp.headers.setdefault("Cache-Control", "no-store")
    return resp

# Include ML Advisor router
app.include_router(ml_advisor_router, prefix="/api/v1", tags=["ML Advisor"])
# Include PHI router
app.include_router(phi_router)
# Include procedure codes router
app.include_router(procedure_codes_router, prefix="/api/v1", tags=["procedure-codes"])
# Metrics router
app.include_router(metrics_router, tags=["metrics"])
# PHI demo cases router (non-PHI metadata)
app.include_router(phi_demo_router)
# Registry extraction router (hybrid-first pipeline)
app.include_router(registry_extract_router, tags=["registry"])
# Registry run persistence router (Diamond Loop)
app.include_router(registry_runs_router, prefix="/api", tags=["registry-runs"])
# Unified process router (UI entry point)
app.include_router(unified_process_router, prefix="/api")
# Bundle process router (multi-doc ZK ingestion)
app.include_router(process_bundle_router, prefix="/api")
# Legacy/API support routers split from this composition root.
app.include_router(legacy_coder_router)
app.include_router(legacy_registry_router)
app.include_router(reporting_router)
app.include_router(qa_router)

def _phi_redactor_response(path: Path) -> FileResponse:
    resp = FileResponse(path)
    # Required for SharedArrayBuffer in modern browsers (cross-origin isolation).
    resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    resp.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    # Avoid stale client-side caching during rapid iteration/debugging.
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _phi_redactor_static_dir() -> Path:
    import ui

    return Path(ui.__file__).resolve().parent / "static" / "phi_redactor"


def _static_files_enabled() -> bool:
    return os.getenv("DISABLE_STATIC_FILES", "").lower() not in ("true", "1", "yes")


@app.get("/ui/phi_redactor")
def phi_redactor_redirect() -> RedirectResponse:
    # Avoid "/ui/phi_redactor" being treated as a file path in the browser (breaks relative URLs).
    # Redirect ensures relative module imports resolve to "/ui/phi_redactor/...".
    resp = RedirectResponse(url="/ui/phi_redactor/")
    resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    resp.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/ui/phi_redactor/")
def phi_redactor_index() -> FileResponse:
    if not _static_files_enabled():
        raise HTTPException(status_code=404, detail="Static files disabled")
    index_path = _phi_redactor_static_dir() / "index.html"
    return _phi_redactor_response(index_path)


@app.get("/ui/phi_redactor/index.html")
def phi_redactor_index_html() -> FileResponse:
    if not _static_files_enabled():
        raise HTTPException(status_code=404, detail="Static files disabled")
    index_path = _phi_redactor_static_dir() / "index.html"
    return _phi_redactor_response(index_path)


@app.get("/ui/phi_redactor/app.js")
def phi_redactor_app_js() -> FileResponse:
    if not _static_files_enabled():
        raise HTTPException(status_code=404, detail="Static files disabled")
    return _phi_redactor_response(_phi_redactor_static_dir() / "app.js")


@app.get("/ui/phi_redactor/redactor.worker.js")
def phi_redactor_worker_js() -> FileResponse:
    if not _static_files_enabled():
        raise HTTPException(status_code=404, detail="Static files disabled")
    return _phi_redactor_response(_phi_redactor_static_dir() / "redactor.worker.js")

@app.get("/ui/redactor.worker.legacy.js")
def phi_redactor_worker_legacy_js() -> FileResponse:
    if not _static_files_enabled():
        raise HTTPException(status_code=404, detail="Static files disabled")
    return _phi_redactor_response(_phi_redactor_static_dir() / "redactor.worker.legacy.js")

@app.get("/ui/protectedVeto.legacy.js")
def phi_redactor_protected_veto_legacy_js() -> FileResponse:
    if not _static_files_enabled():
        raise HTTPException(status_code=404, detail="Static files disabled")
    return _phi_redactor_response(_phi_redactor_static_dir() / "protectedVeto.legacy.js")

@app.get("/ui/transformers.min.js")
def phi_redactor_transformers_min_js() -> FileResponse:
    if not _static_files_enabled():
        raise HTTPException(status_code=404, detail="Static files disabled")
    return _phi_redactor_response(_phi_redactor_static_dir() / "transformers.min.js")


@app.get("/ui/phi_redactor/styles.css")
def phi_redactor_styles_css() -> FileResponse:
    if not _static_files_enabled():
        raise HTTPException(status_code=404, detail="Static files disabled")
    return _phi_redactor_response(_phi_redactor_static_dir() / "styles.css")


@app.get("/ui/phi_redactor/protectedVeto.js")
def phi_redactor_protected_veto_js() -> FileResponse:
    if not _static_files_enabled():
        raise HTTPException(status_code=404, detail="Static files disabled")
    return _phi_redactor_response(_phi_redactor_static_dir() / "protectedVeto.js")


@app.get("/ui/phi_redactor/allowlist_trie.json")
def phi_redactor_allowlist() -> FileResponse:
    if not _static_files_enabled():
        raise HTTPException(status_code=404, detail="Static files disabled")
    return _phi_redactor_response(_phi_redactor_static_dir() / "allowlist_trie.json")


@app.get("/ui/phi_redactor/vendor/{asset_path:path}")
def phi_redactor_vendor_asset(asset_path: str) -> FileResponse:
    if not _static_files_enabled():
        raise HTTPException(status_code=404, detail="Static files disabled")
    vendor_dir = _phi_redactor_static_dir() / "vendor"
    asset = (vendor_dir / asset_path).resolve()
    if vendor_dir not in asset.parents or not asset.exists() or not asset.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return _phi_redactor_response(asset)


@app.get("/ui/phi_redactor/sw.js")
def phi_redactor_sw() -> FileResponse:
    if not _static_files_enabled():
        raise HTTPException(status_code=404, detail="Static files disabled")
    return _phi_redactor_response(_phi_redactor_static_dir() / "sw.js")


@app.get("/ui/phi_identifiers", include_in_schema=False)
def phi_identifiers_doc() -> FileResponse:
    """Serve the PHI identifiers reference used by the UI confirmation modal."""

    doc_path = Path(__file__).resolve().parents[2] / "docs" / "PHI_IDENTIFIERS.md"
    if not doc_path.exists() or not doc_path.is_file():
        raise HTTPException(status_code=404, detail="PHI identifiers doc not found")

    resp = FileResponse(doc_path, media_type="text/markdown")
    # Required for SharedArrayBuffer in modern browsers (cross-origin isolation).
    resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    resp.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    resp.headers["Cache-Control"] = "no-store"
    return resp

# Skip static file mounting when DISABLE_STATIC_FILES is set (useful for testing)
if os.getenv("DISABLE_STATIC_FILES", "").lower() not in ("true", "1", "yes"):
    # Mount PHI Redactor as the main UI (client-side PHI detection).
    phi_redactor_dir = _phi_redactor_static_dir()
    app.mount("/ui", StaticFiles(directory=str(phi_redactor_dir), html=True), name="ui")
    # Also mount vendor directory for ONNX model files
    vendor_dir = phi_redactor_dir / "vendor"
    if vendor_dir.exists():
        app.mount("/ui/vendor", StaticFiles(directory=str(vendor_dir)), name="ui_vendor")
        app.mount(
            "/ui/phi_redactor/vendor",
            StaticFiles(directory=str(vendor_dir)),
            name="ui_phi_redactor_vendor",
        )

# Configure logging
_logger = logging.getLogger(__name__)


# ============================================================================
# Heavy NLP model preloading (delegated to app.infra.nlp_warmup)
# ============================================================================
from app.infra.nlp_warmup import (
    is_nlp_warmed,
)

# NOTE: The lifespan context manager is defined above app creation.
# See lifespan() function for startup/shutdown logic.


class LocalityInfo(BaseModel):
    code: str
    name: str


@lru_cache(maxsize=1)
def _load_gpci_data() -> dict[str, str]:
    """Load GPCI locality data from CSV file.

    Returns a dict mapping locality codes to locality names.
    """
    import csv
    from pathlib import Path

    gpci_file = Path("data/RVU_files/gpci_2025.csv")
    if not gpci_file.exists():
        gpci_file = Path("proc_autocode/rvu/data/gpci_2025.csv")

    localities: dict[str, str] = {}
    if gpci_file.exists():
        try:
            with gpci_file.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    code = row.get("mac_locality", row.get("locality_code", ""))
                    name = row.get("locality_name", "")
                    if code and name:
                        localities[code] = name
        except Exception as e:
            _logger.warning(f"Failed to load GPCI data: {e}")

    # Add default national locality if not present
    if "00" not in localities:
        localities["00"] = "National (Default)"

    return localities


@app.get("/")
async def root(request: Request) -> Any:
    """Root endpoint with API information or redirect to UI."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return RedirectResponse(url="/ui/")

    return {
        "name": "Procedure Suite API",
        "version": "0.3.0",
        "endpoints": {
            "ui": "/ui/",
            "health": "/health",
            "ready": "/ready",
            "knowledge": "/knowledge",
            "docs": "/docs",
            "redoc": "/redoc",
            "unified_process": "/api/v1/process",  # NEW: Combined registry + coder
            "coder": "/v1/coder/run",
            "localities": "/v1/coder/localities",
            "registry": "/v1/registry/run",
            "report_verify": "/report/verify",
            "report_questions": "/report/questions",
            "report_seed_from_text": "/report/seed_from_text",
            "report_render": "/report/render",
            "qa_run": "/qa/run",
            "ml_advisor": {
                "health": "/api/v1/ml-advisor/health",
                "status": "/api/v1/ml-advisor/status",
                "code": "/api/v1/ml-advisor/code",
                "code_with_advisor": "/api/v1/ml-advisor/code_with_advisor",
                "suggest": "/api/v1/ml-advisor/suggest",
                "traces": "/api/v1/ml-advisor/traces",
                "metrics": "/api/v1/ml-advisor/metrics",
            },
            "registry_extract": "/api/registry/extract",
        },
        "note": (
            "Use /api/v1/process for extraction-first pipeline (registry → CPT codes in one call). "
            "Legacy endpoints /v1/coder/run and /v1/registry/run still available."
        ),
    }


@app.get("/health")
async def health(request: Request) -> dict[str, bool]:
    # Liveness probe: keep payload stable and minimal.
    # Readiness is exposed via `/ready`.
    return {"ok": True}


@app.get("/ready")
async def ready(request: Request) -> JSONResponse:
    is_ready = bool(getattr(request.app.state, "model_ready", False))
    if is_ready:
        return JSONResponse(status_code=200, content={"status": "ok", "ready": True})

    model_error = getattr(request.app.state, "model_error", None)
    content: dict[str, Any] = {"status": "warming", "ready": False}
    if model_error:
        content["status"] = "error"
        content["error"] = str(model_error)
        return JSONResponse(status_code=503, content=content)

    return JSONResponse(status_code=503, content=content, headers={"Retry-After": "10"})


@app.get("/health/nlp")
async def nlp_health() -> JSONResponse:
    """Check NLP model readiness.

    Returns 200 OK if NLP models are loaded and ready.
    Returns 503 Service Unavailable if NLP features are degraded.

    This endpoint can be used by load balancers to route requests
    to instances with fully warmed NLP models.
    """
    if is_nlp_warmed():
        return JSONResponse(
            status_code=200,
            content={"status": "ok", "nlp_ready": True},
        )
    return JSONResponse(
        status_code=503,
        content={"status": "degraded", "nlp_ready": False},
    )


@app.get("/knowledge", response_model=KnowledgeMeta)
async def knowledge() -> KnowledgeMeta:
    return KnowledgeMeta(version=knowledge_version() or "unknown", sha256=knowledge_hash() or "")


@app.get("/v1/coder/localities", response_model=List[LocalityInfo])
async def coder_localities() -> List[LocalityInfo]:
    """List available geographic localities for RVU calculation."""
    gpci_data = _load_gpci_data()
    localities = [
        LocalityInfo(code=code, name=name)
        for code, name in gpci_data.items()
    ]
    localities.sort(key=lambda x: x.name)
    return localities


__all__ = ["app", "_shape_registry_payload"]
