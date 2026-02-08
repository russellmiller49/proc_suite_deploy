"""API startup/shutdown bootstrap orchestration."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import httpx
from fastapi import FastAPI

from config.startup_settings import validate_startup_env


class StartupBootstrap:
    """Encapsulate app lifespan startup and shutdown side effects."""

    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.logger = logging.getLogger(__name__)

    async def startup(self) -> None:
        from app.infra.nlp_warmup import (
            should_skip_warmup as _should_skip_warmup,
        )
        from app.infra.nlp_warmup import (
            warm_heavy_resources_sync as _warm_heavy_resources_sync,
        )
        from app.infra.settings import get_infra_settings
        from app.registry.model_runtime import verify_registry_runtime_bundle

        validate_startup_env()

        settings = get_infra_settings()

        self.app.state.model_ready = False
        self.app.state.model_error = None
        self.app.state.ready_event = asyncio.Event()
        self.app.state.cpu_executor = ThreadPoolExecutor(max_workers=settings.cpu_workers)
        self.app.state.llm_sem = asyncio.Semaphore(settings.llm_concurrency)
        self.app.state.llm_http = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=float(settings.llm_timeout_s),
                write=30.0,
                pool=30.0,
            )
        )

        try:
            from app.api.phi_dependencies import engine as phi_engine
            from app.phi import models as _phi_models  # noqa: F401
            from app.phi.db import Base as PHIBase

            PHIBase.metadata.create_all(bind=phi_engine)
            self.logger.info("PHI database tables verified/created")
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Could not initialize PHI tables: %s", exc)

        try:
            runtime_warnings = verify_registry_runtime_bundle()
            for warning in runtime_warnings:
                self.logger.warning("Registry runtime bundle warning: %s", warning)
        except RuntimeError as exc:
            raise RuntimeError(f"Registry runtime bundle validation failed: {exc}") from exc

        loop = asyncio.get_running_loop()

        def _warmup_worker() -> None:
            try:
                _warm_heavy_resources_sync()
            except Exception as exc:  # noqa: BLE001
                ok = False
                error = f"{type(exc).__name__}: {exc}"
                self.logger.error("Warmup failed: %s", error, exc_info=True)
            else:
                ok = True
                error = None
            self.app.state.model_ready = ok
            self.app.state.model_error = error
            loop.call_soon_threadsafe(self.app.state.ready_event.set)

        def _bootstrap_registry_models() -> None:
            try:
                from app.registry.model_bootstrap import ensure_registry_model_bundle

                ensure_registry_model_bundle()
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Registry model bundle bootstrap skipped/failed: %s", exc)

        if settings.skip_warmup or _should_skip_warmup():
            self.logger.info("Skipping heavy NLP warmup (disabled via environment)")
            self.app.state.model_ready = True
            self.app.state.ready_event.set()
        elif settings.background_warmup:
            self.logger.info("Starting background warmup")
            loop.run_in_executor(self.app.state.cpu_executor, _warmup_worker)
        else:
            self.logger.info("Running warmup before serving traffic")
            try:
                await loop.run_in_executor(self.app.state.cpu_executor, _warm_heavy_resources_sync)
            except Exception as exc:  # noqa: BLE001
                ok = False
                error = f"{type(exc).__name__}: {exc}"
                self.logger.error("Warmup failed: %s", error, exc_info=True)
            else:
                ok = True
                error = None
            self.app.state.model_ready = ok
            self.app.state.model_error = error
            self.app.state.ready_event.set()

        loop.run_in_executor(self.app.state.cpu_executor, _bootstrap_registry_models)

    async def shutdown(self) -> None:
        llm_http = getattr(self.app.state, "llm_http", None)
        if llm_http is not None:
            await llm_http.aclose()

        cpu_executor = getattr(self.app.state, "cpu_executor", None)
        if cpu_executor is not None:
            cpu_executor.shutdown(wait=False, cancel_futures=True)


__all__ = ["StartupBootstrap"]
