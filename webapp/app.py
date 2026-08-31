from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agent.main_agent import Agent
from config import constants
from config.env_config import EnvConfig
from llm.factory import create_llm_client
from utils.data_verify import verify_file
from webapp.catalog import CatalogError, CatalogPresenter
from webapp.metrics import UsageRecorder
from webapp.runtime import RuntimeManager
from webapp.schemas import ChatResponse, MessageRequest, SessionResponse
from webapp.service import InvalidMessage, SessionManager, SessionNotFound

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


@dataclass(frozen=True)
class WebRuntime:
    sessions: SessionManager
    catalog: CatalogPresenter


DEFAULT_CFG = {
    "llm_provider": "none",
    "llm_model": "",
    "api_key": "",
    "rerank_backend": "none",
    "retrieval_backend": "auto",
    "output_strategy": "holdback",
    "llm_intent_enabled": False,
    "fingerprint": True,
    "category_expand": True,
    "paraphrase": True,
}


@dataclass
class RuntimeContainer:
    status: Literal["loading", "ready", "failed"] = "loading"
    runtime: WebRuntime | None = None
    manager: RuntimeManager | None = None
    error_code: str | None = None


Initializer = Callable[[Path], WebRuntime]


async def wait_for_initializer(task: asyncio.Task[None]) -> None:
    """Wait for a thread-backed initializer without abandoning it on cancellation."""
    cancelled = False
    while True:
        try:
            await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError


async def wait_for_runtime_switch(
    task: asyncio.Task[tuple[object, str]],
) -> tuple[tuple[object, str], bool]:
    """Finish a thread-backed switch so manager and published runtime stay consistent."""
    cancelled = False
    while True:
        try:
            return await asyncio.shield(task), cancelled
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _unconfigured_initializer(catalog_path: Path) -> WebRuntime:
    raise RuntimeError(f"No runtime initializer configured for {catalog_path}")


def initialize_runtime(
    catalog_path: Path,
    *,
    env_loader: Callable[[], EnvConfig] = EnvConfig.from_env,
    verifier: Callable[[Path, str, str, bool], bool] = verify_file,
    agent_factory: Callable[[Path, EnvConfig, object], Agent] = Agent,
    usage_recorder: UsageRecorder | None = None,
) -> WebRuntime:
    """Build the production web runtime for one selected catalog file."""
    env = env_loader()
    if not env.skip_data_verify:
        verifier(catalog_path, constants.EXPECTED_SHA256_CATALOG, "catalog.jsonl", skip=False)
    catalog = CatalogPresenter.build(catalog_path)
    web_env = EnvConfig(replace(env.app_config, skip_data_verify=True))
    llm_client = create_llm_client(web_env.llm)
    llm_client.initialize()
    agent = agent_factory(catalog_path, web_env, llm_client)
    recorder = usage_recorder if usage_recorder is not None else UsageRecorder.from_env()
    context = {
        "provider": str(web_env.llm.provider or "none"),
        "model": str(web_env.llm.model or ""),
        "retrieval_backend": str(web_env.app_config.retrieval_backend or "auto"),
        "rerank_backend": (
            str(web_env.llm.rerank_backend or "auto") if web_env.llm.rerank_enabled else "none"
        ),
        "output_strategy": "holdback" if web_env.app_config.emit_gate else "full",
    }
    return WebRuntime(
        SessionManager(
            agent,
            catalog,
            top_k=web_env.top_k,
            usage_recorder=recorder,
            usage_context=context,
        ),
        catalog,
    )


def create_app(
    catalog_path: Path = Path("data/catalog.jsonl"),
    initializer: Initializer | None = None,
    runtime: WebRuntime | None = None,
) -> FastAPI:
    container = RuntimeContainer()
    runtime_switch_lock = asyncio.Lock()
    active_initializer = initializer or initialize_runtime

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if runtime is not None:
            container.runtime = runtime
            container.status = "ready"
            yield
            return

        async def initialize() -> None:
            try:
                if initializer is not None:
                    # tests / custom bootstrap: honor the provided initializer
                    # (failure/loading simulation)
                    container.runtime = await asyncio.to_thread(active_initializer, catalog_path)
                else:
                    manager = await asyncio.to_thread(RuntimeManager.create, catalog_path)
                    built, _key = await asyncio.to_thread(manager.switch, dict(DEFAULT_CFG))
                    container.manager = manager
                    container.runtime = built
            except Exception:
                logger.exception("web runtime initialization failed")
                container.error_code = "initialization_failed"
                container.status = "failed"
            else:
                container.status = "ready"

        task = asyncio.create_task(initialize())
        try:
            yield
        finally:
            await wait_for_initializer(task)

    app = FastAPI(lifespan=lifespan)
    app.state.runtime_container = container
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
    dashboard_dist = Path(__file__).parent.parent / "dashboard" / "dist"
    if dashboard_dist.exists():
        app.mount("/dashboard", StaticFiles(directory=dashboard_dist, html=True), name="dashboard")

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable):
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unexpected unhandled web error")
            response = error_response(500, "internal_error", "An internal error occurred.")
        # Allow the in-app /dashboard iframe to be embedded by this same origin
        # (frame-ancestors 'self'); everything else stays clickjack-proof.
        frame_ancestors = "'self'" if request.url.path.startswith("/dashboard") else "'none'"
        response.headers["Content-Security-Policy"] = (
            f"default-src 'self'; img-src 'none'; object-src 'none'; "
            f"frame-ancestors {frame_ancestors}"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        message_error = any(error.get("loc", ())[-1:] == ("message",) for error in exc.errors())
        if message_error:
            return error_response(
                400, "invalid_message", "Message must contain 1 to 4000 characters."
            )
        return error_response(400, "invalid_request", "Request body is invalid.")

    def require_runtime() -> WebRuntime | JSONResponse:
        if container.status == "ready" and container.runtime is not None:
            return container.runtime
        if container.status == "failed":
            return error_response(503, "service_failed", "The service is unavailable.")
        return error_response(503, "service_initializing", "The service is initializing.")

    def session_not_found_response() -> JSONResponse:
        return error_response(404, "session_not_found", "Session not found.")

    def catalog_error_response() -> JSONResponse:
        return error_response(503, "catalog_unavailable", "The catalog is unavailable.")

    def internal_error_response() -> JSONResponse:
        return error_response(500, "internal_error", "An internal error occurred.")

    @app.get("/", include_in_schema=False, response_class=FileResponse)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    def require_manager() -> RuntimeManager | JSONResponse:
        if container.manager is not None:
            return container.manager
        return error_response(503, "service_failed", "Runtime manager unavailable.")

    @app.get("/api/runtime", response_model=None)
    async def get_runtime() -> dict[str, object] | JSONResponse:
        manager = require_manager()
        if isinstance(manager, JSONResponse):
            return manager
        try:
            return manager.runtime_info()
        except Exception:
            logger.exception("runtime info failed")
            return internal_error_response()

    @app.post("/api/runtime/config", response_model=None)
    async def set_runtime_config(payload: dict[str, object]) -> dict[str, object] | JSONResponse:
        manager = require_manager()
        if isinstance(manager, JSONResponse):
            return manager
        cfg = {
            k: v
            for k, v in payload.items()
            if k
            in {
                "llm_provider",
                "llm_model",
                "api_key",
                "rerank_backend",
                "retrieval_backend",
                "output_strategy",
                "llm_intent_enabled",
                "fingerprint",
                "category_expand",
                "paraphrase",
            }
        }
        cfg.setdefault("llm_provider", "none")
        cfg.setdefault("rerank_backend", "none")
        cfg.setdefault("retrieval_backend", "auto")
        cfg.setdefault("output_strategy", "holdback")
        cfg.setdefault("fingerprint", True)
        cfg.setdefault("category_expand", True)
        cfg.setdefault("paraphrase", True)
        cfg.setdefault("llm_intent_enabled", False)
        async with runtime_switch_lock:
            cancelled = False
            try:
                worker = asyncio.create_task(asyncio.to_thread(manager.switch, cfg))
                (built, _key), cancelled = await wait_for_runtime_switch(worker)
                container.runtime = built
            except Exception:
                logger.exception("runtime config switch failed")
                return error_response(
                    500, "config_failed", "Could not build runtime for the selected config."
                )
            info = manager.runtime_info()
            info["sessions_reset"] = True
            if cancelled:
                raise asyncio.CancelledError
            return info

    @app.get("/api/metrics", response_model=None)
    async def get_metrics() -> dict[str, object] | JSONResponse:
        manager = require_manager()
        if isinstance(manager, JSONResponse):
            return manager
        try:
            recorder = getattr(manager, "recorder", None)
            if recorder is None:
                return {
                    "summary": {
                        "total_turns": 0,
                        "online_turns": 0,
                        "offline_turns": 0,
                        "total_prompt_tokens": 0,
                        "total_completion_tokens": 0,
                        "total_tokens": 0,
                        "total_cost_usd": 0.0,
                        "per_provider": [],
                    },
                    "recent": [],
                }
            return {"summary": recorder.summary(), "recent": recorder.recent(limit=50)}
        except Exception:
            logger.exception("metrics failed")
            return internal_error_response()

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        if container.status == "failed":
            return {
                "status": "failed",
                "error": {
                    "code": container.error_code or "initialization_failed",
                    "message": "The service could not be initialized.",
                },
            }
        return {"status": container.status}

    @app.post("/api/sessions", response_model=SessionResponse)
    async def create_session() -> SessionResponse | JSONResponse:
        active_runtime = require_runtime()
        if isinstance(active_runtime, JSONResponse):
            return active_runtime
        try:
            snapshot = await active_runtime.sessions.create_session()
        except CatalogError:
            logger.exception("catalog error while creating session")
            return catalog_error_response()
        except Exception:
            logger.exception("unexpected error while creating session")
            return internal_error_response()
        return SessionResponse(session_id=snapshot.session_id, next_turn=snapshot.next_turn)

    @app.get("/api/sessions/{session_id}", response_model=SessionResponse)
    async def get_session(session_id: UUID) -> SessionResponse | JSONResponse:
        active_runtime = require_runtime()
        if isinstance(active_runtime, JSONResponse):
            return active_runtime
        try:
            snapshot = active_runtime.sessions.get_session(session_id)
        except CatalogError:
            logger.exception("catalog error while looking up session")
            return catalog_error_response()
        except Exception:
            logger.exception("unexpected error while looking up session")
            return internal_error_response()
        if snapshot is None:
            return session_not_found_response()
        return SessionResponse(session_id=snapshot.session_id, next_turn=snapshot.next_turn)

    @app.post("/api/sessions/{session_id}/messages", response_model=ChatResponse)
    async def send_message(
        session_id: UUID, message: MessageRequest
    ) -> ChatResponse | JSONResponse:
        active_runtime = require_runtime()
        if isinstance(active_runtime, JSONResponse):
            return active_runtime
        try:
            envelope = await active_runtime.sessions.send_message(
                session_id, message.message_id, message.message
            )
            return ChatResponse.model_validate(envelope)
        except SessionNotFound:
            return session_not_found_response()
        except InvalidMessage:
            return error_response(
                400, "invalid_message", "Message must contain 1 to 4000 characters."
            )
        except CatalogError:
            logger.exception("catalog error while sending message")
            return catalog_error_response()
        except Exception:
            logger.exception("unexpected adapter error while sending message")
            return internal_error_response()

    @app.get("/api/products/{parent_asin}", response_model=None)
    async def get_product(parent_asin: str) -> dict[str, object] | JSONResponse:
        active_runtime = require_runtime()
        if isinstance(active_runtime, JSONResponse):
            return active_runtime
        try:
            product = await asyncio.to_thread(active_runtime.catalog.detail, parent_asin)
        except CatalogError:
            logger.exception("catalog error while reading product")
            return catalog_error_response()
        except Exception:
            logger.exception("unexpected adapter error while reading product")
            return internal_error_response()
        if product is None:
            return error_response(404, "product_not_found", "Product not found.")
        return product

    return app
