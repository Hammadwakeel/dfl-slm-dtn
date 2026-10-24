"""Digital Twin Network FastAPI application factory."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from alembic import command
from alembic.config import Config
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.engine import Engine

from app.core.config import Settings
from app.core.security import require_api_token
from app.db.session import create_db_engine, create_session_factory
from app.routers.devices import router as devices_router
from app.routers.slm import router as slm_router
from app.routers.ws import router as websocket_router
from app.services.events import EventManager
from app.services.poller import DevicePoller

LOGGER = logging.getLogger(__name__)
BACKEND_DIR = Path(__file__).resolve().parents[1]


def _run_migrations(engine: Engine) -> None:
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.attributes["connection"] = engine.connect()
    try:
        command.upgrade(config, "head")
    finally:
        config.attributes["connection"].close()


def create_app(
    settings: Settings | None = None,
    *,
    run_poller: bool = True,
    run_migrations: bool = True,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    engine = create_db_engine(resolved_settings.database_url)
    sessions = create_session_factory(engine)
    events = EventManager()
    poller = DevicePoller(resolved_settings, sessions, events)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if run_migrations:
            _run_migrations(engine)
        if run_poller:
            await poller.start()
        yield
        if run_poller:
            await poller.stop()
        engine.dispose()

    app = FastAPI(
        title=resolved_settings.app_name,
        version="0.1.0",
        description="Digital Twin Network coordinator for real Jetson Orin Nano devices.",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.engine = engine
    app.state.session_factory = sessions
    app.state.events = events
    app.state.poller = poller

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.include_router(devices_router)
    app.include_router(slm_router)
    app.include_router(websocket_router)

    @app.get("/healthz", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(
        "/api/auth/check",
        tags=["auth"],
        dependencies=[Depends(require_api_token)],
    )
    def auth_check() -> dict[str, str]:
        return {"status": "authenticated"}

    @app.get("/api/openapi.json", include_in_schema=False)
    def openapi_json(request: Request) -> JSONResponse:
        return JSONResponse(request.app.openapi())

    return app


def main() -> None:
    import uvicorn

    settings = Settings.from_env()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    LOGGER.info("Starting DTN backend on %s:%s", settings.host, settings.port)
    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
