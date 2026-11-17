"""Digital Twin Network FastAPI application factory."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from alembic import command
from alembic.config import Config
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.engine import Engine

from app.core.config import REPO_ROOT, Settings
from app.core.security import require_api_token
from app.db.session import create_db_engine, create_session_factory
from app.routers.apg import router as apg_router
from app.routers.devices import router as devices_router
from app.routers.autonomy import router as autonomy_router
from app.routers.database_assistant import router as database_assistant_router
from app.routers.datasets import router as datasets_router
from app.routers.agent_runs import router as agent_runs_router
from app.routers.slm import router as slm_router
from app.routers.ws import router as websocket_router
from app.services.autonomy import AutonomyController, AutonomyError
from app.services.apg import APGJobManager
from app.services.agent_runs import AgentRunManager
from app.services.cluster import cluster_nodes, sync_training_registry
from app.services.dataset_pipeline import preprocess_and_ship
from app.services.datasets import DatasetError, DatasetStore
from app.services.events import EventManager
from app.services.initial_models import InitialModelError, create_and_distribute_initial_model
from app.services.poller import DevicePoller
from app.services.preprocessing import PreprocessingError
from app.services.worker_shards import WorkerShardError
from app.routers.slm import _forward

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
    agent_runs = AgentRunManager(sessions, resolved_settings.database_url)
    events = EventManager()
    poller = DevicePoller(resolved_settings, sessions, events)
    dataset_store = DatasetStore(resolved_settings.dataset_root)

    def execute_action(action: dict[str, Any]) -> dict[str, Any]:
        if action["action"] == "preprocess_dataset":
            parameters = action["parameters"]
            try:
                report = preprocess_and_ship(
                    app.state.datasets,
                    cluster_nodes(resolved_settings, sessions),
                    parameters["dataset_name"],
                    normalization=parameters.get("normalization", "standard"),
                    split_strategy=parameters.get("dataset_split", "stratified"),
                    target_size=(
                        int(parameters["target_size"]["height"]),
                        int(parameters["target_size"]["width"]),
                    )
                    if "target_size" in parameters
                    else (28, 28),
                    seed=parameters.get("seed", 42),
                    replace_existing=parameters.get("replace_existing", False),
                )
            except DatasetError as error:
                raise AutonomyError(str(error), error.status_code) from error
            except (PreprocessingError, WorkerShardError) as error:
                status_code = 502 if isinstance(error, WorkerShardError) else 422
                raise AutonomyError(str(error), status_code) from error
            return {"available": True, "data": report}
        if action["action"] == "create_initial_model":
            dataset_name = action["parameters"]["model_name"]
            try:
                architecture = (
                    {"architecture": action["parameters"]["architecture"]}
                    if action["parameters"].get("architecture")
                    else {}
                )
                report = create_and_distribute_initial_model(
                    app.state.datasets,
                    cluster_nodes(resolved_settings, sessions),
                    dataset_name,
                    seed=action["parameters"].get("seed", 0),
                    **architecture,
                )
            except DatasetError as error:
                raise AutonomyError(str(error), error.status_code) from error
            except (InitialModelError, WorkerShardError) as error:
                status_code = 502 if isinstance(error, WorkerShardError) else 422
                raise AutonomyError(str(error), status_code) from error
            return {"available": True, "data": report}
        return _forward(resolved_settings, "/command", payload=action)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if run_migrations:
            _run_migrations(engine)
        agent_runs.recover_interrupted_runs()
        app.state.autonomy = AutonomyController(
            sessions,
            execute_action,
        )
        sync_training_registry(resolved_settings, sessions)
        app.state.apg_jobs = APGJobManager(
            REPO_ROOT / "results_apg",
            sessions,
            stop_requested=lambda: bool(app.state.autonomy.state()["stop_requested"]),
        )
        if run_poller:
            await poller.start()
        yield
        app.state.apg_jobs.shutdown()
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
    app.state.autonomy = None
    app.state.apg_jobs = None
    app.state.agent_runs = agent_runs
    app.state.datasets = dataset_store

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["DELETE", "GET", "PATCH", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.include_router(devices_router)
    app.include_router(apg_router)
    app.include_router(autonomy_router)
    app.include_router(database_assistant_router)
    app.include_router(datasets_router)
    app.include_router(slm_router)
    app.include_router(agent_runs_router)
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
