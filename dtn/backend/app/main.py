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
from app.routers.slm import router as slm_router
from app.routers.supervisor import router as supervisor_router
from app.routers.ws import router as websocket_router
from app.services.autonomy import AutonomyController, AutonomyError
from app.services.apg import APGJobManager
from app.services.cluster import cluster_nodes, sync_training_registry
from app.services.global_evaluation import evaluate_newest_models
from app.services.pruning import prune_and_distribute, redistribute_trained_model, restore_source_model
from app.services.dataset_pipeline import preprocess_and_ship
from app.services.datasets import DatasetError, DatasetStore
from app.services.events import EventManager
from app.services.initial_models import InitialModelError, create_and_distribute_initial_model
from app.services.poller import DevicePoller
from app.services.preprocessing import PreprocessingError
from app.services.supervisor import Supervisor
from app.services.worker_memory import cleanup_worker_memory_cluster
from app.services.worker_shards import WorkerShardError
from app.routers.slm import _forward, _record_completed_master_action

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
    run_supervisor: bool | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    if run_supervisor is None:
        run_supervisor = run_poller
    engine = create_db_engine(resolved_settings.database_url)
    sessions = create_session_factory(engine)
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
            if action["parameters"].get("from_trained_model") is True:
                try:
                    report = redistribute_trained_model(
                        app.state.datasets,
                        cluster_nodes(resolved_settings, sessions),
                        dataset_name,
                        source_worker=action["parameters"].get("source_worker"),
                        seed=action["parameters"].get("seed", 0),
                    )
                except (InitialModelError, WorkerShardError) as error:
                    status_code = 502 if isinstance(error, WorkerShardError) else 422
                    raise AutonomyError(str(error), status_code) from error
                return {"available": True, "data": report}
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
                    replace_existing=action["parameters"].get("replace_existing", False) is True,
                    restore_baseline=action["parameters"].get("restore_baseline", False) is True,
                    **architecture,
                )
            except DatasetError as error:
                raise AutonomyError(str(error), error.status_code) from error
            except (InitialModelError, WorkerShardError) as error:
                status_code = 502 if isinstance(error, WorkerShardError) else 422
                raise AutonomyError(str(error), status_code) from error
            return {"available": True, "data": report}
        if action["action"] == "prune_model":
            try:
                if action["parameters"].get("restore_source") is True:
                    report = restore_source_model(
                        app.state.datasets,
                        cluster_nodes(resolved_settings, sessions),
                        action["parameters"]["model_name"],
                        seed=action["parameters"].get("seed", 0),
                    )
                    return {"available": True, "data": report}
                report = prune_and_distribute(
                    app.state.datasets,
                    cluster_nodes(resolved_settings, sessions),
                    action["parameters"]["model_name"],
                    ratio=float(action["parameters"].get("ratio", 0.5)),
                    seed=action["parameters"].get("seed", 0),
                )
            except DatasetError as error:
                raise AutonomyError(str(error), error.status_code) from error
            except (InitialModelError, WorkerShardError) as error:
                status_code = 502 if isinstance(error, WorkerShardError) else 422
                raise AutonomyError(str(error), status_code) from error
            return {"available": True, "data": report}
        return _forward(resolved_settings, "/command", payload=action)

    def supervisor_dispatch(action: dict[str, Any], reason: str, run_id: str) -> dict[str, Any]:
        dispatched = app.state.autonomy.dispatch(
            action,
            reason=reason,
            evidence={"source": "supervisor", "run_id": run_id},
            mode="AGENT",
        )
        result = dispatched.get("result")
        if isinstance(result, dict) and result.get("available") is True:
            # Keep the master's action history (used by human-mode advisory chat) complete.
            _record_completed_master_action(resolved_settings, action, dispatched)
        return dispatched

    def supervisor_model_call(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"messages": messages}
        if tools:
            payload["tools"] = tools
        else:
            payload["allow_tools"] = False
        message = _forward(resolved_settings, "/agent/model", payload=payload).get("message")
        if not isinstance(message, dict):
            raise RuntimeError("The SLM master returned an invalid agent model response")
        return message

    def supervisor_memory_cleanup(recycle: bool = False) -> dict[str, Any]:
        workers = tuple(
            node
            for node in cluster_nodes(resolved_settings, sessions)
            if node.role == "worker"
        )
        return cleanup_worker_memory_cluster(workers, timeout_seconds=30.0, recycle=recycle)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if run_migrations:
            _run_migrations(engine)
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
        app.state.supervisor = Supervisor(
            sessions,
            dispatch=supervisor_dispatch,
            master_results=lambda: _forward(resolved_settings, "/results"),
            model_call=supervisor_model_call,
            memory_cleanup=supervisor_memory_cleanup,
            global_evaluate=lambda dataset_name: evaluate_newest_models(
                app.state.datasets,
                cluster_nodes(resolved_settings, sessions),
                dataset_name,
            ),
            min_free_memory_mb=resolved_settings.supervisor_min_free_memory_mb,
            tick_seconds=resolved_settings.supervisor_tick_seconds,
        )
        if run_supervisor:
            app.state.supervisor.start()
        if run_poller:
            await poller.start()
        yield
        app.state.supervisor.shutdown()
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
    app.state.supervisor = None
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
    app.include_router(supervisor_router)
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
