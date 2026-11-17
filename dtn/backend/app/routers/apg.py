"""Authenticated APG simulation jobs and read-only result access."""

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.core.security import require_api_token
from app.services.apg import (
    APGJobManager,
    APGResultsError,
    DATA_SETTINGS,
    build_verdict,
    detailed_run,
    summarize_run,
)

router = APIRouter(
    prefix="/api/apg",
    tags=["APG topology learning"],
    dependencies=[Depends(require_api_token)],
)


class StartAPGJobRequest(BaseModel):
    topology: Literal["apg", "ring", "mesh"]
    data_setting: Literal["iid_stratified", "constrained_dirichlet_a01"]
    seed: Literal[17, 42, 2026]


def get_apg_jobs(request: Request) -> APGJobManager:
    manager = getattr(request.app.state, "apg_jobs", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="APG simulation service is unavailable")
    return manager


def _call(callback: Any) -> Any:
    try:
        return callback()
    except APGResultsError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@router.get("/runs")
def list_runs(manager: APGJobManager = Depends(get_apg_jobs)) -> dict[str, Any]:
    runs = _call(manager.load_runs)
    summaries = [summarize_run(run) for run in runs]
    verdicts = (
        [
            build_verdict(runs, setting, mode)
            for setting in sorted(DATA_SETTINGS)
            for mode in ("all_simulated", "hybrid_shaped_simulation")
        ]
        if runs
        else []
    )
    return {
        "runs": summaries,
        "jobs": manager.list_jobs(),
        "verdicts": verdicts,
    }


@router.get("/runs/{run_id}")
def get_run(
    run_id: str,
    manager: APGJobManager = Depends(get_apg_jobs),
) -> dict[str, Any]:
    run = _call(lambda: manager.find_run(run_id))
    if run is None:
        raise HTTPException(status_code=404, detail="APG run was not found")
    return {"summary": summarize_run(run), "run": detailed_run(run)}


@router.get("/jobs")
def list_jobs(manager: APGJobManager = Depends(get_apg_jobs)) -> dict[str, Any]:
    return {"jobs": manager.list_jobs()}


@router.post("/jobs", status_code=202)
def start_job(
    body: StartAPGJobRequest,
    manager: APGJobManager = Depends(get_apg_jobs),
) -> dict[str, Any]:
    try:
        return manager.start(body.topology, body.data_setting, body.seed)
    except PermissionError as error:
        raise HTTPException(status_code=423, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    manager: APGJobManager = Depends(get_apg_jobs),
) -> dict[str, Any]:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="APG job was not found")
    return job
