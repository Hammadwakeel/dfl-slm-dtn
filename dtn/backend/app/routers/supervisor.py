"""Authenticated API for the always-on DFL supervisor agent."""

from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.security import require_api_token
from app.services.supervisor import Supervisor, SupervisorError

router = APIRouter(
    prefix="/api/supervisor",
    tags=["supervisor"],
    dependencies=[Depends(require_api_token)],
)


class GoalRequest(BaseModel):
    dataset: Literal["mnist", "cifar10"]
    target_accuracy: float = Field(0.95, ge=0.1, le=0.999)
    max_experiments: int = Field(6, ge=1, le=20)
    time_budget_minutes: int = Field(480, ge=10, le=2880)
    dataset_split: Literal[
        "stratified", "dirichlet_a01", "dirichlet_a03", "dirichlet_a05", "dirichlet_a07"
    ] = "dirichlet_a03"
    rounds: int = Field(10, ge=1, le=20)
    local_epochs: int = Field(5, ge=1, le=20)
    topology: Literal["directed_ring", "bidirectional_ring", "mesh", "chain", "star"] = "directed_ring"
    nas_budget_seconds: int = Field(600, ge=90, le=600)
    seed: int = Field(2026, ge=0, le=4_294_967_295)


class StopRequest(BaseModel):
    reason: str = Field("Stopped by the operator.", min_length=1, max_length=500)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def get_supervisor(request: Request) -> Supervisor:
    supervisor = getattr(request.app.state, "supervisor", None)
    if supervisor is None:
        raise HTTPException(status_code=503, detail="The supervisor is not running")
    return supervisor


def _call(callback: Any) -> Any:
    try:
        return callback()
    except SupervisorError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.get("/runs")
def list_runs(supervisor: Supervisor = Depends(get_supervisor)) -> list[dict[str, Any]]:
    return supervisor.list_runs()


@router.post("/runs", status_code=201)
def create_run(
    goal: GoalRequest,
    supervisor: Supervisor = Depends(get_supervisor),
) -> dict[str, Any]:
    return _call(lambda: supervisor.create_run(goal.model_dump()))


@router.get("/runs/{run_id}")
def get_run(run_id: UUID, supervisor: Supervisor = Depends(get_supervisor)) -> dict[str, Any]:
    return _call(lambda: supervisor.get_run(str(run_id)))


@router.post("/runs/{run_id}/stop")
def stop_run(
    run_id: UUID,
    body: StopRequest | None = None,
    supervisor: Supervisor = Depends(get_supervisor),
) -> dict[str, Any]:
    reason = body.reason if body else "Stopped by the operator."
    return _call(lambda: supervisor.request_stop(str(run_id), reason))


@router.post("/runs/{run_id}/ask")
def ask(
    run_id: UUID,
    body: QuestionRequest,
    supervisor: Supervisor = Depends(get_supervisor),
) -> dict[str, Any]:
    return _call(lambda: supervisor.ask(str(run_id), body.question.strip()))


@router.get("/cluster")
def cluster(supervisor: Supervisor = Depends(get_supervisor)) -> dict[str, Any]:
    return supervisor.cluster_summary()
