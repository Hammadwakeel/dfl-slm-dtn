"""Authenticated API for the always-on DFL supervisor agent."""

from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from app.core.security import require_api_token
from app.services.supervisor import Supervisor, SupervisorError

router = APIRouter(
    prefix="/api/supervisor",
    tags=["supervisor"],
    dependencies=[Depends(require_api_token)],
)


class AblationConfig(BaseModel):
    """One configuration an ablation run trains with the baseline's budget."""

    label: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9 _.,()+-]+$")
    architecture: dict[str, Any]
    topology: Literal["directed_ring", "bidirectional_ring", "mesh", "chain", "star", "learned"]
    aggregation: Literal["fedavg", "uniform", "metropolis", "trimmed_mean"]
    sync_mode: Literal["sync", "semi_sync", "async"]
    # Receiver -> the workers it receives from; required by (and only for) "learned",
    # so an ablation can train on a fixed graph such as a k-regular circulant.
    senders: dict[str, list[str]] | None = Field(None, min_length=2, max_length=64)
    # Overrides of the baseline's budget, e.g. more rounds of fewer local epochs.
    rounds: int | None = Field(None, ge=1, le=100)
    local_epochs: int | None = Field(None, ge=1, le=50)

    @model_validator(mode="after")
    def senders_match_topology(self) -> "AblationConfig":
        if (self.topology == "learned") != (self.senders is not None):
            raise ValueError("senders are required for, and only allowed with, the learned topology")
        return self


class GoalRequest(BaseModel):
    dataset: Literal["mnist", "cifar10", "fashion_mnist", "femnist"]
    target_accuracy: float = Field(0.95, ge=0.1, le=0.999)
    max_experiments: int = Field(6, ge=1, le=20)
    time_budget_minutes: int = Field(480, ge=10, le=2880)
    dataset_split: Literal[
        "stratified", "dirichlet_a01", "dirichlet_a03", "dirichlet_a05", "dirichlet_a07", "dirichlet_a09"
    ] = "dirichlet_a03"
    rounds: int = Field(10, ge=1, le=20)
    local_epochs: int = Field(5, ge=1, le=20)
    topology: Literal["directed_ring", "bidirectional_ring", "mesh", "chain", "star"] = "directed_ring"
    nas_budget_seconds: int = Field(600, ge=90, le=600)
    aggregation: Literal["fedavg", "uniform", "metropolis", "trimmed_mean"] = "fedavg"
    sync_mode: Literal["sync", "semi_sync", "async"] = "semi_sync"
    sync_deadline_seconds: int = Field(300, ge=30, le=3600)
    seed: int = Field(2026, ge=0, le=4_294_967_295)
    # None = dataset default (flip_shift for CIFAR-10, none for MNIST).
    augmentation: Literal["none", "flip_shift"] | None = None
    # v2: shared-split objective, extended CIFAR-10 space, federated finalist check.
    nas_version: Literal["v1", "v2"] = "v2"
    # Fair evaluation: every training uses rounds x local_epochs from the same initial
    # weights, and each search step is followed by a measurement of its effect.
    measure_each_step: bool = False
    # Fair mode keeps a step's change when its measurement is at least the best
    # accuracy minus this tolerance. Set it from measured run-to-run noise; a negative
    # value keeps a change only when it gains at least that much.
    accept_tolerance: float = Field(0.005, ge=-0.2, le=0.2)
    # Ablation: train these configurations in order after the baseline (no search,
    # no language-model decisions), each with the baseline's rounds x local_epochs.
    ablation_configs: list[AblationConfig] | None = Field(None, min_length=1, max_length=12)


class StopRequest(BaseModel):
    reason: str = Field("Stopped by the operator.", min_length=1, max_length=500)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class EvaluateRequest(BaseModel):
    dataset: Literal["mnist", "cifar10", "fashion_mnist", "femnist"]
    # Only score models from this master run id (all workers' newest otherwise).
    run_id: str | None = Field(None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")


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


@router.post("/evaluate")
def evaluate(body: EvaluateRequest, supervisor: Supervisor = Depends(get_supervisor)) -> dict[str, Any]:
    """Global validation + official test accuracy of every worker's newest model."""
    return _call(lambda: supervisor.evaluate_newest_models(body.dataset, body.run_id))
