"""Authenticated automatic action, stop, and loop controls."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.core.security import require_api_token
from app.services.autonomy import AutonomyController, AutonomyError

router = APIRouter(
    prefix="/api/autonomy",
    tags=["autonomy"],
    dependencies=[Depends(require_api_token)],
)


class ActionRequest(BaseModel):
    action: dict[str, Any]
    reason: str = Field(default="Authenticated coordinator action", max_length=4000)
    evidence: dict[str, Any] = Field(default_factory=dict)


class LoopRequest(BaseModel):
    max_iterations: int = Field(default=10, ge=1, le=100)
    time_limit_seconds: int = Field(default=3600, ge=1, le=86400)
    stop_rule: str = "max_iterations"
    target_metric: float | None = None
    workflow_id: UUID | None = None


class IterationRequest(BaseModel):
    metric: float | None = None
    worker_failure: str | None = Field(default=None, max_length=1000)


class LoopStopRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=1000)


class WorkflowDeleteRequest(BaseModel):
    workflow_ids: list[str] = Field(default_factory=list, max_length=500)
    conversation_ids: list[UUID] = Field(default_factory=list, max_length=500)


def get_autonomy(request: Request) -> AutonomyController:
    controller = request.app.state.autonomy
    if controller is None:
        raise HTTPException(status_code=503, detail="Autonomy control is not initialized")
    return controller


def _call(callback: Any) -> Any:
    try:
        return callback()
    except AutonomyError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.get("/state")
def state(controller: AutonomyController = Depends(get_autonomy)) -> dict[str, Any]:
    return controller.state()


@router.post("/stop")
def stop(request: Request, controller: AutonomyController = Depends(get_autonomy)) -> dict[str, Any]:
    state = controller.stop()
    apg_jobs = getattr(request.app.state, "apg_jobs", None)
    if apg_jobs is not None:
        apg_jobs.request_stop_all()
    return state


@router.post("/reset-stop")
def reset_stop(controller: AutonomyController = Depends(get_autonomy)) -> dict[str, Any]:
    return controller.reset_stop()


@router.post("/dispatch")
def dispatch(
    request: Request,
    body: ActionRequest,
    controller: AutonomyController = Depends(get_autonomy),
) -> dict[str, Any]:
    result = _call(
        lambda: controller.dispatch(
            body.action,
            reason=body.reason,
            evidence=body.evidence,
        )
    )
    from app.routers.slm import _record_completed_master_action

    history_warning = _record_completed_master_action(
        request.app.state.settings,
        body.action,
        result,
    )
    if history_warning:
        result["history_warning"] = history_warning
    return result


@router.get("/history")
def history(
    limit: int = Query(default=250, ge=1, le=500),
    controller: AutonomyController = Depends(get_autonomy),
) -> list[dict[str, Any]]:
    return controller.history(limit=limit)


@router.delete("/workflows")
def delete_workflows(
    body: WorkflowDeleteRequest,
    controller: AutonomyController = Depends(get_autonomy),
) -> dict[str, int]:
    return _call(
        lambda: controller.delete_workflows(
            body.workflow_ids,
            [str(conversation_id) for conversation_id in body.conversation_ids],
        )
    )


@router.post("/loops")
def start_loop(
    body: LoopRequest = Body(default_factory=LoopRequest),
    controller: AutonomyController = Depends(get_autonomy),
) -> dict[str, Any]:
    return _call(
        lambda: controller.start_loop(
            max_iterations=body.max_iterations,
            time_limit_seconds=body.time_limit_seconds,
            stop_rule=body.stop_rule,
            target_metric=body.target_metric,
            workflow_id=str(body.workflow_id) if body.workflow_id else None,
        )
    )


@router.post("/loops/{loop_id}/iterations")
def record_iteration(
    loop_id: str,
    body: IterationRequest,
    controller: AutonomyController = Depends(get_autonomy),
) -> dict[str, Any]:
    return _call(
        lambda: controller.record_iteration(
            loop_id,
            metric=body.metric,
            worker_failure=body.worker_failure,
        )
    )


@router.post("/loops/{loop_id}/stop")
def stop_loop(
    loop_id: str,
    body: LoopStopRequest,
    controller: AutonomyController = Depends(get_autonomy),
) -> dict[str, Any]:
    return _call(lambda: controller.stop_loop(loop_id, body.reason))
