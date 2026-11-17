"""Authenticated action dispatch, global STOP, and execution history."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
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
