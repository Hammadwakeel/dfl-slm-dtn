"""Authenticated coordinator proxy for the SLM master's command and results API.

Human-mode chat goes from the browser straight to the master's read-only RAG endpoints;
agentic operation is handled by the server-side supervisor (app.services.supervisor).
"""

import json
import logging
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import ProxyHandler, Request as UrlRequest, build_opener

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.core.config import Settings
from app.core.security import require_api_token
from app.routers.autonomy import get_autonomy
from app.services.autonomy import AutonomyController, AutonomyError

router = APIRouter(
    prefix="/api/slm",
    tags=["slm"],
    dependencies=[Depends(require_api_token)],
)
LOGGER = logging.getLogger(__name__)
ACTION_HISTORY_EXCLUDED = frozenset(
    {
        "get_status",
        "get_logs",
        "get_results",
        "compare_runs",
        "suggest_next_action",
        "clarify",
        "refuse",
        "redirect",
    }
)


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def _forward(
    settings: Settings,
    path: str,
    *,
    payload: dict[str, object] | None = None,
    query: str = "",
    method: str | None = None,
) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    url = f"{settings.master_slm_url}{path}{query}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {settings.master_slm_token}",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    upstream = UrlRequest(
        url,
        data=body,
        headers=headers,
        method=method or ("POST" if body else "GET"),
    )
    try:
        opener = build_opener(ProxyHandler({}))
        with opener.open(upstream, timeout=settings.slm_timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        if 400 <= error.code < 500:
            upstream_detail = ""
            try:
                body = json.loads(error.read(4096).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                body = None
            if isinstance(body, dict):
                message = body.get("error", body.get("detail"))
                if isinstance(message, str):
                    upstream_detail = message.strip()[:300]
            raise HTTPException(
                status_code=error.code,
                detail=(
                    f"The master request failed (HTTP {error.code})"
                    + (f": {upstream_detail}" if upstream_detail else "")
                ),
            ) from error
        raise HTTPException(status_code=502, detail="The SLM service returned an error") from error
    except (URLError, TimeoutError, OSError) as error:
        raise HTTPException(
            status_code=502,
            detail="The SLM master is currently unreachable",
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=502,
            detail="The SLM master returned an invalid response",
        ) from error
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="The SLM master returned an invalid response")
    return result


def _record_completed_master_action(
    settings: Settings,
    action: dict[str, object],
    dispatch_result: dict[str, object],
) -> str | None:
    if action.get("action") in ACTION_HISTORY_EXCLUDED:
        return None
    if dispatch_result.get("status") != "executed":
        return None
    result = dispatch_result.get("result")
    if not isinstance(result, dict) or result.get("available") is not True:
        return None
    try:
        recorded = _forward(
            settings,
            "/action-history",
            payload={"action": action, "result": result},
        )
    except HTTPException as error:
        warning = (
            "The action completed, but the master did not save its history record "
            f"(HTTP {error.status_code}). Do not repeat the action; update the master service."
        )
        LOGGER.warning(
            "Completed action=%s but master history sync failed with HTTP %s",
            action.get("action"),
            error.status_code,
        )
        return warning
    if recorded.get("recorded") is not True:
        warning = (
            "The action completed, but the master did not save its history record. "
            "Do not repeat the action; check the SLM service."
        )
        LOGGER.warning(
            "Completed action=%s but master did not confirm history sync",
            action.get("action"),
        )
        return warning
    return None


@router.post("/command")
def command(
    body: dict[str, object],
    settings: Settings = Depends(get_settings),
    controller: AutonomyController = Depends(get_autonomy),
) -> dict[str, object]:
    if set(body) - {"action", "reason", "evidence"}:
        raise HTTPException(status_code=422, detail="Unexpected command fields")
    action = body.get("action")
    if not isinstance(action, dict):
        raise HTTPException(status_code=422, detail="An action object is required")
    reason = body.get("reason", "Authenticated coordinator command")
    evidence = body.get("evidence", {})
    if not isinstance(reason, str) or not isinstance(evidence, dict):
        raise HTTPException(status_code=422, detail="Invalid reason or evidence")
    try:
        result = controller.dispatch(action, reason=reason, evidence=evidence)
        history_warning = _record_completed_master_action(settings, action, result)
        if history_warning:
            result["history_warning"] = history_warning
        return result
    except AutonomyError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.get("/status")
def status(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    return _forward(settings, "/status")


@router.get("/logs")
def logs(
    limit: int = Query(default=50, ge=1, le=100),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    return _forward(settings, "/logs", query=f"?limit={limit}")


# The dashboard polls results every few seconds from every open tab, and one
# master read serializes megabytes on a Jetson behind Wi-Fi. Overlapping polls
# piled up (5 concurrent reads kept the master at 100% CPU and ~5 MB/s uplink,
# congesting the whole cluster network), so reads are single-flight and cached.
RESULTS_CACHE_SECONDS = 10.0
_RESULTS_CACHE: dict[tuple[str, str], tuple[float, dict[str, object]]] = {}
_RESULTS_LOCK = threading.Lock()


def _cached_results(settings: Settings, query: str) -> dict[str, object]:
    key = (settings.master_slm_url, query)
    with _RESULTS_LOCK:
        cached = _RESULTS_CACHE.get(key)
        if cached is not None and time.monotonic() - cached[0] < RESULTS_CACHE_SECONDS:
            return cached[1]
        result = _forward(settings, "/results", query=query)
        _RESULTS_CACHE[key] = (time.monotonic(), result)
        return result


def clear_results_cache() -> None:
    with _RESULTS_LOCK:
        _RESULTS_CACHE.clear()


@router.get("/results")
def results(
    group_id: str | None = Query(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    limit: int = Query(default=20, ge=1, le=20),
    compact: bool = Query(default=True),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    """Master results; ask for one group_id or a small limit (all 20 groups can be ~40 MB).

    compact (default) keeps the per-worker architecture, confusion matrices and
    per-class arrays only in each run's final round; compact=false returns them all.
    """
    query = f"?group_id={quote(group_id, safe='')}" if group_id else f"?limit={limit}"
    query += "&compact=1" if compact else "&compact=0"
    return _cached_results(settings, query)


@router.delete("/results/{group_id}")
def delete_result_group(
    group_id: str,
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    deleted = _forward(
        settings,
        f"/results/{quote(group_id, safe='')}",
        method="DELETE",
    )
    clear_results_cache()
    return deleted
