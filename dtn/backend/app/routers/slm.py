"""Authenticated coordinator proxy for the SLM master service."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest, urlopen

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.security import require_api_token

router = APIRouter(
    prefix="/api/slm",
    tags=["slm-assistant"],
    dependencies=[Depends(require_api_token)],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def _forward(
    settings: Settings,
    path: str,
    *,
    payload: dict[str, object] | None = None,
    query: str = "",
) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    url = f"{settings.master_slm_url}{path}{query}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {settings.master_slm_token}",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    upstream = UrlRequest(url, data=body, headers=headers, method="POST" if body else "GET")
    try:
        with urlopen(upstream, timeout=settings.slm_timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        if 400 <= error.code < 500:
            raise HTTPException(
                status_code=error.code,
                detail="The SLM service rejected the request",
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


@router.post("/chat")
def chat(
    body: ChatRequest,
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    return _forward(settings, "/chat", payload={"message": body.message})


@router.post("/command")
def command(
    body: dict[str, object],
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    return _forward(settings, "/command", payload=body)


@router.get("/status")
def status(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    return _forward(settings, "/status")


@router.get("/logs")
def logs(
    limit: int = Query(default=50, ge=1, le=100),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    return _forward(settings, "/logs", query=f"?limit={limit}")
