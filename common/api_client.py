"""Small standard-library HTTP client for authenticated Jetson APIs."""

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ApiClientError(RuntimeError):
    """Raised when a node request fails or returns invalid JSON."""


@dataclass(frozen=True)
class NodeEndpoint:
    node_id: str
    base_url: str
    token: str


class ApiClient:
    """Send token-authenticated JSON requests to node status APIs."""

    def __init__(self, endpoint: NodeEndpoint, timeout_seconds: float = 5.0) -> None:
        if not endpoint.token:
            raise ValueError("Node API token must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("Request timeout must be positive")
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds

    def get_status(self) -> dict[str, Any]:
        """Fetch the authenticated live status for this node."""
        url = self._endpoint.base_url.rstrip("/") + "/v1/status"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._endpoint.token}",
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise ApiClientError(
                f"Node {self._endpoint.node_id} returned HTTP {error.code}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise ApiClientError(
                f"Could not reach node {self._endpoint.node_id}: {error}"
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ApiClientError(
                f"Node {self._endpoint.node_id} returned invalid JSON"
            ) from error

        if not isinstance(payload, dict):
            raise ApiClientError(f"Node {self._endpoint.node_id} returned a non-object response")
        return payload
