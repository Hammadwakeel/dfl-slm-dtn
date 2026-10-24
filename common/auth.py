"""Bearer-token authentication helpers for trusted node API calls."""

import hmac


def is_authorized(authorization: str | None, expected_token: str) -> bool:
    """Compare a bearer token without leaking the expected token via timing."""
    if not authorization or not authorization.startswith("Bearer "):
        return False

    supplied_token = authorization.removeprefix("Bearer ")
    if not supplied_token:
        return False
    return hmac.compare_digest(supplied_token.encode("utf-8"), expected_token.encode("utf-8"))
