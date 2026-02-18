from datetime import datetime, timezone


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def extract_bearer_token(authorization_header: str) -> str | None:
    if not authorization_header:
        return None
    parts = authorization_header.strip().split()
    if len(parts) != 2 or parts[0] != "Bearer":
        return None
    return parts[1]
