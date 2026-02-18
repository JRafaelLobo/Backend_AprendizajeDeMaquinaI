from datetime import datetime, timedelta, timezone

import jwt
from django.conf import settings


def _create_token(user_id: str, token_type: str, ttl_seconds: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "type": token_type,
        "iat": now,
        "exp": now + timedelta(seconds=ttl_seconds),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


def create_access_token(user_id: str) -> str:
    return _create_token(user_id=user_id, token_type="access", ttl_seconds=settings.JWT_ACCESS_TTL)


def create_refresh_token(user_id: str) -> str:
    return _create_token(user_id=user_id, token_type="refresh", ttl_seconds=settings.JWT_REFRESH_TTL)


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
