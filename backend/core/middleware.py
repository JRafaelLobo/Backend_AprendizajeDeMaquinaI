import jwt

from core.jwt import decode_token
from core.utils import extract_bearer_token


class JWTContextMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request.jwt_payload = None
        token = extract_bearer_token(request.headers.get("Authorization", ""))
        if token:
            try:
                request.jwt_payload = decode_token(token)
            except jwt.InvalidTokenError:
                request.jwt_payload = None
        return self.get_response(request)
