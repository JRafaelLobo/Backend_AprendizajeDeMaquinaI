from dataclasses import dataclass

import jwt
from bson import ObjectId
from bson.errors import InvalidId
from rest_framework import exceptions
from rest_framework.authentication import BaseAuthentication

from core.jwt import decode_token
from users.models import User


@dataclass
class AuthenticatedUser:
    id: str
    email: str
    is_authenticated: bool = True


class JWTAuthentication(BaseAuthentication):
    keyword = "Bearer"

    def authenticate_header(self, request):
        return self.keyword

    def authenticate(self, request):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header:
            return None

        parts = auth_header.split()
        if len(parts) == 1:
            token = parts[0]
        elif len(parts) == 2 and parts[0].lower() == self.keyword.lower():
            token = parts[1]
        else:
            raise exceptions.AuthenticationFailed("Authorization header inválido")

        try:
            payload = decode_token(token)
        except jwt.ExpiredSignatureError as exc:
            raise exceptions.AuthenticationFailed("Token expirado") from exc
        except jwt.InvalidTokenError as exc:
            raise exceptions.AuthenticationFailed("Token inválido") from exc

        if payload.get("type") != "access":
            raise exceptions.AuthenticationFailed("Se requiere access token")

        user_id = payload.get("sub")
        if not user_id:
            raise exceptions.AuthenticationFailed("Token sin subject")

        try:
            object_id = ObjectId(user_id)
        except InvalidId as exc:
            raise exceptions.AuthenticationFailed("Subject inválido") from exc

        user = User.objects(id=object_id).first()
        if not user:
            raise exceptions.AuthenticationFailed("Usuario no encontrado")

        return AuthenticatedUser(id=str(user.id), email=user.email), None