import jwt
from bson import ObjectId
from bson.errors import InvalidId
from drf_yasg.utils import swagger_auto_schema
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from core.jwt import create_access_token, create_refresh_token, decode_token
from core.utils import now_utc
from users.models import User
from users.serializers import AuthResponseSerializer, LoginSerializer, RefreshResponseSerializer
from users.serializers import RefreshSerializer, RegisterSerializer, UserSerializer

USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"


def _get_user_document(user_id: str):
    try:
        object_id = ObjectId(user_id)
    except InvalidId:
        return None
    return User.objects(id=object_id).first()


def _get_authenticated_user_document(request):
    request_user = getattr(request, "user", None)
    if not getattr(request_user, "is_authenticated", False):
        return None, Response({"detail": "Autenticación requerida"}, status=status.HTTP_401_UNAUTHORIZED)

    jwt_payload = getattr(request, "jwt_payload", None)
    token_subject = jwt_payload.get("sub") if isinstance(jwt_payload, dict) else None
    if token_subject and token_subject != request_user.id:
        return None, Response({"detail": "Token inválido"}, status=status.HTTP_401_UNAUTHORIZED)

    user = _get_user_document(request_user.id)
    if not user:
        return None, Response({"detail": "Usuario no encontrado"}, status=status.HTTP_404_NOT_FOUND)

    return user, None


def _serialize_message(message):
    role = ASSISTANT_ROLE if message.sender_id == ASSISTANT_ROLE else USER_ROLE
    return {
        "content": message.content,
        "senderId": message.sender_id,
        "sendTime": message.send_time,
        "isAI": role == ASSISTANT_ROLE,
        "role": role,
    }


def _serialize_chat(chat, owner_id: str):
    return {
        "id": chat.id,
        "title": chat.title,
        "ownerId": owner_id,
        "participantA": owner_id,
        "participantB": ASSISTANT_ROLE,
        "messages": [_serialize_message(message) for message in chat.messages],
    }


def _serialize_user(user):
    owner_id = str(user.id)
    return {
        "id": owner_id,
        "email": user.email,
        "chats": [_serialize_chat(chat, owner_id) for chat in user.chats],
    }


class RegisterView(APIView):
    permission_classes = [permissions.AllowAny]

    @swagger_auto_schema(
        tags=["Users"],
        request_body=RegisterSerializer,
        responses={201: AuthResponseSerializer},
    )
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        if User.objects(email=data["email"]).first():
            return Response({"detail": "El email ya está registrado"}, status=status.HTTP_400_BAD_REQUEST)

        now = now_utc()
        user = User(
            email=data["email"],
            chats=[],
            created_at=now,
            updated_at=now,
        )
        user.set_password(data["password"])
        user.save()

        user_id = str(user.id)
        return Response(
            {
                "user": _serialize_user(user),
                "access_token": create_access_token(user_id),
                "refresh_token": create_refresh_token(user_id),
            },
            status=status.HTTP_201_CREATED,
        )


class LoginView(APIView):
    permission_classes = [permissions.AllowAny]

    @swagger_auto_schema(
        tags=["Login / Auth"],
        request_body=LoginSerializer,
        responses={200: AuthResponseSerializer},
    )
    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        user = User.objects(email=data["email"]).first()
        if not user or not user.check_password(data["password"]):
            return Response({"detail": "Credenciales inválidas"}, status=status.HTTP_401_UNAUTHORIZED)

        user_id = str(user.id)
        return Response(
            {
                "access_token": create_access_token(user_id),
                "refresh_token": create_refresh_token(user_id),
                "user": _serialize_user(user),
            }
        )


class RefreshView(APIView):
    permission_classes = [permissions.AllowAny]

    @swagger_auto_schema(
        tags=["Login / Auth"],
        request_body=RefreshSerializer,
        responses={200: RefreshResponseSerializer},
    )
    def post(self, request):
        serializer = RefreshSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        refresh_token = serializer.validated_data["refresh_token"]

        try:
            payload = decode_token(refresh_token)
        except jwt.ExpiredSignatureError:
            return Response({"detail": "Refresh token expirado"}, status=status.HTTP_401_UNAUTHORIZED)
        except jwt.InvalidTokenError:
            return Response({"detail": "Refresh token inválido"}, status=status.HTTP_401_UNAUTHORIZED)

        if payload.get("type") != "refresh":
            return Response({"detail": "Token incorrecto"}, status=status.HTTP_401_UNAUTHORIZED)

        user_id = payload.get("sub")
        if not user_id:
            return Response({"detail": "Usuario no encontrado"}, status=status.HTTP_401_UNAUTHORIZED)

        try:
            object_id = ObjectId(user_id)
        except InvalidId:
            return Response({"detail": "Usuario no encontrado"}, status=status.HTTP_401_UNAUTHORIZED)

        if not User.objects(id=object_id).first():
            return Response({"detail": "Usuario no encontrado"}, status=status.HTTP_401_UNAUTHORIZED)

        return Response(
            {
                "access_token": create_access_token(user_id),
                "refresh_token": create_refresh_token(user_id),
            }
        )


class MeView(APIView):
    @swagger_auto_schema(
        tags=["Users"],
        responses={200: UserSerializer},
    )
    def get(self, request):
        user, error_response = _get_authenticated_user_document(request)
        if error_response:
            return error_response

        return Response(_serialize_user(user))