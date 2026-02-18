from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from drf_yasg.utils import swagger_auto_schema
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from chat.serializers import ChatCreateSerializer, ChatListResponseSerializer, ChatSummarySerializer
from chat.serializers import ChatWithMessagesSerializer, MessageCreateSerializer, MessageSerializer
from users.models import Chat, Message, User


def _serialize_message(message):
    return {
        "content": message.content,
        "senderId": message.sender_id,
        "sendTime": message.send_time,
    }


def _serialize_chat(chat):
    return {
        "id": chat.id,
        "title": chat.title,
        "participantA": chat.participant_a,
        "participantB": chat.participant_b,
    }


def _serialize_chat_with_messages(chat):
    return {
        "id": chat.id,
        "title": chat.title,
        "participantA": chat.participant_a,
        "participantB": chat.participant_b,
        "messages": [_serialize_message(message) for message in chat.messages],
    }


def _get_user_document(user_id: str):
    try:
        object_id = ObjectId(user_id)
    except InvalidId:
        return None
    return User.objects(id=object_id).first()


def _get_user_by_email(email: str):
    return User.objects(email=email).first()


def _find_chat(user: User, chat_id: int):
    for chat in user.chats:
        if chat.id == chat_id:
            return chat
    return None


def _chat_participants(chat: Chat):
    return {chat.participant_a, chat.participant_b}


def _is_chat_participant(chat: Chat, user_id: str):
    return user_id in _chat_participants(chat)


def _find_existing_conversation(user: User, first_user_id: str, second_user_id: str):
    wanted = {first_user_id, second_user_id}
    for chat in user.chats:
        if _chat_participants(chat) == wanted:
            return chat
    return None


def _sync_chat_for_user(user: User, chat: Chat):
    for index, existing_chat in enumerate(user.chats):
        if existing_chat.id == chat.id:
            user.chats[index] = chat
            user.updated_at = datetime.now(timezone.utc)
            user.save()
            return

    user.chats.append(chat)
    user.updated_at = datetime.now(timezone.utc)
    user.save()


class ChatListCreateView(APIView):
    @swagger_auto_schema(
        tags=["Chats"],
        responses={200: ChatListResponseSerializer},
    )
    def get(self, request):
        user = _get_user_document(request.user.id)
        if not user:
            return Response({"detail": "Usuario no encontrado"}, status=status.HTTP_404_NOT_FOUND)

        return Response({"chats": [_serialize_chat(chat) for chat in user.chats]})

    @swagger_auto_schema(
        tags=["Chats"],
        request_body=ChatCreateSerializer,
        responses={201: ChatSummarySerializer},
    )
    def post(self, request):
        serializer = ChatCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        current_user = _get_user_document(request.user.id)
        if not current_user:
            return Response({"detail": "Usuario no encontrado"}, status=status.HTTP_404_NOT_FOUND)

        participant_user = _get_user_by_email(serializer.validated_data["participant_email"])
        if not participant_user:
            return Response({"detail": "Participante no encontrado"}, status=status.HTTP_404_NOT_FOUND)

        if str(participant_user.id) == request.user.id:
            return Response({"detail": "No puedes crear chat contigo mismo"}, status=status.HTTP_400_BAD_REQUEST)

        existing = _find_existing_conversation(current_user, request.user.id, str(participant_user.id))
        if existing:
            return Response(_serialize_chat(existing), status=status.HTTP_200_OK)

        chat_id = int(datetime.now(timezone.utc).timestamp() * 1000)
        chat = Chat(
            id=chat_id,
            title=serializer.validated_data["title"],
            participant_a=request.user.id,
            participant_b=str(participant_user.id),
            messages=[],
        )

        _sync_chat_for_user(current_user, chat)
        _sync_chat_for_user(participant_user, chat)

        return Response(_serialize_chat(chat), status=status.HTTP_201_CREATED)


class MessageListCreateView(APIView):
    @swagger_auto_schema(
        tags=["Messages"],
        responses={200: ChatWithMessagesSerializer},
    )
    def get(self, request, chat_id: int):
        user = _get_user_document(request.user.id)
        if not user:
            return Response({"detail": "Usuario no encontrado"}, status=status.HTTP_404_NOT_FOUND)

        chat = _find_chat(user, chat_id)
        if not chat:
            return Response({"detail": "Chat no encontrado"}, status=status.HTTP_404_NOT_FOUND)
        if not _is_chat_participant(chat, request.user.id):
            return Response({"detail": "No autorizado para este chat"}, status=status.HTTP_403_FORBIDDEN)

        return Response(_serialize_chat_with_messages(chat))

    @swagger_auto_schema(
        tags=["Messages"],
        request_body=MessageCreateSerializer,
        responses={201: MessageSerializer},
    )
    def post(self, request, chat_id: int):
        serializer = MessageCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        current_user = _get_user_document(request.user.id)
        if not current_user:
            return Response({"detail": "Usuario no encontrado"}, status=status.HTTP_404_NOT_FOUND)

        chat = _find_chat(current_user, chat_id)
        if not chat:
            return Response({"detail": "Chat no encontrado"}, status=status.HTTP_404_NOT_FOUND)
        if not _is_chat_participant(chat, request.user.id):
            return Response({"detail": "No autorizado para este chat"}, status=status.HTTP_403_FORBIDDEN)

        other_user_id = chat.participant_b if chat.participant_a == request.user.id else chat.participant_a
        other_user = _get_user_document(other_user_id)
        if not other_user:
            return Response({"detail": "Participante no encontrado"}, status=status.HTTP_404_NOT_FOUND)

        message = Message(
            content=serializer.validated_data["content"],
            sender_id=request.user.id,
            send_time=datetime.now(timezone.utc),
        )
        chat.messages.append(message)
        _sync_chat_for_user(current_user, chat)
        _sync_chat_for_user(other_user, chat)

        return Response(_serialize_message(message), status=status.HTTP_201_CREATED)
