from functools import lru_cache
import logging

from bson import ObjectId
from bson.errors import InvalidId
from drf_yasg.utils import swagger_auto_schema
from rest_framework import serializers, status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from core.utils import now_utc
from users.models import Chat, Message, User

from .embeddings import (
    FaissIndexNotFoundError,
    ensure_faiss_index_exists,
    get_embeddings,
    get_faiss_index_path,
    is_faiss_index_ready,
    resolve_embeddings_mode,
)
from .serializers import ChatCreateSerializer, ChatListResponseSerializer, ChatSummarySerializer, ChatWithMessagesSerializer, MessageCreateSerializer

logger = logging.getLogger(__name__)

USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
NO_CHAT_HISTORY = "Sin historial previo."


def _serialize_message(message: Message):
    role = ASSISTANT_ROLE if message.sender_id == ASSISTANT_ROLE else USER_ROLE
    return {
        "content": message.content,
        "senderId": message.sender_id,
        "sendTime": message.send_time,
        "isAI": role == ASSISTANT_ROLE,
        "role": role,
    }


def _serialize_chat(chat: Chat, owner_id: str):
    return {
        "id": chat.id,
        "title": chat.title,
        "ownerId": owner_id,
        "participantA": owner_id,
        "participantB": ASSISTANT_ROLE,
    }


def _serialize_chat_with_messages(chat: Chat, owner_id: str):
    chat_data = _serialize_chat(chat, owner_id)
    chat_data["messages"] = [_serialize_message(message) for message in chat.messages]
    return chat_data


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
        logger.warning("Subject del token no coincide con el usuario autenticado")
        return None, Response({"detail": "Token inválido"}, status=status.HTTP_401_UNAUTHORIZED)

    user = _get_user_document(request_user.id)
    if not user:
        return None, Response({"detail": "Usuario no encontrado"}, status=status.HTTP_404_NOT_FOUND)

    return user, None


def _find_chat(user: User, chat_id: int):
    for chat in user.chats:
        if chat.id == chat_id:
            return chat
    return None


def _save_chat_for_user(user: User, chat: Chat):
    for index, existing_chat in enumerate(user.chats):
        if existing_chat.id == chat.id:
            user.chats[index] = chat
            user.updated_at = now_utc()
            user.save()
            return

    user.chats.append(chat)
    user.updated_at = now_utc()
    user.save()


def _delete_chat_for_user(user: User, chat_id: int) -> bool:
    for index, existing_chat in enumerate(user.chats):
        if existing_chat.id == chat_id:
            del user.chats[index]
            user.updated_at = now_utc()
            user.save()
            return True
    return False


def _build_chat_history(chat: Chat, latest_user_message: str | None = None) -> str:
    history_lines = []

    for message in chat.messages:
        speaker = "IA" if message.sender_id == ASSISTANT_ROLE else "Usuario"
        history_lines.append(f"{speaker}: {message.content}")

    if latest_user_message:
        history_lines.append(f"Usuario: {latest_user_message}")

    return "\n".join(history_lines) or NO_CHAT_HISTORY


def _normalize_chat_history_payload(chat_history) -> str:
    if not chat_history:
        return NO_CHAT_HISTORY

    if isinstance(chat_history, str):
        normalized_history = chat_history.strip()
        return normalized_history or NO_CHAT_HISTORY

    if isinstance(chat_history, list):
        history_lines = []
        for item in chat_history:
            if isinstance(item, dict):
                content = str(item.get("content", "")).strip()
                if not content:
                    continue

                role = item.get("role")
                if role not in {USER_ROLE, ASSISTANT_ROLE}:
                    role = ASSISTANT_ROLE if item.get("isAI") else USER_ROLE

                speaker = "IA" if role == ASSISTANT_ROLE else "Usuario"
                history_lines.append(f"{speaker}: {content}")
                continue

            if isinstance(item, str) and item.strip():
                history_lines.append(item.strip())

        return "\n".join(history_lines) or NO_CHAT_HISTORY

    return NO_CHAT_HISTORY


def _invoke_rag(message: str, chat_history: str, embeddings_mode: str | None = None):
    selected_mode = resolve_embeddings_mode(embeddings_mode)
    result = _build_rag_chain(selected_mode).invoke(
        {
            "input": message,
            "chat_history": chat_history or NO_CHAT_HISTORY,
        }
    )
    sources = [doc.page_content[:200] for doc in result.get("context", [])]
    return result["answer"], sources, selected_mode


class ChatDeleteRequestSerializer(serializers.Serializer):
    chat_id = serializers.IntegerField(min_value=1)


class ChatMessageRequestSerializer(serializers.Serializer):
    message = serializers.CharField(max_length=4000)
    chat_history = serializers.JSONField(required=False)


class ChatMessageResponseSerializer(serializers.Serializer):
    answer = serializers.CharField()
    sources = serializers.ListField(child=serializers.CharField())
    chat_history = serializers.CharField(required=False)


class ChatRoundTripResponseSerializer(serializers.Serializer):
    content = serializers.CharField()
    senderId = serializers.CharField()
    sendTime = serializers.DateTimeField()
    isAI = serializers.BooleanField()
    role = serializers.CharField()
    chatId = serializers.IntegerField()
    userMessage = serializers.DictField()
    messages = serializers.ListField(child=serializers.DictField())
    sources = serializers.ListField(child=serializers.CharField())


class ChatListCreateView(APIView):
    @swagger_auto_schema(
        tags=["Chats"],
        responses={200: ChatListResponseSerializer},
        security=[{"Bearer": []}],
    )
    def get(self, request):
        user, error_response = _get_authenticated_user_document(request)
        if error_response:
            return error_response

        return Response({"chats": [_serialize_chat(chat, str(user.id)) for chat in user.chats]})

    @swagger_auto_schema(
        tags=["Chats"],
        request_body=ChatCreateSerializer,
        responses={201: ChatSummarySerializer},
        security=[{"Bearer": []}],
    )
    def post(self, request):
        serializer = ChatCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        current_user, error_response = _get_authenticated_user_document(request)
        if error_response:
            return error_response

        chat = Chat(
            id=int(now_utc().timestamp() * 1000000),
            title=serializer.validated_data["title"],
            participant_a=str(current_user.id),
            participant_b=ASSISTANT_ROLE,
            messages=[],
        )
        _save_chat_for_user(current_user, chat)

        return Response(_serialize_chat(chat, str(current_user.id)), status=status.HTTP_201_CREATED)

    @swagger_auto_schema(
        tags=["Chats"],
        query_serializer=ChatDeleteRequestSerializer,
        responses={204: ""},
        security=[{"Bearer": []}],
    )
    def delete(self, request):
        serializer = ChatDeleteRequestSerializer(data=request.query_params or request.data)
        serializer.is_valid(raise_exception=True)

        current_user, error_response = _get_authenticated_user_document(request)
        if error_response:
            return error_response

        chat_id = serializer.validated_data["chat_id"]
        if not _delete_chat_for_user(current_user, chat_id):
            return Response({"detail": "Chat no encontrado"}, status=status.HTTP_404_NOT_FOUND)

        return Response(status=status.HTTP_204_NO_CONTENT)


class MessageListCreateView(APIView):
    @swagger_auto_schema(
        tags=["Messages"],
        responses={200: ChatWithMessagesSerializer},
        security=[{"Bearer": []}],
    )
    def get(self, request, chat_id: int):
        current_user, error_response = _get_authenticated_user_document(request)
        if error_response:
            return error_response

        chat = _find_chat(current_user, chat_id)
        if not chat:
            return Response({"detail": "Chat no encontrado"}, status=status.HTTP_404_NOT_FOUND)

        return Response(_serialize_chat_with_messages(chat, str(current_user.id)))

    @swagger_auto_schema(
        tags=["Messages"],
        request_body=MessageCreateSerializer,
        responses={201: ChatRoundTripResponseSerializer},
        security=[{"Bearer": []}],
    )
    def post(self, request, chat_id: int):
        serializer = MessageCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        current_user, error_response = _get_authenticated_user_document(request)
        if error_response:
            return error_response

        chat = _find_chat(current_user, chat_id)
        if not chat:
            return Response({"detail": "Chat no encontrado"}, status=status.HTTP_404_NOT_FOUND)

        user_message = Message(
            content=serializer.validated_data["content"],
            sender_id=USER_ROLE,
            send_time=now_utc(),
        )
        chat_history = _build_chat_history(chat, latest_user_message=user_message.content)

        try:
            answer, sources, embeddings_mode = _invoke_rag(user_message.content, chat_history)
        except FaissIndexNotFoundError as exc:
            logger.warning("RAG no disponible para el chat %s: %s", chat_id, exc)
            return Response(
                {
                    "error": str(exc),
                    "embeddings_mode": embeddings_mode if "embeddings_mode" in locals() else resolve_embeddings_mode(),
                    "faiss_index_path": get_faiss_index_path(resolve_embeddings_mode()),
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception as exc:
            logger.error("Error al generar respuesta RAG para el chat %s: %s", chat_id, exc, exc_info=True)
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        assistant_message = Message(
            content=answer,
            sender_id=ASSISTANT_ROLE,
            send_time=now_utc(),
        )
        chat.messages.extend([user_message, assistant_message])
        _save_chat_for_user(current_user, chat)

        response_payload = _serialize_message(assistant_message)
        response_payload["chatId"] = chat.id
        response_payload["userMessage"] = _serialize_message(user_message)
        response_payload["messages"] = [_serialize_message(message) for message in chat.messages]
        response_payload["sources"] = sources
        return Response(response_payload, status=status.HTTP_201_CREATED)

    @swagger_auto_schema(
        tags=["Chats"],
        responses={204: ""},
        security=[{"Bearer": []}],
    )
    def delete(self, request, chat_id: int):
        current_user, error_response = _get_authenticated_user_document(request)
        if error_response:
            return error_response

        if not _delete_chat_for_user(current_user, chat_id):
            return Response({"detail": "Chat no encontrado"}, status=status.HTTP_404_NOT_FOUND)

        return Response(status=status.HTTP_204_NO_CONTENT)


SYSTEM_PROMPT = (
    "Eres el Dr. Nefros, un médico especialista en nefrología y urología con amplia experiencia clínica y docente. "
    "Tu conocimiento está basado exclusivamente en la Sección 7 del Harrison: Principios de Medicina Interna, "
    "que cubre la función renal y las vías urinarias.\n\n"
    "## TU ROL COMO EDUCADOR\n"
    "- Explica los temas de forma progresiva: comienza con conceptos básicos antes de profundizar.\n"
    "- Define SIEMPRE los términos médicos o técnicos la primera vez que los uses. "
    "Ejemplo: 'la creatinina (una sustancia de desecho que filtra el riñón)...'\n"
    "- Usa analogías cotidianas para facilitar la comprensión cuando sea posible.\n"
    "- Estructura tus respuestas con claridad: usa párrafos cortos, y cuando sea útil, listas o pasos numerados.\n\n"
    "## MEMORIA DE LA CONVERSACIÓN\n"
    "{chat_history}\n\n"
    "## MANEJO DEL CONTEXTO\n"
    "- Basa tus respuestas ÚNICAMENTE en el contexto proporcionado a continuación.\n"
    "- Si la pregunta es médica pero no está cubierta en el contexto, responde: "
    "'Esa pregunta es interesante, pero no tengo información sobre ese tema en mi base de conocimientos actual. "
    "Te recomiendo consultar directamente el Harrison o un especialista.'\n"
    "- No inventes información ni extrapoles más allá de lo que indica el contexto.\n\n"
    "## TONO Y COMUNICACIÓN\n"
    "- Sé siempre amable, paciente, empático y motivador. Nunca uses un tono frío o condescendiente.\n"
    "- Si alguien comete un error conceptual, corrígelo con gentileza y sin hacerle sentir mal.\n"
    "- Si te hacen preguntas no relacionadas con medicina renal o vías urinarias, responde con amabilidad:\n"
    "  'Me especializo en temas de función renal y vías urinarias. Para esta pregunta, te sugiero "
    "consultar otra fuente o especialista. ¿Hay algo relacionado con los riñones en lo que pueda ayudarte?'\n\n"
    "## CONTEXTO DE REFERENCIA\n"
    "{context}"
)


@lru_cache(maxsize=2)
def _build_rag_chain(mode: str):
    embeddings_mode = resolve_embeddings_mode(mode)
    faiss_path = ensure_faiss_index_exists(embeddings_mode)
    embeddings = get_embeddings(embeddings_mode)

    logger.info("Inicializando RAG con EMBEDDINGS_MODE=%s", embeddings_mode)
    logger.info("Cargando índice FAISS desde: %s", faiss_path)

    vector_store = FAISS.load_local(
        faiss_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    documentos = list(vector_store.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(documentos)
    bm25_retriever.k = 3

    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5],
    )

    llm = OllamaLLM(model="llama3.2", temperature=0.2)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    logger.info("RAG inicializado correctamente.")
    return chain


@swagger_auto_schema(
    method="post",
    request_body=ChatMessageRequestSerializer,
    responses={200: ChatMessageResponseSerializer},
    operation_description="Enviar un mensaje al modelo RAG y obtener respuesta.",
)
@api_view(["POST"])
@authentication_classes([])
@permission_classes([AllowAny])
def chat_view(request):
    """
    POST /chat/message/
    Body JSON: { "message": "¿Qué es la creatinina?", "chat_history": [...] }
    Respuesta: { "answer": "...", "sources": ["..."] }
    """
    message = request.data.get("message", "").strip()
    if not message:
        return Response(
            {"error": "El mensaje no puede estar vacío."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    chat_history = _normalize_chat_history_payload(request.data.get("chat_history"))

    try:
        answer, sources, _embeddings_mode = _invoke_rag(message, chat_history)
        return Response({"answer": answer, "sources": sources, "chat_history": chat_history})

    except FaissIndexNotFoundError as exc:
        embeddings_mode = resolve_embeddings_mode()
        logger.warning("RAG no disponible: %s", exc)
        return Response(
            {
                "error": str(exc),
                "embeddings_mode": embeddings_mode,
                "faiss_index_path": get_faiss_index_path(embeddings_mode),
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    except Exception as exc:
        logger.error("Error en chat_view: %s", exc, exc_info=True)
        return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["GET"])
@authentication_classes([])
@permission_classes([AllowAny])
def health_view(request):
    """GET /chat/health/"""
    embeddings_mode = resolve_embeddings_mode()
    return Response(
        {
            "status": "ok",
            "model": "llama3.2",
            "embeddings_mode": embeddings_mode,
            "faiss_index_path": get_faiss_index_path(embeddings_mode),
            "faiss_index_ready": is_faiss_index_ready(embeddings_mode),
        }
    )