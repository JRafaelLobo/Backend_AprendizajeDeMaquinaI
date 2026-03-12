from datetime import datetime, timezone
from functools import lru_cache

from bson import ObjectId
from bson.errors import InvalidId
from drf_yasg.utils import swagger_auto_schema
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

import logging

from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework import serializers

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from .embeddings import (
    FaissIndexNotFoundError,
    ensure_faiss_index_exists,
    get_embeddings,
    get_faiss_index_path,
    is_faiss_index_ready,
    resolve_embeddings_mode,
)

logger = logging.getLogger(__name__)

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
    

# ══════════════════════════════════════════════
# PARÁMETROS
# ══════════════════════════════════════════════

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


# ══════════════════════════════════════════════
# RAG — se inicializa de forma diferida y se reutiliza luego
# ══════════════════════════════════════════════

@lru_cache(maxsize=2)
def _build_rag_chain(mode: str):
    embeddings_mode = resolve_embeddings_mode(mode)
    faiss_path = ensure_faiss_index_exists(embeddings_mode)
    embeddings = get_embeddings(embeddings_mode)

    logger.info("⏳ Inicializando RAG con EMBEDDINGS_MODE=%s", embeddings_mode)
    logger.info("⏳ Cargando índice FAISS desde: %s", faiss_path)

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
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    logger.info("✅ RAG inicializado correctamente.")
    return chain


# ══════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════

class ChatMessageRequestSerializer(serializers.Serializer):
    message = serializers.CharField(max_length=4000)

class ChatMessageResponseSerializer(serializers.Serializer):
    answer = serializers.CharField()
    sources = serializers.ListField(child=serializers.CharField())

@swagger_auto_schema(
    method="post",
    request_body=ChatMessageRequestSerializer,
    responses={200: ChatMessageResponseSerializer},
    operation_description="Enviar un mensaje al modelo RAG y obtener respuesta."
)
@api_view(["POST"])
@authentication_classes([])
@permission_classes([AllowAny])
def chat_view(request):
    """
    POST /chat/message/
    Body JSON:  { "message": "¿Qué es la creatinina?" }
    Respuesta:  { "answer": "...", "sources": ["..."] }
    """
    message = request.data.get("message", "").strip()
    if not message:
        return Response(
            {"error": "El mensaje no puede estar vacío."},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        embeddings_mode = resolve_embeddings_mode()
        result = _build_rag_chain(embeddings_mode).invoke({"input": message})
        sources = [doc.page_content[:200] for doc in result.get("context", [])]
        return Response({"answer": result["answer"], "sources": sources})

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

    except Exception as e:
        logger.error(f"Error en chat_view: {e}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


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