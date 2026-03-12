import os
from unittest.mock import Mock, call, patch

from rest_framework import status
from rest_framework.test import APITestCase

from chat.embeddings import (
    FaissIndexNotFoundError,
    LazyEmbeddings,
    ensure_faiss_index_exists,
    get_faiss_index_file_path,
    get_faiss_index_path,
    resolve_embeddings_mode,
)
from users.models import User
from chat import views


class ChatEndpointTests(APITestCase):
    def setUp(self):
        User.objects.delete()
        register_a = self.client.post(
            "/auth/register",
            {"email": "chat@example.com", "password": "Password123!"},
            format="json",
        )
        register_b = self.client.post(
            "/auth/register",
            {"email": "chat2@example.com", "password": "Password123!"},
            format="json",
        )
        self.access_token_a = register_a.data["access_token"]
        self.access_token_b = register_b.data["access_token"]

    def test_send_and_history(self):
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {self.access_token_a}")

        create_chat_response = self.client.post(
            "/chat/",
            {"participant_email": "chat2@example.com", "title": "Chat de prueba"},
            format="json",
        )
        self.assertEqual(create_chat_response.status_code, status.HTTP_201_CREATED)
        chat_id = create_chat_response.data["id"]

        send_response = self.client.post(f"/chat/{chat_id}/messages", {"content": "Hola"}, format="json")
        self.assertEqual(send_response.status_code, status.HTTP_201_CREATED)
        self.assertIn("senderId", send_response.data)

        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {self.access_token_b}")
        send_response_b = self.client.post(f"/chat/{chat_id}/messages", {"content": "Que tal"}, format="json")
        self.assertEqual(send_response_b.status_code, status.HTTP_201_CREATED)

        history_response = self.client.get(f"/chat/{chat_id}/messages")
        self.assertEqual(history_response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(history_response.data["messages"]), 2)
        self.assertEqual(history_response.data["messages"][0]["content"], "Hola")


class EmbeddingsConfigurationTests(APITestCase):
    def setUp(self):
        views._build_rag_chain.cache_clear()
        self.addCleanup(views._build_rag_chain.cache_clear)

    def test_default_embeddings_mode_is_full(self):
        original_value = os.environ.pop("EMBEDDINGS_MODE", None)
        try:
            self.assertEqual(resolve_embeddings_mode(), "full")
        finally:
            if original_value is not None:
                os.environ["EMBEDDINGS_MODE"] = original_value

    def test_lite_mode_uses_dedicated_faiss_index(self):
        self.assertTrue(get_faiss_index_path("full").endswith("faiss_index_renal"))
        self.assertTrue(get_faiss_index_path("lite").endswith("faiss_index_renal_lite"))

    def test_lazy_embeddings_build_selected_backend_only_on_first_use(self):
        backend = Mock()
        backend.embed_query.return_value = [0.1, 0.2]
        builder = Mock(return_value=backend)

        embeddings = LazyEmbeddings(mode="lite", builders={"lite": builder, "full": Mock()})

        builder.assert_not_called()
        self.assertEqual(embeddings.embed_query("creatinina"), [0.1, 0.2])
        builder.assert_called_once_with()
        backend.embed_query.assert_called_once_with("creatinina")

    def test_lazy_embeddings_without_fixed_mode_follows_active_mode(self):
        full_backend = Mock()
        full_backend.embed_query.return_value = [1.0]
        lite_backend = Mock()
        lite_backend.embed_query.return_value = [2.0]
        full_builder = Mock(return_value=full_backend)
        lite_builder = Mock(return_value=lite_backend)

        embeddings = LazyEmbeddings(builders={"full": full_builder, "lite": lite_builder})

        with patch.dict(os.environ, {"EMBEDDINGS_MODE": "full"}, clear=False):
            self.assertEqual(embeddings.embed_query("urea"), [1.0])

        with patch.dict(os.environ, {"EMBEDDINGS_MODE": "lite"}, clear=False):
            self.assertEqual(embeddings.embed_query("creatinina"), [2.0])

        with patch.dict(os.environ, {"EMBEDDINGS_MODE": "full"}, clear=False):
            self.assertEqual(embeddings.embed_query("filtracion"), [1.0])

        full_builder.assert_called_once_with()
        lite_builder.assert_called_once_with()

    @patch("chat.views.create_retrieval_chain")
    @patch("chat.views.create_stuff_documents_chain")
    @patch("chat.views.ChatPromptTemplate.from_messages")
    @patch("chat.views.OllamaLLM")
    @patch("chat.views.EnsembleRetriever")
    @patch("chat.views.BM25Retriever.from_documents")
    @patch("chat.views.FAISS.load_local")
    @patch("chat.views.get_embeddings")
    @patch("chat.views.ensure_faiss_index_exists")
    def test_rag_chain_cache_is_isolated_per_embeddings_mode(
        self,
        mock_ensure_index,
        mock_get_embeddings,
        mock_load_local,
        mock_bm25_from_documents,
        mock_ensemble_retriever,
        _mock_ollama,
        _mock_prompt_from_messages,
        _mock_create_stuff_documents_chain,
        mock_create_retrieval_chain,
    ):
        mock_ensure_index.side_effect = lambda mode: f"/tmp/{mode}"
        mock_get_embeddings.side_effect = lambda mode: f"embeddings::{mode}"

        vector_store = Mock()
        vector_store.docstore._dict = {"doc": Mock()}
        vector_store.as_retriever.return_value = Mock()
        mock_load_local.return_value = vector_store

        bm25_retriever = Mock()
        mock_bm25_from_documents.return_value = bm25_retriever
        mock_ensemble_retriever.return_value = Mock()

        full_chain = Mock(name="full_chain")
        lite_chain = Mock(name="lite_chain")
        mock_create_retrieval_chain.side_effect = [full_chain, lite_chain]

        first_full = views._build_rag_chain("full")
        second_full = views._build_rag_chain("full")
        first_lite = views._build_rag_chain("lite")

        self.assertIs(first_full, second_full)
        self.assertIsNot(first_full, first_lite)
        self.assertEqual(mock_get_embeddings.call_args_list, [call("full"), call("lite")])
        self.assertEqual(
            mock_load_local.call_args_list,
            [
                call("/tmp/full", "embeddings::full", allow_dangerous_deserialization=True),
                call("/tmp/lite", "embeddings::lite", allow_dangerous_deserialization=True),
            ],
        )

    def test_invalid_embeddings_mode_raises_error(self):
        with self.assertRaises(ValueError):
            resolve_embeddings_mode("otro")

    def test_get_faiss_index_file_path_appends_index_filename(self):
        self.assertTrue(get_faiss_index_file_path("full").endswith("faiss_index_renal/index.faiss"))
        self.assertTrue(get_faiss_index_file_path("lite").endswith("faiss_index_renal_lite/index.faiss"))

    @patch("chat.embeddings.Path.exists", return_value=False)
    def test_missing_faiss_index_raises_helpful_error(self, _mock_exists):
        with self.assertRaises(FaissIndexNotFoundError) as exc_context:
            ensure_faiss_index_exists("lite")

        self.assertIn("EMBEDDINGS_MODE=lite", str(exc_context.exception))
        self.assertIn("faiss_index_renal_lite/index.faiss", str(exc_context.exception))
