from rest_framework import status
from rest_framework.test import APITestCase

from users.models import User


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

        create_chat_response = self.client.post("/chat/", {"participant_email": "chat2@example.com"}, format="json")
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
