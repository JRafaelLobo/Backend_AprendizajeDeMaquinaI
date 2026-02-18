from rest_framework import status
from rest_framework.test import APITestCase

from users.models import User


class AuthFlowTests(APITestCase):
    def setUp(self):
        User.objects.delete()

    def test_register(self):
        payload = {
            "email": "test@example.com",
            "password": "Password123!",
        }
        response = self.client.post("/auth/register", payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn("access_token", response.data)
        self.assertIn("refresh_token", response.data)
        self.assertEqual(User.objects(email="test@example.com").count(), 1)

    def test_login(self):
        self.client.post(
            "/auth/register",
            {"email": "login@example.com", "password": "Password123!"},
            format="json",
        )

        response = self.client.post(
            "/auth/login",
            {"email": "login@example.com", "password": "Password123!"},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("access_token", response.data)

    def test_refresh_and_jwt_validation(self):
        register_response = self.client.post(
            "/auth/register",
            {"email": "jwt@example.com", "password": "Password123!"},
            format="json",
        )
        refresh_token = register_response.data["refresh_token"]

        refresh_response = self.client.post(
            "/auth/refresh",
            {"refresh_token": refresh_token},
            format="json",
        )
        self.assertEqual(refresh_response.status_code, status.HTTP_200_OK)
        access_token = refresh_response.data["access_token"]

        unauthorized = self.client.get("/chat/")
        self.assertEqual(unauthorized.status_code, status.HTTP_401_UNAUTHORIZED)

        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {access_token}")
        authorized = self.client.get("/chat/")
        self.assertEqual(authorized.status_code, status.HTTP_200_OK)
