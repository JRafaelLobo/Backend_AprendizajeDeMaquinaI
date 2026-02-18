from django.urls import path

from chat.views import ChatListCreateView, MessageListCreateView

urlpatterns = [
    path("", ChatListCreateView.as_view(), name="chat-list-create"),
    path("<int:chat_id>/messages", MessageListCreateView.as_view(), name="message-list-create"),
]
