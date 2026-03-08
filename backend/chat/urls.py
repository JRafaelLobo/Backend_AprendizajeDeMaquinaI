from django.urls import path
from . import views

from chat.views import ChatListCreateView, MessageListCreateView

urlpatterns = [
    path("", ChatListCreateView.as_view(), name="chat-list-create"),
    path("<int:chat_id>/messages", MessageListCreateView.as_view(), name="message-list-create"),
    path("message/", views.chat_view, name="chat-message"),
    path("health/", views.health_view, name="chat-health"),

]
