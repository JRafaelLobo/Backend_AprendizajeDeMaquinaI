from rest_framework import serializers

from users.serializers import MessageSerializer


class ChatCreateSerializer(serializers.Serializer):
    participant_email = serializers.EmailField()
    title = serializers.CharField(max_length=200)


class ChatSummarySerializer(serializers.Serializer):
    id = serializers.IntegerField()
    title = serializers.CharField()
    participantA = serializers.CharField()
    participantB = serializers.CharField()


class ChatListResponseSerializer(serializers.Serializer):
    chats = ChatSummarySerializer(many=True)


class MessageCreateSerializer(serializers.Serializer):
    content = serializers.CharField(max_length=4000)


class ChatWithMessagesSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    title = serializers.CharField()
    participantA = serializers.CharField()
    participantB = serializers.CharField()
    messages = MessageSerializer(many=True)
