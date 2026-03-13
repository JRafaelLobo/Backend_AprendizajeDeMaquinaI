from rest_framework import serializers

from users.serializers import MessageSerializer


FORBIDDEN_IDENTITY_FIELDS = {
    "email",
    "ownerId",
    "participantA",
    "participantB",
    "participant_email",
    "user",
    "userId",
    "user_id",
}


class ChatCreateSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=200)

    def validate(self, attrs):
        forbidden_fields = sorted(set(self.initial_data).intersection(FORBIDDEN_IDENTITY_FIELDS))
        if forbidden_fields:
            fields = ", ".join(forbidden_fields)
            raise serializers.ValidationError(
                f"No envíes identidad del usuario en el body ({fields}). El backend toma el usuario desde Authorization: Bearer <token>."
            )
        return attrs


class ChatSummarySerializer(serializers.Serializer):
    id = serializers.IntegerField()
    title = serializers.CharField()
    ownerId = serializers.CharField()
    participantA = serializers.CharField()
    participantB = serializers.CharField()


class ChatListResponseSerializer(serializers.Serializer):
    chats = ChatSummarySerializer(many=True)


class MessageCreateSerializer(serializers.Serializer):
    content = serializers.CharField(max_length=4000)

    def validate(self, attrs):
        forbidden_fields = sorted(set(self.initial_data).intersection(FORBIDDEN_IDENTITY_FIELDS))
        if forbidden_fields:
            fields = ", ".join(forbidden_fields)
            raise serializers.ValidationError(
                f"No envíes identidad del usuario en el body ({fields}). El backend toma el usuario desde Authorization: Bearer <token>."
            )
        return attrs


class ChatWithMessagesSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    title = serializers.CharField()
    ownerId = serializers.CharField()
    participantA = serializers.CharField()
    participantB = serializers.CharField()
    messages = MessageSerializer(many=True)
