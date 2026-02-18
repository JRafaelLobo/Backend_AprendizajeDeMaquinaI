from rest_framework import serializers


class RegisterSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(min_length=8, write_only=True)


class MessageSerializer(serializers.Serializer):
    content = serializers.CharField()
    senderId = serializers.CharField()
    sendTime = serializers.DateTimeField()


class ChatSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    title = serializers.CharField()
    participantA = serializers.CharField()
    participantB = serializers.CharField()
    messages = MessageSerializer(many=True)


class UserSerializer(serializers.Serializer):
    id = serializers.CharField()
    email = serializers.EmailField()
    chats = ChatSerializer(many=True)


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)


class RefreshSerializer(serializers.Serializer):
    refresh_token = serializers.CharField()


class AuthResponseSerializer(serializers.Serializer):
    access_token = serializers.CharField()
    refresh_token = serializers.CharField()
    user = UserSerializer()


class RefreshResponseSerializer(serializers.Serializer):
    access_token = serializers.CharField()
    refresh_token = serializers.CharField()
