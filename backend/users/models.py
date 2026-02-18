from django.contrib.auth.hashers import check_password, make_password
from mongoengine import DateTimeField, Document, EmailField, EmbeddedDocument
from mongoengine import EmbeddedDocumentListField, IntField, StringField


class Message(EmbeddedDocument):
    content = StringField(required=True)
    sender_id = StringField(required=True, db_field="senderId")
    send_time = DateTimeField(required=True, db_field="sendTime")


class Chat(EmbeddedDocument):
    id = IntField(required=True)
    title = StringField(required=True, max_length=200)
    participant_a = StringField(required=True, db_field="participantA")
    participant_b = StringField(required=True, db_field="participantB")
    messages = EmbeddedDocumentListField(Message, default=list)


class User(Document):
    email = EmailField(required=True, unique=True)
    password = StringField(required=True)
    chats = EmbeddedDocumentListField(Chat, default=list)
    created_at = DateTimeField(required=True)
    updated_at = DateTimeField(required=True)

    meta = {
        "collection": "users",
        "indexes": ["email"],
    }

    def set_password(self, raw_password: str):
        self.password = make_password(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password(raw_password, self.password)
