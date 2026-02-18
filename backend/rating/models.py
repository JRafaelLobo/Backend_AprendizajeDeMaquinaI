from mongoengine import DateTimeField, Document, IntField, StringField


class Rating(Document):
    user_id = StringField(required=True)
    score = IntField(required=True, min_value=1, max_value=5)
    created_at = DateTimeField(required=True)

    meta = {
        "collection": "ratings",
        "indexes": ["user_id", "-created_at"],
    }
