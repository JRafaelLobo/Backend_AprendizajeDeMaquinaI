from django.conf import settings
from mongoengine import connect
from mongoengine.connection import get_connection


def connect_mongo():
    try:
        get_connection(alias="default")
        return
    except Exception:
        pass

    connect_kwargs = {
        "host": settings.MONGO_URI,
        "alias": "default",
    }

    if getattr(settings, "MONGO_USE_MOCK", False):
        import mongomock

        connect_kwargs["mongo_client_class"] = mongomock.MongoClient

    connect(**connect_kwargs)
