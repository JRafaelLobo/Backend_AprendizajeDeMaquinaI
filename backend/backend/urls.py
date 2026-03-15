from django.urls import include, path
from drf_yasg import openapi
from drf_yasg.views import get_schema_view
from rest_framework.permissions import AllowAny
from django.shortcuts import redirect


schema_view = get_schema_view(
    openapi.Info(
        title="Backend API",
        default_version='v1',
        description="API de autenticación y chat para el proyecto de aprendizaje de máquina",
    ),
    public=True,
    permission_classes=(AllowAny,),
)

def redirect_to_docs(request):
    return redirect("/swagger/")

urlpatterns = [
    path("", redirect_to_docs),
    path("api/auth/", include("users.urls")),
    path("api/chat/", include("chat.urls")),
    path("swagger/", schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path("redoc/", schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
]
