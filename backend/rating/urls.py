from django.urls import path

from rating.views import MyRatingsView, RatingCreateView

urlpatterns = [
    path("", RatingCreateView.as_view(), name="rating-create"),
    path("me", MyRatingsView.as_view(), name="rating-me"),
]
