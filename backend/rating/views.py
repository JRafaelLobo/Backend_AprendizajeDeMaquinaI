from datetime import datetime, timezone

from rest_framework.response import Response
from rest_framework.views import APIView

from rating.models import Rating
from rating.serializers import RatingSerializer


class RatingCreateView(APIView):
    def post(self, request):
        serializer = RatingSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        rating = Rating(
            user_id=request.user.id,
            score=serializer.validated_data["score"],
            created_at=datetime.now(timezone.utc),
        )
        rating.save()

        return Response(
            {
                "id": str(rating.id),
                "user_id": rating.user_id,
                "score": rating.score,
                "created_at": rating.created_at.isoformat(),
            }
        )


class MyRatingsView(APIView):
    def get(self, request):
        ratings = Rating.objects(user_id=request.user.id).order_by("-created_at")
        return Response(
            {
                "ratings": [
                    {
                        "id": str(item.id),
                        "score": item.score,
                        "created_at": item.created_at.isoformat(),
                    }
                    for item in ratings
                ]
            }
        )
