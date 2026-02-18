from rest_framework import serializers


class RatingSerializer(serializers.Serializer):
    score = serializers.IntegerField(min_value=1, max_value=5)
