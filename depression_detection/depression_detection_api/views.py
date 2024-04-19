from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from depression_detection_api.processing import DataPreprocessor, Predict


class DepressionDetectionApiView(APIView):

    def get(self, *args, **kwargs):
        return Response("Depression detection API", status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        message = request.data.get('message')

        if not message:
            return Response("No Message Provided", status=status.HTTP_400_BAD_REQUEST)

        preprocessor = DataPreprocessor()

        message_with_sentiment = preprocessor.get_sentiment(message)

        predictor = Predict(message_with_sentiment)
        prediction = predictor.make_prediction()

        message_with_sentiment = message_with_sentiment.drop('Message Size', axis=1)

        response = {
            "is_depressed": prediction,
            "sentiment": message_with_sentiment
        }

        return Response(response, status=status.HTTP_200_OK)
