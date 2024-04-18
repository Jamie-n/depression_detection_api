from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from depression_detection_api.processing import DataPreprocessor, Predict


class TodoListApiView(APIView):

    def post(self, request, *args, **kwargs):
        message = request.data.get('body')

        preprocessor = DataPreprocessor()

        message_with_sentiment = preprocessor.get_sentiment(message)

        predictor = Predict(message_with_sentiment)
        prediction = predictor.make_prediction()

        return Response(prediction, status=status.HTTP_200_OK)
