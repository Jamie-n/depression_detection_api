from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


class TodoListApiView(APIView):

    def post(self, request, *args, **kwargs):
        message = request.data.get('body')

        return Response(message, status=status.HTTP_200_OK)
