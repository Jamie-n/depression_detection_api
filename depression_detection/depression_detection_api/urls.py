from django.urls import path, include
from .views import (
    TodoListApiView,
)

urlpatterns = [
    path('detect', TodoListApiView.as_view()),
]