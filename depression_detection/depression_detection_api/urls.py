from django.urls import path, include
from .views import (
    DepressionDetectionApiView,
)

urlpatterns = [
    path('detect-depression', DepressionDetectionApiView.as_view()),
]
