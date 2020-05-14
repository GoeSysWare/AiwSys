from django.urls import path
from django.conf.urls import url
from myApp.consumer import PushConsumer

websocket_urlpatterns = [
    path('image/', PushConsumer),
]