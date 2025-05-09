# predict/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('home/', views.home, name='home'),
    path('predict/ajax/', views.predict_digit, name='predict_digit'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
]
