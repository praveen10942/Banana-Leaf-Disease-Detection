from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing, name='landing'),
    path('analyze/', views.upload, name='upload'),
    path('history/', views.history, name='history'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('about/', views.about, name='about'),
    path('model-stats/', views.model_stats, name='model_stats'),
]