from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing, name='landing'),
    path('analyze/', views.upload, name='upload'),
    path('history/', views.history, name='history'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('about/', views.about, name='about'),
    path('model-stats/', views.model_stats, name='model_stats'),
    path('privacy/', views.privacy_policy, name='privacy_policy'),
    path('terms/', views.terms_conditions, name='terms_conditions'),
    path('cookies/', views.cookie_policy, name='cookie_policy'),
    path('disclaimer/', views.disclaimer, name='disclaimer'),
    path('api/outbreaks/', views.get_outbreaks, name='get_outbreaks'),
]