from django.contrib import admin
from .models import Prediction

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display  = ('user', 'disease', 'confidence', 'is_healthy', 'created_at')
    list_filter   = ('is_healthy', 'disease')
    search_fields = ('user__username', 'disease')
