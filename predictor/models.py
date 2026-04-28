from django.db import models
from django.contrib.auth.models import User


class Prediction(models.Model):
    user       = models.ForeignKey(User, on_delete=models.CASCADE)
    image      = models.ImageField(upload_to='uploads/')
    disease    = models.CharField(max_length=100)
    confidence = models.FloatField()
    is_healthy = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} — {self.disease} ({self.confidence}%)"
