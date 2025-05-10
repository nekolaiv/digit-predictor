from django.db import models

# Create your models here.

class Prediction(models.Model):
    digit = models.IntegerField()
    confidence = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.digit} ({self.confidence:.2f}%)"
