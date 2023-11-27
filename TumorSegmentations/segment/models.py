from django.db import models

# Create your models here.
class PredtictionModel(models.Model):
    image = models.URLField()

    def __str__(self):
        return str(id)