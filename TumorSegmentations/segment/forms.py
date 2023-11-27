from django import forms
from django.forms import ModelForm
from .models import PredtictionModel

class PredictForm(forms.ModelForm):
    class Meta:
        model = PredtictionModel
        fields = "__all__"