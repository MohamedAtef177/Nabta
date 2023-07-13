from statistics import mode
from rest_framework import serializers

from .models import *


class  PlantSerializer(serializers.ModelSerializer):
    class Meta:
        model = Plant
        fields = ('id', 'name', 'category', 'photos', 
        'description', 'temperature', 'care', 'sunlight', 
        'watering', 'pests', 'diseases', 'soil_type', 
        'soil_drainage', 'uses', 'humidity', 'fertilizing')


class  PlantSearchSerializer(serializers.ModelSerializer):
    class Meta:
        model = Plant
        fields = ('id', 'name', 'category', 'photo')