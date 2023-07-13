import json
import time
from django.forms.models import model_to_dict
from rest_framework import viewsets, generics, filters
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django_filters.rest_framework import DjangoFilterBackend

from django.core import serializers
from ml_models.predict import DISEASES_TYPES_NAMES, PLANTS_TYPES_NAMES, identify_disease, identify_plant, identify_ripeness
from nabta.models import Plant
from nabta.serializers import PlantSearchSerializer, PlantSerializer

# Create your views here.


def save_photo(photo):
    # name = time.strftime("%Y%m%d-%H%M%S")
    file = default_storage.save(photo.name, photo)
    return default_storage.path(file)


class IdentifyPlant(generics.ListAPIView):
    queryset = Plant.objects.all()
    serializer_class = PlantSerializer

    def post(self, request, *args, **kwargs):
        # try:
        result = identify_plant(save_photo(request.FILES["photo"]))
        plant = self.queryset.get(id__exact=PLANTS_TYPES_NAMES[result][1])
        return Response(self.serializer_class(plant).data)
        # except:
        #     return Response({
        #     })


class IdentifyDiagnose(generics.ListAPIView):
    queryset = Plant.objects.all()
    serializer_class = PlantSerializer

    def post(self, request, *args, **kwargs):
        try:
            result = identify_disease(save_photo(request.FILES["photo"]))
            print(result[0])
            print(result[1])
            print(PLANTS_TYPES_NAMES[result[0]])
            print(PLANTS_TYPES_NAMES[result[0]][1])
            return Response({
                'disease': DISEASES_TYPES_NAMES[result[0]][result[1]],
                'plant': self.serializer_class(self.queryset.get(id__exact=PLANTS_TYPES_NAMES[result[0]][1])).data,
            })
        except:
            return Response({
                
            })


class IdentifyRipeness(APIView):
    def post(self, request):
        try:
            result = identify_ripeness(save_photo(request.FILES["photo"]))
            return Response({
                'plant_type': PLANTS_TYPES_NAMES[result[0]],
                'disease_type': DISEASES_TYPES_NAMES[result[0]][result[1]]
            })
        except:
            return Response({
                
            })


class Search(generics.ListAPIView):
    queryset = Plant.objects.all()
    serializer_class = PlantSearchSerializer
    filter_backends = [filters.SearchFilter]
    search_fields = ['name', 'category']

    def get_queryset(self):
        name = self.request.query_params.get('name')
        category = self.request.query_params.get('category')
        print(name)
        print(category)
        try:
            if(category is None or category == ''):
                return self.queryset.filter(name__icontains=name)
            if(name is None or category == ''):
                return self.queryset.filter(category__iexact=category)
            else:
                return self.queryset.filter(name__icontains=name, category__iexact=category)
        except:
            return []


class GetPlant(generics.ListAPIView):
    queryset = Plant.objects.all()
    serializer_class = PlantSerializer

    def get_queryset(self):
        try:
            id = self.request.query_params.get('id')
            return self.queryset.filter(id__exact=id,)
        except:
            return []