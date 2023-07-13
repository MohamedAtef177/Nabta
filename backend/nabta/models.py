from django.db import models
import time
from graduation_project.settings import MEDIA_ROOT

# Create your models here.


# class Photo(models.Model):
#     photo = models.ImageField(upload_to=MEDIA_URL, name=time.strftime("%Y%m%d-%H%M%S"))

LOCAL_LINK = 'http://192.168.43.254:8000/nabta/'

class Plant(models.Model):
    name = models.CharField(max_length=100)
    category = models.TextField()
    photos_str = models.TextField()
    description = models.TextField()
    temperature = models.CharField(max_length=100)
    care = models.CharField(max_length=100)
    sunlight = models.CharField(max_length=100)
    watering = models.CharField(max_length=100)
    pests = models.TextField(null=True)
    diseases = models.TextField(null=True)
    soil_type = models.TextField(null=True)
    soil_drainage = models.TextField(null=True)
    uses = models.TextField(null=True)
    humidity = models.TextField(null=True)
    fertilizing = models.TextField(null=True)

    def photos(self):
        photos_list = []
        for i in self.photos_str.split(','):
            photos_list.append(LOCAL_LINK + i)
            
        return  photos_list

    def photo(self):
        return  LOCAL_LINK + self.photos_str.split(',')[0]
