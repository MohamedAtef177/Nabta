from django.contrib import admin

from nabta.models import Plant

# Register your models here.

class PlantAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'category', 'photos_str', 'temperature', 'care', 'sunlight', 'watering', 'pests', 'diseases']
    list_filter = ['id']
    search_fields = ['name', 'category']
    

admin.site.register(Plant, PlantAdmin)