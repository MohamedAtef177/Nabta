from django.conf.urls.static import static
from django.conf import settings
from django.urls import path
from . import views

urlpatterns = [
    path("identify/plant", views.IdentifyPlant().as_view(), name="identify_plant"),
    path("identify/diagnose", views.IdentifyDiagnose().as_view(), name="identify_diagnose"),
    path("identify/ripeness", views.IdentifyRipeness().as_view(), name="identify_ripeness"),
    path("search", views.Search().as_view(), name="search"),
    path("get_plant", views.GetPlant().as_view(), name="get_plant"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)