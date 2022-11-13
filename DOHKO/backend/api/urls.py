from django.conf import settings
from django.conf.urls.static import static

from django.urls import path
from . import views

urlpatterns = [
    path('get/', views.getProjects),
    path('post/', views.postProject),
    path('put/<int:pk>/', views.putProject),
    path('delete/<int:pk>/', views.deleteProject),
    path('getPreviewDf/<int:pk>/', views.getPreviewDataframe),
    path('eda/dataTypes/<int:pk>/', views.getDataTypes),
    path('eda/dataShape/<int:pk>/', views.getDataShape),
    path('eda/dataNull/<int:pk>/', views.getDataNull),
    path('eda/dataDescribe/<int:pk>/', views.getDataDescribe),
    path('eda/dataCorrelation/<int:pk>/', views.getDataCorrelation),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
