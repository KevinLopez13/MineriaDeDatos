from django.conf import settings
from django.conf.urls.static import static

from django.urls import path
from . import views

urlpatterns = [
    path('get/', views.getProjects),
    path('post/', views.postProject),
    path('put/<int:pk>/', views.putProject),
    path('delete/<int:pk>/', views.deleteProject),

    path('eda/dataPreview/<int:pk>/', views.eda),
    path('eda/dataTypes/<int:pk>/', views.eda),
    path('eda/dataShape/<int:pk>/', views.eda),
    path('eda/dataNull/<int:pk>/', views.eda),
    path('eda/dataDescribe/<int:pk>/', views.eda),
    path('eda/dataCorrelation/<int:pk>/', views.eda),
    path('eda/dataHistogram/<int:pk>/', views.eda),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
