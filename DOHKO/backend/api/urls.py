from django.conf import settings
from django.conf.urls.static import static

from django.urls import path, re_path
from . import views

urlpatterns = [
    path('get/', views.getProjects),
    path('post/', views.postProject),
    path('put/<int:pk>/', views.putProject),
    path('delete/<int:pk>/', views.deleteProject),

    re_path(r'(\S+/(?P<pk>[0-9]+)/$)', views.getPost),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
