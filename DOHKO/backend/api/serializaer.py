from rest_framework.serializers import ModelSerializer
from .models import Project

class ProjectSerializer(ModelSerializer):
    class Meta:
        model = Project
        fields = ['id','name','desc','url','dataFile','cols','rows']

class DataSerializer(ModelSerializer):
    class Meta:
        model = Project
        fields = ['id','code','cols','rows','dataGraph']