from rest_framework.serializers import ModelSerializer
from .models import Project

class ProjectSerializer(ModelSerializer):
    class Meta:
        model = Project
        fields = ['id','name','desc','url','dataFile','fileName']

class DataSerializer(ModelSerializer):
    class Meta:
        model = Project
        fields = ['id',
                    'resType',
                    'code',
                    'cols',
                    'rows',
                    'dataGraph',
                    'complement',
                    'command',
                    'kargs',
                    'vars',
                    'checkBoxType',
                    'multiline',
                    'default_args',
                    'images']