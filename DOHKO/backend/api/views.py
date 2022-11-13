from rest_framework.response import Response
from rest_framework.serializers import Serializer
from rest_framework.decorators import api_view

from .models import Project
from .serializaer import ProjectSerializer, DataSerializer


@api_view(['GET'])
def getPreviewDataframe(request, pk):
    """Función que retorna los primeros registros del conjunto de datos."""
    project = Project.objects.get(id=pk)
    project.getPreviewDataframe()
    serializer = ProjectSerializer(project, many=False)
    return Response(serializer.data)

@api_view(['GET'])
def getDataTypes(request, pk):
    project = Project.objects.get(id=pk)
    project.dataTypes()
    serializer = DataSerializer(project, many=False)
    return Response(serializer.data)

@api_view(['GET'])
def getDataShape(request, pk):
    project = Project.objects.get(id=pk)
    project.dataShape()
    serializer = DataSerializer(project, many=False)
    return Response(serializer.data)

@api_view(['GET'])
def getDataNull(request, pk):
    project = Project.objects.get(id=pk)
    project.dataNull()
    serializer = DataSerializer(project, many=False)
    return Response(serializer.data)

@api_view(['GET'])
def getDataDescribe(request, pk):
    project = Project.objects.get(id=pk)
    project.dataDescribe()
    serializer = DataSerializer(project, many=False)
    return Response(serializer.data)

@api_view(['GET'])
def getDataCorrelation(request, pk):
    project = Project.objects.get(id=pk)
    project.dataCorrelation()
    serializer = DataSerializer(project, many=False)
    return Response(serializer.data)

# Projects
@api_view(['GET'])
def getProjects(request):
    project = Project.objects.all()
    serializer = ProjectSerializer(project, many=True)
    return Response(serializer.data)


@api_view(['POST'])
def postProject(request):
    data = request.data
    project = Project.objects.create(
        name=data['name'],
        url=data['url'],
        dataFile=data.get('dataFile',None),
        desc=data['desc']
    )
    serializer = ProjectSerializer(project, many=False)
    return Response(serializer.data)

@api_view(['PUT'])
def putProject(request, pk):
    data = request.data
    project = Project.objects.get(id=pk)
    serializer = ProjectSerializer(instance=project, data=data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)

@api_view(['DELETE'])
def deleteProject(request, pk):
    project = Project.objects.get(id=pk)
    project.delete()
    return Response('Projecto Eliminado')