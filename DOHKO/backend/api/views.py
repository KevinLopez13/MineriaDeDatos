from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Project
from .serializaer import DataSerializer, ProjectSerializer
import re


@api_view(['GET', 'POST'])
def getPost(request, pk):
    """Funcion que ejecuta la funcion correspondiente al url de cada operación."""
    data = request.data

    project = Project.objects.get(id=pk)
    
    kargs = {}
    args = data.get('kgs', None)
    if args:
        args = args.split(',')
        for arg in args:
            # g = re.search('=(?!=)',c)
            # span = (s,e)
            # cons = arg[:s]
            # param = arg[e:]
            param = arg.split('=')
            kargs[param[0].strip()] = param[1].strip()
    
    kvars = data.get('vars')

    if kvars:
        kargs.update({'vars':kvars})

    fun = request.get_full_path().split('/')[2]
    f = getattr(project, fun)
    f(**kargs)
    
    serializer = DataSerializer(project, many=False)
    return Response(serializer.data)


# Proyectos
@api_view(['GET'])
def getProjects(request):
    """Recupera todos los proyectos existentes."""
    project = Project.objects.all()
    serializer = ProjectSerializer(project, many=True)
    return Response(serializer.data)


@api_view(['POST'])
def postProject(request):
    """Crea un nuevo proyecto."""
    data = request.data

    name = data['name']
    url = data.get('url',None)
    dataFile = data.get('dataFile',None)
    desc = data.get('desc',None)
    fileName = "Archivo: "+str(dataFile) if dataFile else "URL: "+re.search('(\w+.csv)', url, re.M).group(0)

    project = Project.objects.create(
        name=name, url=url, dataFile=dataFile, desc=desc, fileName=fileName
        )
    serializer = ProjectSerializer(project, many=False)
    return Response(serializer.data)

@api_view(['PUT'])
def putProject(request, pk):
    """Actualiza un proyecto."""
    data = request.data
    project = Project.objects.get(id=pk)
    serializer = ProjectSerializer(instance=project, data=data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)

@api_view(['DELETE'])
def deleteProject(request, pk):
    """Elimina un proyecto."""
    project = Project.objects.get(id=pk)
    project.delete()
    return Response('Projecto Eliminado')