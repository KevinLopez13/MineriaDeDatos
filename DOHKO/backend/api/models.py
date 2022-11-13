from django.db import models
import pandas as pd
import numpy as np
import os
from django.conf import settings

class Project(models.Model):
    name = models.CharField(max_length=20)
    desc = models.CharField(max_length=200, null=True)
    url = models.CharField(max_length=200, null=True)
    dataFile = models.FileField(upload_to='datasets/', null=True, default=None)

    def delete(self):
        os.remove(os.path.join(settings.MEDIA_ROOT, self.dataFile.name))
        return super().delete()

    #dataframe = models.ForeignKey(DataFrame, on_delete=models.CASCADE)
    #class DataFrame(models.Model):
    
    dataframe = None
    rows = None
    cols = None
    code = None

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.dataframe = pd.read_csv('C:\\Users\\kevin\\Desktop\\melb_data.csv')

    def load_data(self):
        pass

    def getPreviewDataframe(self):
        self.dataframe = pd.read_csv(self.dataFile)
        
        # Columna para numero de fila
        self.cols = np.insert( self.dataframe.columns.values, 0, '')

        rows_lst = self.dataframe.values.tolist()
        l = len(rows_lst)
        # Primeros y ultimos 5 elementos
        preview_rows = [0,1,2,3,4,l-5,l-4,l-3,l-2,l-1]
        # Conversion de elementos a str + numero de fila
        self.rows = [ [r] + list(map(lambda cell: str(cell), rows_lst[r])) for r in preview_rows]
        # Separacion
        self.rows.insert(5, ['...' for i in range(len(self.cols))])

    def data_structure_description(self):
        self.dataframe = pd.read_csv(self.dataFile)
        self.code = {
            'command':'dataframe.shape',
            'result':str(self.dataframe.shape)}
        self.cols = ['Variable','Tipo']
        self.rows = [(k,str(v)) for k,v in self.dataframe.dtypes.items()]

    def nulldata_identify(self):
        self.cols = ['Variable','Cuenta']
        self.rows = list(self.dataframe.isnull().sum().items())

    def dif_values(self):
        # Histogramas

        # Resumen estadistico
        self.cols = self.dataframe.describe().columns.tolist()
        self.rows = self.dataframe.describe().values.tolist()

        # Diagramas de bigote

        # Variables categoricas
        self.cols = self.dataframe.describe(include='object').columns.tolist()
        self.rows = self.dataframe.describe(include='object').values.tolist()

    def variable_relations(self):
        self.cols = self. self.dataframe.corr().columns.tolist()
        self.rows = self.dataframe.corr().values.tolist()



