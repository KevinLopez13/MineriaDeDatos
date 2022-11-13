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
    

    dataframe = None
    rows = None
    cols = None
    code = None
    dataGraph = None

    def delete(self):
        os.remove(os.path.join(settings.MEDIA_ROOT, self.dataFile.name))
        return super().delete()

    def load_data(self):
        if not self.dataframe:
            self.dataframe = pd.read_csv(self.dataFile or self.url)

    def getPreviewDataframe(self):
        # self.dataframe = pd.read_csv(self.dataFile)
        self.load_data()
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

    
    def dataShape(self):
        """Dimesión del dataframe."""
        self.load_data()
        self.code = {
            'command':'dataframe.shape',
            'result':str(self.dataframe.shape)}

    def dataTypes(self):
        """Tipo de datos del dataframe."""
        self.load_data()
        self.cols = ['Variable','Tipo']
        self.rows = [(k,str(v)) for k,v in self.dataframe.dtypes.items()]

    def dataNull(self):
        """Datos nulos del dataframe."""
        self.load_data()
        self.cols = ['Variable','Cuenta']
        self.rows = list(self.dataframe.isnull().sum().items())

    def dataDescribe(self):
        """Resumen estadístico de variables numéricas."""
        self.load_data()
        self.cols = self.dataframe.describe().columns.tolist()
        self.cols.insert(0,'')
        measures = ['Cuenta', 'Media', 'Std.', 'Min','25%','50%','75%','Max']
        self.rows = self.dataframe.describe().values.tolist()

        for row, measure in zip(self.rows, measures):
            row.insert(0, measure)

    def dataCorrelation(self):
        self.load_data()
        self.cols = self.dataframe.corr().columns.tolist()
        self.rows = self.dataframe.corr().values.tolist()
        rows_lst = self.dataframe.corr().values
        self.rows = [ list(map(lambda cell: str(cell) if pd.isna(cell) else cell, row)) for row in rows_lst]

        data = []
        for col, row in zip(self.cols,rows_lst):
            data.append(
                {
                    'id':col,
                    'data':[{'x':x, 'y':None if pd.isna(y) else y} for x,y in zip(self.cols,row)]
                }
            )

        self.dataGraph = data

    def dif_values(self):
        # Histogramas

        # Diagramas de bigote

        # Variables categoricas
        self.cols = self.dataframe.describe(include='object').columns.tolist()
        self.rows = self.dataframe.describe(include='object').values.tolist()



