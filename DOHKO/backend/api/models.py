from django.db import models
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import os
from django.conf import settings


class Project(models.Model):
    name = models.CharField(max_length=20)
    desc = models.CharField(max_length=200, null=True)
    url = models.CharField(max_length=200, null=True)
    dataFile = models.FileField(upload_to='datasets/', null=True, default=None)
    fileName = models.CharField(max_length=100, null=True)

    dataframe = None
    rows = None
    cols = None
    code = None
    dataGraph = None
    resType = None

    def delete(self):
        if self.dataFile:
            os.remove(os.path.join(settings.MEDIA_ROOT, self.dataFile.name))
        return super().delete()

    def load_data(self):
        self.dataframe = pd.read_csv(self.dataFile or self.url)


    def dataPreview(self):
        self.load_data()
        self.resType = 'table'

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
        self.resType = 'text'
        self.code = {
            'command':'dataframe.shape',
            'result':str(self.dataframe.shape)
            }

    def dataTypes(self):
        """Tipo de datos del dataframe."""
        self.load_data()
        self.resType = 'table'
        self.cols = ['Variable','Tipo']
        self.rows = [(k,str(v)) for k,v in self.dataframe.dtypes.items()]

    def dataNull(self):
        """Datos nulos del dataframe."""
        self.load_data()
        self.resType = 'table'
        self.cols = ['Variable','Cuenta']
        self.rows = list(self.dataframe.isnull().sum().items())

    def dataDescribe(self):
        """Resumen estadístico de variables numéricas."""
        self.load_data()
        self.resType = 'table'
        self.cols = self.dataframe.describe().columns.tolist()
        self.cols.insert(0,'')
        measures = ['Cuenta', 'Media', 'Std.', 'Min','25%','50%','75%','Max']
        self.rows = self.dataframe.describe().values.tolist()

        for row, measure in zip(self.rows, measures):
            row.insert(0, measure)

    def dataCorrelation(self):
        self.load_data()
        self.resType = 'table'
        
        self.cols = np.insert( self.dataframe.corr().columns, 0, '').tolist()
        
        # Redondeo a 6 digitos
        rows_lst = [ [round(c, 6) for c in row] for row in self.dataframe.corr().values]
        # Conversion a string + columna de variables
        self.rows = [ [col] + [str(c) if pd.isna(c) else c for c in row] for col, row in zip(self.cols[1:], rows_lst)]

        data = []
        for col, row in zip(self.cols, rows_lst):
            data.append(
                {
                    'id':col,
                    'data':[ { 'x': x, 'y': None if pd.isna(y) else y } for x, y in zip( self.cols, row ) ]
                }
            )

        self.dataGraph = data

    def dataHistogram(self):
        self.load_data()
        self.resType = 'hist'
        
        datas = []       
        vars = [k for k,v in self.dataframe.dtypes.items() if str(v) != 'object']

        for var in vars:

            hist = self.dataframe[var].hist()
            ax = plt.gca()
            p = ax.patches

            # Alto de las barras (y)
            ys = [p[i].get_height() for i in range(len(p))]
            # Inicio de las barras (x)
            xs = [p[i].get_x() for i in range(len(p))]
            
            dif = (p[1].get_x() - p[0].get_x()) /2
            xs = [ round(dif + i, 1) for i in xs]

            data = [ { "id":x, "value":y } for x, y in zip(xs,ys) ]
            
            datas.append(
                {
                    'data':data,
                    'title':var
                }
            )
            plt.clf()

        self.dataGraph = datas

