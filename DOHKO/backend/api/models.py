from django.db import models
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import os
from django.conf import settings

from .tools import *

# PCA
from sklearn.decomposition import PCA
import sklearn.preprocessing as skp

#Decision Tree
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


dataframe = None
dataframeScaler = None
pca = None
Pronostico = None

X = None
Y = None
X_train = None
X_test = None
Y_train = None
Y_test = None

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
    complement = None


    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.dataframe = pd.read_csv(self.dataFile or self.url)


    def delete(self):
        if self.dataFile:
            os.remove(os.path.join(settings.MEDIA_ROOT, self.dataFile.name))
        return super().delete()

    def load_data(self):
        global dataframe
        # self.dataframe = pd.read_csv(self.dataFile or self.url)
        dataframe = pd.read_csv(self.dataFile or self.url)

    def getVariables(self, **kargs):
        self.load_data()
        global dataframe

        self.cols = dataframe.columns.values.tolist()

    # EDA
    def dataPreview(self,**kwargs):
        self.load_data()
        global dataframe
        self.resType = 'table'

        # Columna para numero de fila
        # self.cols = np.insert( self.dataframe.columns.values, 0, '')
        self.cols = np.insert( dataframe.columns.values, 0, '')

        # rows_lst = self.dataframe.values.tolist()
        rows_lst = dataframe.values.tolist()

        l = len(rows_lst)
        # Primeros y ultimos 5 elementos
        preview_rows = [0,1,2,3,4,l-5,l-4,l-3,l-2,l-1]
        # Conversion de elementos a str + numero de fila
        self.rows = [ [r] + list(map(lambda cell: str(cell), rows_lst[r])) for r in preview_rows]
        # Separacion
        self.rows.insert(5, ['...' for i in range(len(self.cols))])

    def dataShape(self):
        """Dimesión del dataframe."""
        # self.load_data()
        global dataframe
        self.resType = 'text'

        self.code = {
            'command':'dataframe.shape',
            'result':str(dataframe.shape)
            }


    def dataTypes(self):
        """Tipo de datos del dataframe."""
        # self.load_data()
        global dataframe
        self.resType = 'table'
        self.cols = ['Variable','Tipo']
        # self.rows = [(k,str(v)) for k,v in self.dataframe.dtypes.items()]
        self.rows = [(k,str(v)) for k,v in dataframe.dtypes.items()]

    def dataInfo(self):
        """Descripción de los datos"""
        global dataframe
        self.resType = 'table'

        cols = ['Variable','Nulos','Tipo']
        
        info = []
        variables = dataframe.columns.values.tolist()
        types = dataframe.dtypes
        nulls = dataframe.isnull().sum()

        for var in variables:
            info.append([
                var,
                str(nulls[var]),
                str(types[var])
            ])

        self.rows, self.cols = rowsEnumerate(info, cols)


        

    def dataNull(self):
        """Datos nulos del dataframe."""
        # self.load_data()
        global dataframe
        self.resType = 'table'
        self.cols = ['Variable','Cuenta']
        # self.rows = list(self.dataframe.isnull().sum().items())
        self.rows = list(dataframe.isnull().sum().items())

    def dataDescribe(self):
        """Resumen estadístico de variables numéricas."""
        # self.load_data()
        global dataframe
        self.resType = 'table'
        # self.cols = self.dataframe.describe().columns.tolist()
        self.cols = dataframe.describe().columns.tolist()
        self.cols.insert(0,'')
        measures = ['Cuenta', 'Media', 'Std.', 'Min','25%','50%','75%','Max']
        # self.rows = self.dataframe.describe().values.tolist()
        self.rows = [ clsNan(row) for row in dataframe.describe().values.tolist()]

        for row, measure in zip(self.rows, measures):
            row.insert(0, measure)


    def dataCorrelation(self, **kargs):
        global dataframe
        self.resType = 'table'
        self.complement = 'heatmap'

        self.cols = np.insert( dataframe.corr().columns, 0, '#').tolist()

        # Redondeo a 6 digitos
        rows_lst = [ [round(c, 6) for c in row] for row in dataframe.corr().values]
        # Conversion a string + columna de variables
        self.rows = [ [col] + [str(c) if pd.isna(c) else c for c in row] for col, row in zip(self.cols[1:], rows_lst)]

        data = []
        for col, row in zip(self.cols, rows_lst):
            data.append(
                {
                    'id':col,
                    'data':[ { 'x': x, 'y': None if pd.isna(y) else round(y,2) } for x, y in zip( self.cols, row ) ]
                }
            )

        self.dataGraph = data


    def dataHistogram(self, **kargs):
        global dataframe
        self.resType = 'hist'

        datas = []
        vars = [k for k,v in dataframe.dtypes.items() if str(v) != 'object']

        for var in vars:

            hist = dataframe[var].hist(figsize=(14,14))
            ax = plt.gca()
            p = ax.patches

            # Alto de las barras (y)
            ys = [p[i].get_height() for i in range(len(p))]
            # Inicio de las barras (x)
            xs = [p[i].get_x() for i in range(len(p))]

            xs = [ round(i, 1) for i in xs]
            # xs = [ format(i,'.1e') for i in xs]

            data = [ { "id":x, "value":y } for x, y in zip(xs,ys) ]

            datas.append(
                {
                    'data':data,
                    'title':var,
                    'layout':'vertical'
                }
            )
            plt.clf()

        self.dataGraph = datas


    def dataBoxplot(self, **kargs):
        global dataframe
        self.resType = 'boxp'

        vars = [k for k,v in dataframe.dtypes.items() if str(v) != 'object']
        boxes = []

        for var in vars:
            boxes.append(
                {
                    'name':var,
                    'data':clsNan( dataframe[var].tolist() ),
                }
            )

        self.dataGraph = boxes


    def dataObjectBar(self, **kargs):
        global dataframe
        self.resType = 'hist'

        condition = kargs.get('condition', 'dataframe[var].nunique() < 10')

        vars = dataframe.select_dtypes(include='object')
        datas = []

        for var in vars:

            # if dataframe[var].nunique() < 10:
            #     continue

            if not eval(condition):
                continue

            data = dataframe[var].value_counts()
            x = data.tolist()
            y = data.index.tolist()

            data = [ { "id":y, "value":x } for x, y in zip(x, y) ]

            datas.append(
                {
                    'data':data,
                    'title':var,
                    'layout':'horizontal'
                }
            )

        self.dataGraph = datas


    def dataScaler(self, **kargs):
        global dataframe, dataframeScaler
        self.resType = 'table'

        scalerFunc = skp.StandardScaler()

        scaler = kargs.get('scaler', None)
        if scaler:
            scaler = scaler.replace('(','').replace(')','')

            try:
                scalerFunc = getattr(skp, scaler)()
            except:
                pass
        
        subframe = dataframe.select_dtypes(exclude='object')

        standard = scalerFunc.fit_transform( subframe )
        dataframeScaler = pd.DataFrame(standard, columns=subframe.columns)

        self.cols, self.rows = dataframePreview(dataframeScaler)


    def dataComponents(self, **kargs):
        global dataframeScaler, pca
        self.resType = 'table'

        n_comp = int(kargs.get('n_components', 0))

        pca = PCA(n_components=(n_comp or None))
        pca.fit(dataframeScaler)
        rows = pca.components_.tolist()
        #cols = []
        self.rows, self.cols = rowsEnumerate(rows)


    def dataVariance(self, **kargs):
        global dataframeScaler, pca
        self.resType = 'text'

        varianza = pca.explained_variance_ratio_
        self.code = { 'result':str(varianza) }


    def dataVarianceAcum(self, **kargs):
        global dataframeScaler, pca
        self.resType = 'text'

        n = int(kargs.get('n',0))

        varianza = pca.explained_variance_ratio_
        sumvar = sum(varianza[0:n])
        self.code = { 'result':str(sumvar) }

    def dataVarianceAcumLine(self, **kargs):
        global pca
        self.resType = 'line'

        y = np.cumsum(pca.explained_variance_ratio_).tolist()
        vals = [{"x":str(x), "y":y} for x,y in enumerate(y)]

        data = {
            'id':"varianza",
            'data':vals
        }

        line = [{
            'datas':data,
            'title':"Titulo",
            'x_legend':"Número de componentes",
            'y_legend':"Varianza acumulada"
        }]

        self.dataGraph = line


    def dataWeights(self, **kargs):
        global dataframe, pca

        self.resType = 'table'

        rows = abs(pca.components_).tolist()
        cols = dataframe.columns.values.tolist()

        self.rows, self.cols = rowsEnumerate(rows, cols)


    def dataDrop(self, **kargs):
        global dataframe
        self.resType = 'table'

        dropCols = kargs.get('columns',None)
        if dropCols:
            dropCols = dropCols.replace('[','').replace(']','').split(',')
            dropCols = [col.strip() for col in dropCols]

        df = dataframe.drop(columns=dropCols or [])

        cols = df.columns.values.tolist()
        rows = df.values.tolist()

        rows, self.cols = rowsEnumerate(rows, cols)
        self.rows = rowsPreview(rows)


    def dataDropna(self, **kargs):
        global dataframe
        self.resType = 'table'

        dataframe = dataframe.dropna()

        self.cols, self.rows = dataframePreview(dataframe)


    def dataVPredict(self, **kargs):
        global dataframe, X
        self.resType = 'table'

        cols = []

        vars = kargs.get('vars', None)
        if vars:
            cols = str2list(vars)
            cols = cols

        X = np.array(dataframe[cols])
        rows = dataframe[cols].values.tolist()

        rows, self.cols = rowsEnumerate( rows, cols)
        self.rows = rowsPreview(rows)


    def dataVPronostic(self, **kargs):
        global dataframe, Y
        self.resType = 'table'

        cols = []

        var = kargs.get('var', None)
        if var:
            cols = str2list(var)
            cols = cols[:1]

        Y = np.array(dataframe[cols])
        rows = dataframe[cols].values.tolist()

        rows, self.cols = rowsEnumerate( rows, cols)
        self.rows = rowsPreview(rows)


    def dataDivision(self, **kargs):
        global dataframe, X,Y,X_train, X_test, Y_train, Y_test

        self.resType = 'table'

        tst_sz = kargs.get('test_size', None)
        tst_sz = float(tst_sz) if tst_sz else 0.2

        rndm_stt = kargs.get('random_state', None)
        rndm_stt = int(rndm_stt) if rndm_stt else 0

        shffl = kargs.get('shuffle', None)
        shffl = True if shffl == 'True' else False

        X_train, X_test, Y_train, Y_test =\
            model_selection.train_test_split(X, Y,
            test_size = tst_sz,
            random_state = rndm_stt,
            shuffle = shffl)

        rows, self.cols = rowsEnumerate(X_test.tolist())
        self.rows = rowsPreview(rows)

    def dataModelTrain(self, **kargs):
        global X_train, Y_train, Pronostico

        self.resType = 'text'

        mx_dpth = kargs.get('max_depth',None)
        mx_dpth = int(mx_dpth) if mx_dpth else 10

        mn_smpls_splt = kargs.get('min_samples_split', None)
        mn_smpls_splt = int(mn_smpls_splt) if mn_smpls_splt else 4

        mn_smpls_lf = kargs.get('min_samples_leaf', None)
        mn_smpls_lf = int(mn_smpls_lf) if mn_smpls_lf else 2

        rndm_stt = kargs.get('random_state', None)
        rndm_stt = int(rndm_stt) if rndm_stt else 0

        Pronostico = DecisionTreeRegressor(
            max_depth = mx_dpth,
            min_samples_split = mn_smpls_splt,
            min_samples_leaf = mn_smpls_lf,
            random_state = rndm_stt)

        Pronostico.fit(X_train, Y_train)
        self.code = { 'result':'DecisionTreeRegressor()' }

    def dataPronostic(self, **kwargs):
        global X_test, Pronostico

        self.resType = 'table'

        Y_Pronostico = Pronostico.predict(X_test)
        rows, self.cols = rowsEnumerate(Y_Pronostico.tolist())
        self.rows = rowsPreview(rows)
