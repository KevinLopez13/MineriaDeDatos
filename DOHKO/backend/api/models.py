from django.conf import settings
from django.db import models
import pandas as pd
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime as dt 

import numpy as np
import os
from django.conf import settings
from django.core.files.images import ImageFile

from .tools import *

# PCA
from sklearn.decomposition import PCA
import sklearn.preprocessing as skp

# Decision Tree
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import plot_tree

from sklearn.tree import DecisionTreeClassifier

# RandomForest
from sklearn.ensemble import RandomForestRegressor

# Clustering
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# SVM
from sklearn.svm import SVC
from sklearn.metrics import RocCurveDisplay

dataframe = None
dataframeScaler = None #Matriz estandarizada
pca = None
Pronostico = None

X = None
Y = None
X_train = None
X_test = None
Y_train = None
Y_test = None
Y_Pronostico = None
x_vars = None
y_var = None

SSE = None
cluster_range = None
class_var = None

def upload_to(*args, **kwargs):
    pass

class Project(models.Model):
    name = models.CharField(max_length=20)
    desc = models.CharField(max_length=200, null=True)
    url = models.CharField(max_length=200, null=True)
    fileName = models.CharField(max_length=100, null=True)
    dataFile = models.FileField(upload_to='datasets/', null=True, default=None)
    images = models.ImageField(upload_to='img/', blank=True, null=True)


    dataframe = None
    rows = None
    cols = None
    code = None
    dataGraph = None
    resType = None
    complement = None
    imgs = None

    command = None
    kargs = None
    vars = None
    showVars = None
    checkBoxType = None
    multiline = None
    default_args = None

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.dataframe = pd.read_csv(self.dataFile or self.url)


    def delete(self):
        if self.dataFile:
            os.remove(os.path.join(settings.MEDIA_ROOT, self.dataFile.name))
        return super().delete()


    def load_data(self):
        """Función que carga los datos de un archivo o una url a un dataframe."""

        global dataframe
        if self.dataFile:
            dataframe = pd.read_csv(self.dataFile)
        elif self.url:
            dataframe = pd.read_csv(self.url)


    def getVariables(self, **kargs):
        self.load_data()
        global dataframe
        self.cols = dataframe.columns.values.tolist()

    # EDA
    def dataPreview(self, **kargs):
        """Función que muestra una previsualización del dataframe."""
        
        self.load_data()
        
        global dataframe
        self.command = 'dataframe'

        if kargs.get('config'): return
        self.resType = 'table'

        # Columna para numero de fila
        self.cols = np.insert( dataframe.columns.values, 0, '')
        rows_lst = dataframe.values.tolist()

        l = len(rows_lst)
        preview_rows = [0,1,2,3,4,l-5,l-4,l-3,l-2,l-1]
        # Conversion de elementos a str + numero de fila
        self.rows = [ [r] + list(map(lambda cell: str(cell), rows_lst[r])) for r in preview_rows]
        self.rows.insert(5, ['...' for i in range(len(self.cols))])


    def dataShape(self, **kargs):
        """Función que obtiene la dimesión del dataframe."""
        if kargs.get('config'):
            self.command = 'dataframe.shape'
            return

        global dataframe
        self.resType = 'text'

        self.code = {
            'result':str(dataframe.shape)
            }


    def dataTypes(self, **kargs):
        """Función que obtiene el tipo de datos del dataframe."""
        if kargs.get('config'):
            self.command = 'dataframe.dtype'
            return

        global dataframe
        self.resType = 'table'
        self.cols = ['Variable','Tipo']
        self.rows = [(k,str(v)) for k,v in dataframe.dtypes.items()]


    def dataInfo(self, **kargs):
        """Imitación de la función info() que obtiene la descripción de los datos."""
        if kargs.get('config'):
            self.command = 'dataframe.info()'
            return

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
        

    def dataNull(self, **kargs):
        """Función que calcula los datos nulos del dataframe."""
        if kargs.get('config'):
            self.command = 'dataframe.isnull().sum()'
            return

        global dataframe
        self.resType = 'table'
        self.cols = ['Variable','Cuenta']
        self.rows = list(dataframe.isnull().sum().items())


    def dataDescribe(self, **kargs):
        """Función que obtiene un resumen estadístico de variables numéricas."""
        if kargs.get('config'):
            self.command = 'dataframe.describe()'
            return

        global dataframe
        self.resType = 'table'
        
        self.cols = dataframe.describe().columns.tolist()
        self.cols.insert(0,'')
        measures = ['Cuenta', 'Media', 'Std.', 'Min','25%','50%','75%','Max']

        rows = rowsRound( dataframe.describe().values.tolist(), digits = 2)
        self.rows = [ clsNan(row) for row in rows]

        for row, measure in zip(self.rows, measures):
            row.insert(0, measure)


    def dataCorrelation(self, **kargs):
        """Función que obtiene la correlación entre los datos del dataframe."""
        if kargs.get('config'):
            self.command = 'dataframe.corr()'
            return

        global dataframe
        self.resType = 'table'
        self.complement = 'heatmap'

        # self.cols = np.insert( dataframe.corr().columns, 0, '#').tolist()
        self.cols = dataframe.corr().columns
        rows_lst = rowsRound( dataframe.corr().values)

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

        self.cols = self.cols.insert(0, '#')

        self.dataGraph = data


    def dataHistogram(self, **kargs):
        """Función que obtiene los valores para graficar un histograma de cada variable."""

        if kargs.get('config'):
            self.command = 'dataframe.hist()'
            return
            
        global dataframe
        self.resType = 'hist'

        datas = []
        vars = [k for k,v in dataframe.dtypes.items() if str(v) != 'object']

        for var in vars:
            hist = dataframe[var].hist(figsize=(14,14))
            ax = plt.gca()
            p = ax.patches
            ys = [p[i].get_height() for i in range(len(p))] # Alto de las barras (y)
            xs = [p[i].get_x() for i in range(len(p))]      # Inicio de las barras (x)
            xs = [ round(i, 1) for i in xs]                 # Redondeo a 1 digito decimal
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
        """Función que prepara los datos para graficar una caja de bigotes de cada variable."""

        global dataframe
        self.command = 'sns.boxplot(variable, data = dataframe)'

        if kargs.get('config'): return
        self.resType = 'boxp'

        vars = dataframe.select_dtypes(exclude='object').columns.tolist()
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
        """Función que prepara los datos datos para realizar una gráfica de barras
        de los datos no numéricos.
        """
        global dataframe

        self.command = 'sns.countplot(y = col, data = dataframe)'
        self.checkBoxType = 'checkbox'


        if kargs.get('config'):
            self.vars = dataframe.select_dtypes(include='object').columns.tolist()
            return

        self.resType = 'hist'

        vars = kargs.get('vars',[])
        datas = []

        for var in vars:

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
        """Función que estandariza los datos."""

        global dataframe, dataframeScaler
        def_scaler = 'StandardScaler'

        self.default_args = [
            {
                'label':'scaler',
                'type': 'select',
                'options':[
                    {'label':'StandardScaler', 'value': 'StandardScaler' },
                    {'label':'MinMaxScaler', 'value': 'MinMaxScaler'}
                ]
            }
        ]

        if kargs.get('config'):
            self.command = 'standard_df = scaler.fit_transform(dataframe)'
            return

        self.resType = 'table'

        scaler = kargs.get('scaler', def_scaler)
        scalerFunc = getattr(skp, scaler)()

        subframe = dataframe.select_dtypes(exclude='object')

        standard = scalerFunc.fit_transform( subframe )
        dataframeScaler = pd.DataFrame(standard, columns=subframe.columns)

        cols = dataframeScaler.columns.values.tolist()
        rows = rowsRound(dataframeScaler.values.tolist())
        rows, self.cols = rowsEnumerate(rows,cols)
        self.rows = rowsPreview(rows)


    def dataComponents(self, **kargs):
        """Función que obtiene los componentes del modelo."""
        
        global dataframeScaler, pca
        def_n_components = None

        self.default_args = [
            {
                'label':'n_components',
                'type':'text',
                'default': def_n_components
            }
        ]

        self.command = 'PCA(n_components).fit(standard_df)'

        if kargs.get('config'):
            return

        self.resType = 'table'

        n_comp = kargs.get('n_components', 0)
        n_comp = int(n_comp) if n_comp else None

        pca = PCA(n_components=(n_comp))
        pca.fit(dataframeScaler)
        rows = rowsRound( pca.components_.tolist() )
        self.rows, self.cols = rowsEnumerate(rows)


    def dataVariance(self, **kargs):
        """Función que obtiene la varianza de los componentes."""
        if kargs.get('config'):
            self.command = "varianza"
            return
        global dataframeScaler, pca
        self.resType = 'table'

        varianza = pca.explained_variance_ratio_
        rows = rowsRound([varianza.tolist()])
        self.rows, self.cols = rowsEnumerate(rows[0])


    def dataVarianceAcum(self, **kargs):
        """Función que obtiene la varianza acumulada de n componentes.

            Args:
                n: Numero de componentes.
        """
        global dataframeScaler, pca
        def_n = 1        

        self.default_args = [
            {
                'label':'n_components',
                'type':'text',
                'default': def_n,
            }
        ]
        self.command = "sum(varianza[0:n_components])"

        if kargs.get('config'): return
        self.resType = 'text'

        n = int(kargs.get('n_components', def_n))

        varianza = pca.explained_variance_ratio_
        sumvar = sum(varianza[0:n])
        self.code = { 'result':str(sumvar) }


    def dataVarianceAcumLine(self, **kargs):
        """Función que prepara los datos para graficar la varianza acumulada."""
        
        if kargs.get('config'):
            self.command = 'plt.plot(np.cumsum(pca.explained_variance_ratio_))'
            return

        global pca
        self.resType = 'line'

        y = np.cumsum(pca.explained_variance_ratio_).tolist()
        vals = [{"x":str(x), "y":y} for x,y in enumerate(y)]

        data = {
            'id':"varianza",
            'data':vals
        }

        line = {
            'datas':[data],
            'title':"Titulo",
            'x_legend':"Número de componentes",
            'y_legend':"Varianza acumulada"
        }

        self.dataGraph = line


    def dataWeights(self, **kargs):
        """Función que obtiene la proporción de relevancia de las variables."""

        if kargs.get('config'):
            self.command = 'pd.DataFrame( abs(pca.components_) )'
            return

        global dataframe, pca

        self.resType = 'table'

        df = pd.DataFrame(abs(pca.components_), columns = dataframe.columns)
        rows = rowsRound(df.values.tolist())
        cols = df.columns.values.tolist()
        self.rows, self.cols = rowsEnumerate(rows, cols)


    def dataDrop(self, **kargs):
        """Función que elimina columnas del dataframe. Devuelve una copia.

            Args:
                vars: Lista de variables a eliminar.
        """
        global dataframe
        self.command = 'dataframe.drop(columns)'
        self.checkBoxType = 'checkbox'

        if kargs.get('config'):
            # self.load_data()
            self.vars = dataframe.columns.tolist()
            return

        self.resType = 'table'

        dropCols = kargs.get('vars',[])

        df = dataframe.drop(columns=dropCols)

        cols = df.columns.values.tolist()
        rows = df.values.tolist()

        rows, self.cols = rowsEnumerate(rows, cols)
        self.rows = rowsPreview(rows)

    def dataDropInplace(self, **kargs):
        """ Función que elimina columnas del dataframe
        """

        global dataframe
        self.command = 'dataframe.drop(colum, axis=1, inplace=True)'
        self.checkBoxType = 'checkbox'

        if kargs.get('config'):
            # self.load_data()
            self.vars = dataframe.columns.tolist()
            return

        self.resType = 'table'

        dropCols = kargs.get('vars',[])

        try:
            dataframe.drop(columns=dropCols, axis=1, inplace=True)
        except:
            pass

        cols = dataframe.columns.values.tolist()
        rows = dataframe.values.tolist()

        rows, self.cols = rowsEnumerate(rows, cols)
        self.rows = rowsPreview(rows)
    


    def dataDropna(self, **kargs):
        """Función que elimina los valores nulos."""

        if kargs.get('config'):
            self.command = 'dataframe.dropna()'
            return

        global dataframe
        self.resType = 'table'

        dataframe = dataframe.dropna()
        self.rows, self.cols = dataframePreview(dataframe)


    def dataVPredict(self, **kargs):
        """Función que crea un arreglo con las variables predictoras.

            Args:
                vars: Lista de variables a incluir en el arreglo.
        """
        global dataframe, X, x_vars

        self.command = 'X = np.array( dataframe[vars] )'
        self.checkBoxType = 'checkbox'
        self.vars = dataframe.columns.tolist()

        if kargs.get('config'):
            #self.load_data()
            return

        
        self.resType = 'table'

        x_vars = kargs.get('vars', [])
        if '#' in x_vars:
            x_vars.remove('#')
        X = np.array(dataframe[x_vars])
        rows = dataframe[x_vars].values.tolist()

        rows, self.cols = rowsEnumerate( rows, x_vars)
        self.rows = rowsPreview(rows)
        


    def dataVPronostic(self, **kargs):
        """Función que crea un arreglo con la variable a pronosticar.

            Args:
                vars: Lista con la variable a pronosticar.
        """
        global dataframe, Y, y_var
        
        self.command = 'Y = np.array( dataframe[var] )'
        self.checkBoxType = 'radio'

        if kargs.get('config'):
            # self.load_data()
            self.vars = dataframe.columns.tolist()
            return 

        self.resType = 'table'

        y_var = kargs.get('vars', [])
        if '#' in y_var:
            y_var.remove('#')
        Y = np.array(dataframe[y_var])
        rows = dataframe[y_var].values.tolist()

        rows, self.cols = rowsEnumerate( rows, y_var)
        self.rows = rowsPreview(rows)


    def dataDivision(self, **kargs):
        """Función que realiza la división de datos.
            Args:
                test_size: Tamaño de la prueba.
                random_state: Estado aleatorio.
                shuffle: 
        """
        global dataframe, X,Y,X_train, X_test, Y_train, Y_test
        
        def_tst_sz = 0.2
        def_rndm_stt = 0
        def_shffl = False

        self.default_args = [
                {
                    'label':'test_size',
                    'type':'text',
                    'default': def_tst_sz
                },
                {
                    'label':'random_state',
                    'type':'text',
                    'default': def_rndm_stt
                },
                {
                    'label':'shuffle',
                    'type':'select',
                    'default': def_shffl,
                    'options': [
                        {'label':'True', 'value': True},
                        {'label':'False', 'value': False},
                    ]
                },
        ]

        self.command = 'X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(test_size, random_state, shuffle)'

        if kargs.get('config'): return
        self.resType = 'table'

        tst_sz = float(kargs.get('test_size', def_tst_sz))
        rndm_stt = int(kargs.get('random_state', def_rndm_stt))
        shffl = kargs.get('shuffle', def_shffl)
        shffl = True if shffl == 'True' else False

        X_train, X_test, Y_train, Y_test =\
            model_selection.train_test_split(X, Y,
                test_size = tst_sz,
                random_state = rndm_stt,
                shuffle = shffl)

        self.cols = ['Conjunto', 'Cantidad de datos']
        self.rows = [['X_train',len(X_train)],
                    ['Y_train', len(Y_train)],
                    ['X_validation', len(X_test)],
                    ['Y_validation', len(Y_test)]]


    # Arbol de decisión: Pronóstico
    def dataModelTrainDT(self, **kargs):
        """Función que entrena el modelo del pronóstico de árbol de decisión.
            Args:
                max_depth: Profundidad del arbol
                min_samples_split: Minimo de elementos para dividir
                min_samples_leaf: Minimo de elementos por hoja
                random_state: Estado
        """
        global X_test, X_train, Y_train, Y_test, Pronostico, Y_Pronostico

        def_mx_dpth = 10
        def_mn_smpls_splt = 4
        def_mn_smpls_lf = 2
        def_rndm_stt = 0

        self.default_args = [
                {
                    'label':'max_depth',
                    'type':'text',
                    'default': def_mx_dpth
                },
                {
                    'label':'min_samples_split',
                    'type':'text',
                    'default':def_mn_smpls_splt
                },
                {
                    'label':'min_samples_leaf',
                    'type':'text',
                    'default':def_mn_smpls_lf
                },
                {
                    'label':'random_state',
                    'type':'text',
                    'default':def_rndm_stt
                }
            ]

        self.command = "Pronostico = DecisionTreeRegressor().fit(X_train, Y_train)\nY_Pronostico = Pronostico.predict(X_validation)"
        self.multiline = True

        if kargs.get('config'): return
        self.resType = 'table'

        mx_dpth = int(kargs.get('max_depth', def_mx_dpth))
        mn_smpls_splt = int(kargs.get('min_samples_split', def_mn_smpls_splt))
        mn_smpls_lf = int(kargs.get('min_samples_leaf', def_mn_smpls_lf))
        rndm_stt = int(kargs.get('random_state', def_rndm_stt))

        # Entrenamiento
        Pronostico = DecisionTreeRegressor(
            max_depth = mx_dpth,
            min_samples_split = mn_smpls_splt,
            min_samples_leaf = mn_smpls_lf,
            random_state = rndm_stt)

        Pronostico.fit(X_train, Y_train)        

        # Pronostico
        Y_Pronostico = Pronostico.predict(X_test)
        dfp = pd.DataFrame(Y_Pronostico, columns=['Y_Pronostico'])
        dft = pd.DataFrame(Y_test, columns=['Y_validation'])
        df = dft.join(dfp)
        rows = rowsRound( df.values.tolist() )
        rows, self.cols = rowsEnumerate( rows, df.columns.values.tolist() )
        self.rows = rowsPreview(rows)

    # Arbol de decisión Clasificación
    def dataModelTrainCDT(self, **kargs):
        """Función que entrena el modelo del pronóstico de árbol de decisión.
            Args:
                max_depth: Profundidad del arbol
                min_samples_split: Minimo de elementos para dividir
                min_samples_leaf: Minimo de elementos por hoja
                random_state: Estado
        """
        global X_test, X_train, Y_train, Y_test, Pronostico, Y_Pronostico

        def_mx_dpth = 10
        def_mn_smpls_splt = 4
        def_mn_smpls_lf = 2
        def_rndm_stt = 0

        self.default_args = [
                {
                    'label':'max_depth',
                    'type':'text',
                    'default': def_mx_dpth
                },
                {
                    'label':'min_samples_split',
                    'type':'text',
                    'default':def_mn_smpls_splt
                },
                {
                    'label':'min_samples_leaf',
                    'type':'text',
                    'default':def_mn_smpls_lf
                },
                {
                    'label':'random_state',
                    'type':'text',
                    'default':def_rndm_stt
                }
            ]

        self.command = "Clasificacion = DecisionTreeClassifier().fit(X_train, Y_train)\nY_Clasificacion = Clasificacion.predict(X_validation)"
        self.multiline = True

        if kargs.get('config'): return
        # self.resType = 'text'
        self.resType = 'table'

        mx_dpth = int(kargs.get('max_depth', def_mx_dpth))
        mn_smpls_splt = int(kargs.get('min_samples_split', def_mn_smpls_splt))
        mn_smpls_lf = int(kargs.get('min_samples_leaf', def_mn_smpls_lf))
        rndm_stt = int(kargs.get('random_state', def_rndm_stt))

        # Entrenamiento
        Pronostico = DecisionTreeClassifier(
            max_depth = mx_dpth,
            min_samples_split = mn_smpls_splt,
            min_samples_leaf = mn_smpls_lf,
            random_state = rndm_stt)

        Pronostico.fit(X_train, Y_train)        
        # self.code = { 'result':'DecisionTreeRegressor()' }

        # Pronostico
        Y_Pronostico = Pronostico.predict(X_test)
        dfp = pd.DataFrame(Y_Pronostico, columns=['Y_Clasificacion'])
        dft = pd.DataFrame(Y_test, columns=['Y_validation'])
        df = dft.join(dfp)
        rows = rowsRound( df.values.tolist() )
        self.rows, self.cols = rowsEnumerate( rows, df.columns.values.tolist() )
        # self.rows = rowsPreview(rows)

    def dataVariableStatus(self, **kargs):
        """ Función que devuelve una tabla con la importancia de las variables.
        """

        global Y_test, Y_Pronostico, Pronostico, x_vars
        
        self.command = "Pronostico.feature_importances_"
        
        if kargs.get('config'): return
        self.resType = 'table'
        
        if '#' in x_vars:
            x_vars.remove('#')

        Importancia = pd.DataFrame(
                {
                    'Variable': x_vars,
                    'Importancia': Pronostico.feature_importances_
                }
            ).sort_values('Importancia', ascending=False)

        self.cols = Importancia.columns.values.tolist()
        self. rows = rowsRound(Importancia.values.tolist())

    
    def dataMatrixClassification(self, **kargs):
        """ Función que calcula la matriz de clasificación.
        """
        global Pronostico, X_test, Y_test
        
        self.command = "pandas.crosstab( Y_validation.ravel(), Clasificacion.predict(X_validation) )"
        
        if kargs.get('config'): return
        self.resType = 'table'

        #Matriz de clasificación
        Modelo = Pronostico.predict(X_test)
        Matriz = pd.crosstab(Y_test.ravel(), 
                                        Modelo, 
                                        rownames=['Reales'], 
                                        colnames=['Clasificación']) 

        rows = []
        for v,l in zip(Matriz.columns.tolist(), Matriz.values.tolist()):
            row = [v] + l
            rows.append(row)

        self.cols = ["Reales\Clasificación"] + Matriz.columns.tolist()
        self.rows = rows


    # Bosques aleatorios
    def dataModelTrainRF(self, **kargs):
        """Función que entrena el modelo del pronóstico de bosque aleatorio.
            Args:
                n_estimators: Número de árboles
                max_depth: Profundidad del arbol
                min_samples_split: Minimo de elementos para dividir
                min_samples_leaf: Minimo de elementos por hoja
                random_state: Estado
        """
        global X_train, Y_train, Pronostico

        def_n_estimator = 100
        def_mx_dpth = 10
        def_mn_smpls_splt = 4
        def_mn_smpls_lf = 2
        def_rndm_stt = 0

        self.default_args = [
                {
                    'label':'n_estimator',
                    'type': 'text',
                    'default': def_n_estimator
                },
                {
                    'label':'max_depth',
                    'type':'text',
                    'default': def_mx_dpth
                },
                {
                    'label':'min_samples_split',
                    'type':'text',
                    'default':def_mn_smpls_splt
                },
                {
                    'label':'min_samples_leaf',
                    'type':'text',
                    'default':def_mn_smpls_lf
                },
                {
                    'label':'random_state',
                    'type':'text',
                    'default':def_rndm_stt
                }
            ]

        self.command = "Pronostico = RandomForestRegressor(n_estimators).fit(X_train, Y_train)\nY_Pronostico = Pronostico.predict(X_test)"
        self.multiline = True

        if kargs.get('config'): return
        self.resType = 'table'

        n_stmtr = int(kargs.get('n_estimator', def_n_estimator))
        mx_dpth = int(kargs.get('max_depth', def_mx_dpth))
        mn_smpls_splt = int(kargs.get('min_samples_split', def_mn_smpls_splt))
        mn_smpls_lf = int(kargs.get('min_samples_leaf', def_mn_smpls_lf))
        rndm_stt = int(kargs.get('random_state', def_rndm_stt))

        Pronostico = RandomForestRegressor(
            n_estimators = n_stmtr,
            max_depth = mx_dpth,
            min_samples_split = mn_smpls_splt,
            min_samples_leaf = mn_smpls_lf,
            random_state = rndm_stt)

        Pronostico.fit(X_train, Y_train)
        # self.code = { 'result':'RandomForestRegressor()' }

        # Pronostico
        Y_Pronostico = Pronostico.predict(X_test)
        dfp = pd.DataFrame(Y_Pronostico, columns=['Y_Pronostico'])
        dft = pd.DataFrame(Y_test, columns=['Y_validation'])
        df = dft.join(dfp)
        rows = rowsRound( df.values.tolist() )
        rows, self.cols = rowsEnumerate( rows, df.columns.values.tolist() )
        self.rows = rowsPreview(rows)


    def dataScore(self, **kargs):
        """Función que obtiene el puntaje del pronóstico."""

        global Y_test, Y_Pronostico, Pronostico
        self.command = "r2_score(Y_validacion, Y_Pronostico)"
        
        if kargs.get('config'): return
        self.resType = 'table'

        self.cols = ['Medida', 'Valor']
        self.rows = [['Score', '%.4f' % r2_score(Y_test, Y_Pronostico)]]

    
    def dataAccuracyScore(self, **kargs):
        """Función que obtiene el puntaje de exactitud del pronóstico."""

        global Y_test, Y_Pronostico, Pronostico
        self.command = "accuracy_score(Y_validacion, Y_Clasificacion)"
        
        if kargs.get('config'): return
        self.resType = 'table'

        self.cols = ['Medida', 'Valor']
        self.rows = [['Exactitud', '%.4f' % accuracy_score(Y_test, Y_Pronostico)]]



    def dataPlotDTree(self, **kargs):
        """Función que crea el árbol del pronóstico.

            Args:
                feature_names:
        """

        global Pronostico, x_vars
        self.command = "plot_tree(Pronostico, feature_names)"        
        
        if kargs.get('config'): return
        self.resType = 'img'
        
        img_name = f'image{dt.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        plot_tree(Pronostico, feature_names = x_vars)
        plt.subplots_adjust(top=1.0, bottom=0.01, left=0.01, right=1.0)
        plt.savefig(settings.MEDIA_ROOT+"\img\\"+img_name)
        self.images = 'img\\'+img_name
        self.save()

    def dataPlotCTree(self, **kargs):
        """Función que crea el árbol de clasificación.

            Args:
                feature_names:
        """

        global Pronostico, x_vars
        self.command = "plot_tree(Estimador, feature_names)"        
        
        if kargs.get('config'): return
        self.resType = 'img'
        
        img_name = f'image{dt.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        Estimador = Pronostico.estimators_[99]
        plot_tree(Estimador, feature_names = x_vars)

        plt.subplots_adjust(top=1.0, bottom=0.01, left=0.01, right=1.0)
        plt.savefig(settings.MEDIA_ROOT+"\img\\"+img_name)
        self.images = 'img\\'+img_name
        self.save()

    def dataPlotClasTree(self, **kargs):
        """Función que crea el árbol de clasificación.

            Args:
                feature_names:
        """

        global Pronostico, x_vars
        self.command = "plot_tree(Estimador, feature_names)"        
        
        if kargs.get('config'): return
        self.resType = 'img'
        
        img_name = f'image{dt.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        plot_tree(Pronostico, feature_names = x_vars)

        plt.subplots_adjust(top=1.0, bottom=0.01, left=0.01, right=1.0)
        plt.savefig(settings.MEDIA_ROOT+"\img\\"+img_name)
        self.images = 'img\\'+img_name
        self.save()

    def dataNewPronostic(self, **kargs):
        """ Función que calcula nuevos pronósticos.
        """

        global x_vars, Pronostico, y_var

        self.command = 'Pronostico.predict()'
        
        self.default_args = []
        if '#' in x_vars:
            x_vars.remove('#')

        if x_vars:
            for var in x_vars:
                self.default_args.append(
                    {
                        'label': var,
                        'type':'text',
                        'default': 0
                    }
                )

        if kargs.get('config'): return
        self.resType = 'text'
        
        values = { key:[kargs.get(key, 0)] for key in x_vars }

        newDF = pd.DataFrame(values)
        res = Pronostico.predict(newDF)
        
        self.code = { 'result': f'{y_var[0]}: {round(res[0],6)}' }

    # ----------------------------
    # Clustering
    # ----------------------------

    def dataClustersInit(self, **kargs):
        """ Función que define los clústers.
        """

        global dataframe, dataframeScaler, SSE, cluster_range

        def_min_clusters = 2
        def_max_clusters = 10

        self.default_args = [
            {
                'label' : 'min_clusters',
                'type' : 'text',
                'default' : def_min_clusters
            },
            {
                'label' : 'max_clusters',
                'type' : 'text',
                'default' : def_max_clusters
            },
        ]

        self.command = "KMeans.inertia_"

        if kargs.get('config'): return
        self.resType = 'line'
        
        min_c = int(kargs.get('min_clusters', def_min_clusters))
        max_c = int(kargs.get('max_clusters', def_max_clusters))
        
        cluster_range = range(min_c, max_c)

        SSE = []
        sse_line = []
        for i in cluster_range:
            km = KMeans(n_clusters=i, random_state=0)
            km.fit(dataframeScaler)
            SSE.append(km.inertia_)
            sse_line.append({"x": i, "y":km.inertia_})

        data = {
            'id':"clusters",
            'data':sse_line
        }

        line = {
            'datas':[data],
            'title':"Elbow Method",
            'x_legend':"Cantidad de clusters k",
            'y_legend':"SSE"
        }

        self.dataGraph = line

    def dataKneeLocator(self, **kargs):
        """ Función que identifica el codo (cantidad de clusters).
        """
        global SSE, cluster_range
        
        self.command = "KneeLocator(range, SSE)\nKneeLocator.elbow"
        self.multiline = True

        if kargs.get('config'): return
        self.resType = 'text'

        kl = KneeLocator(cluster_range, SSE, curve="convex", direction="decreasing")
        # kl.plot_knee()
        
        self.code = { 'result': 'n_clusters: %s' % kl.elbow }


    def dataClusterLebels(self, **kargs):
        """ Función que crea las etiquetas de los elementos en los clusters."""
        
        global dataframe, dataframeScaler
        def_n_clusters = 1
        def_rdm_state = 0
        self.default_args = [
            {
                'label' : 'n_clusters',
                'type' : 'text',
                'default' : def_n_clusters
            },
            {
                'label' : 'random_state',
                'type' : 'text',
                'default' : def_rdm_state
            }
        ]

        self.command = "KMeans(n_clusters, random_state).fit(dataframe)\nKmeans.labels_"
        self.multiline = True
        if kargs.get('config'): return
        self.resType = 'table'

        n_clstrs = int(kargs.get('n_clusters', def_n_clusters))
        rndm_stt = int(kargs.get('random_state', def_rdm_state))

        MParticional = KMeans(n_clusters=n_clstrs, random_state=rndm_stt).fit(dataframeScaler)
        MParticional.predict(dataframeScaler)
        dataframe['clusterP'] = MParticional.labels_

        cols = dataframe.columns.values.tolist()
        rows = dataframe.values.tolist()
        rows, self.cols = rowsEnumerate(rows, cols)
        self.rows = rowsPreview(rows)

    def dataGetGroups(self, **kargs):
        """ Función que obtiene los centroides.
        """

        global dataframe

        self.command = "dataframe.groupby('clusterP').mean()"

        if kargs.get('config'): return
        self.resType = 'table'

        CentroidesP = dataframe.groupby('clusterP').mean()
        
        cols = CentroidesP.columns.values.tolist()
        rows = CentroidesP.values.tolist()

        rows, self.cols = rowsEnumerate(rows, cols)

        self.cols[0] = 'clusterP'
        self.rows = rowsRound(rows)

    
    def dataModelTrainCRF(self, **kargs):
        """Función que entrena el modelo del pronóstico de bosque aleatorio para cladificación.
            Args:
                n_estimators: Número de árboles
                max_depth: Profundidad del arbol
                min_samples_split: Minimo de elementos para dividir
                min_samples_leaf: Minimo de elementos por hoja
                random_state: Estado
        """
        global X_train, Y_train, X_test, Pronostico, Y_Pronostico

        def_n_estimators = 100
        def_mx_dpth = 10
        def_mn_smpls_splt = 4
        def_mn_smpls_lf = 2
        def_rndm_stt = 0

        self.default_args = [
                {
                    'label':'n_estimators',
                    'type': 'text',
                    'default': def_n_estimators
                },
                {
                    'label':'max_depth',
                    'type':'text',
                    'default': def_mx_dpth
                },
                {
                    'label':'min_samples_split',
                    'type':'text',
                    'default':def_mn_smpls_splt
                },
                {
                    'label':'min_samples_leaf',
                    'type':'text',
                    'default':def_mn_smpls_lf
                },
                {
                    'label':'random_state',
                    'type':'text',
                    'default':def_rndm_stt
                }
            ]

        self.command = "Clasificacion = RandomForestClassifier(n_estimators, max_depth, min_samples_split, min_samples_leaf, random_state).fit(X_train, Y_train)\nY_Clasificacion = Clasificacion.predict(X_validation)"
        self.multiline = True

        if kargs.get('config'): return
        self.resType = 'table'

        n_stmtr = int(kargs.get('n_estimator', def_n_estimators))
        mx_dpth = int(kargs.get('max_depth', def_mx_dpth))
        mn_smpls_splt = int(kargs.get('min_samples_split', def_mn_smpls_splt))
        mn_smpls_lf = int(kargs.get('min_samples_leaf', def_mn_smpls_lf))
        rndm_stt = int(kargs.get('random_state', def_rndm_stt))

        Pronostico = RandomForestClassifier(
            n_estimators = n_stmtr,
            max_depth = mx_dpth,
            min_samples_split = mn_smpls_splt,
            min_samples_leaf = mn_smpls_lf,
            random_state = rndm_stt)

        Pronostico.fit(X_train, Y_train)
        # self.code = { 'result':str(Pronostico) }

        if kargs.get('config'): return
        self.resType = 'table'

        Y_Pronostico = Pronostico.predict(X_test)
        dfp = pd.DataFrame(Y_Pronostico, columns=['Y_Clasificacion'])
        dft = pd.DataFrame(Y_test, columns=['Y_Validation'])
        df = dft.join(dfp)
        rows = rowsRound( df.values.tolist() )
        rows, self.cols = rowsEnumerate( rows, df.columns.values.tolist() )
        self.rows = rowsPreview(rows)

    def dataGroupBySize(self, **kargs):
        """ Función que devuelve 
        """
        
        global dataframe

        self.command = "dataframe.groupby(variable).size()"
        self.checkBoxType = 'radio'

        if kargs.get('config'):
            try:
                self.vars = dataframe.columns.tolist()
            except:
                self.load_data()
                self.vars = dataframe.columns.tolist()
            return

        self.resType = 'table'

        var = kargs.get('vars', [])

        serie = dataframe.groupby(str(var[0])).size()

        self.cols = ['Grupo', 'Cantidad']
        self.rows = [[k,v] for k,v in serie.to_dict().items()]

    # ----------------------------
    # SVM
    # ----------------------------

    def dataModelTrainSVM(self, **kargs):
        """ Función que entrena el modelo SVM."""
        
        global X_train, Y_train, Pronostico
        def_degree = 3
        self.default_args = [
            {
                'label':'kernel',
                'type':'select',
                'options':[
                    {'label': 'linear', 'value':'linear'},
                    {'label': 'poly', 'value':'poly'},
                    {'label': 'rbf', 'value':'rbf'},
                    {'label': 'sigmoid', 'value':'sigmoid'}
                ]
            },{
                'label':'degree',
                'type':'text',
                'default': def_degree
            }
        ]
        self.command = 'SVC(kernel, degree).fit(X_train, Y_train)'
        if kargs.get('config'): return
        self.resType = 'table'

        knl = kargs.get('kernel', 'linear')
        drg = int(kargs.get('degree', def_degree))

        Pronostico = SVC(kernel=knl, degree=drg)
        Pronostico.fit(X_train, Y_train)
        
        Y_Pronostico = Pronostico.predict(X_test)
        dfp = pd.DataFrame(Y_Pronostico, columns=['Y_Clasificacion'])
        dft = pd.DataFrame(Y_test, columns=['Y_Validation'])
        df = dft.join(dfp)
        rows = rowsRound( df.values.tolist() )
        rows, self.cols = rowsEnumerate( rows, df.columns.values.tolist() )
        self.rows = rowsPreview(rows)

    def dataMeanScore(self, **kargs):
        """Función que obtiene el puntaje promedio del pronóstico."""

        global Y_test, X_test, Pronostico
        
        #Se calcula la exactitud promedio de la validación
        self.command = "modelo.score(X_validacion, Y_validacion)"

        if kargs.get('config'): return
        self.resType = 'table'

        self.cols = ['Medida', 'Valor']
        self.rows = [['Score', '%.4f' % Pronostico.score(X_test, Y_test)]]


    def dataPlotROC(self, **kargs):
        """ Función que crea la curva ROC.
        """
        global pca, Pronostico, X_test, Y_test
        
        self.command = 'RocCurveDisplay.from_estimator(ModeloSVM, X_validation, Y_validation)'

        if kargs.get('config'): return        
        self.resType = 'img'
        
        img_name = f'image{dt.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        CurvaROC = RocCurveDisplay.from_estimator(Pronostico, X_test, Y_test)
        plt.subplots_adjust(top=1.0, bottom=0.10, left=0.10, right=0.95)

        CurvaROC.figure_.savefig(settings.MEDIA_ROOT+"\img\\"+img_name)
        self.images = 'img\\'+img_name
        self.save()