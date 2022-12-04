import pandas as pd
import numpy as np

def clsNan(data):
    "Función que convierte los valores NaN a string."
    return [str(c) if pd.isna(c) else c for c in data]

def nan2null(data):
    "Función que convierte los valores NaN a nulos."
    return [None if pd.isna(c) else c for c in data]

def dataframePreview(dataframe):
    "Función que retorna los primeros y últimos 5 elementos de un dataframe."
    
    cols = dataframe.columns.values.tolist()
    rows = dataframe.values.tolist()
    
    rows, cols = rowsEnumerate(rows, cols)
    rows = rowsPreview(rows)

    rows = [ clsNan(r) for r in rows ]

    return cols, rows

def rowsEnumerate(rows, cols=[]):
    "Función que enumera las filas de una lista. Modifica las columnas para este número."

    if not cols:
        val = rows[0]
        if type(val) is list:
            cols = list(range(len(val)))
    
    cols.insert(0,'#')

    rows = [ r if type(r) is list else list(r) for r in rows]

    for i,row in enumerate(rows):
        row.insert(0,i)

    return rows, cols


def rowsPreview(rows):
    cols = len(rows[0])
    l=len(rows)
    r = rows[:5] + [['...' for i in range(cols)]] + rows[l-5:]
    return r




def str2list(params):
    if not params:
        return ''
    params = params.strip()

    chars = ['[',']',"'",'"']
    for char in chars:
        params = params.replace(char,'')
    
    params = params.split(',')
    return [val.strip() for val in params]