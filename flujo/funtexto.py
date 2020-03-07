import pandas as pd 
import numpy as np

#leemos el dataset de colores para convertir los valores del array en colores
colores = pd.read_csv("../data/colores.csv")

def capa(x,param):
    nuevo = []
    param = str(f"{param}")
    for a in x:
        b = colores.iloc[a][f"{param}"]
        nuevo.append(b)
    return nuevo

def arrayReshape(lista):
    array = np.asarray(lista)
    otro = array.reshape(300,300)
    return otro