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


def formaUno(normalizado):

    #Generamos 4 listas desde normalizado ordenadas de diferente manera.
    no2 = sorted(normalizado)
    no3 = sorted(normalizado,key = lambda a: a%2 == 0)
    no4 = sorted(no3)
    
    #sumamos todas y las aplanamos
    todas = [normalizado,no3,no3,no4]
    normalizado2 = [a for b in todas for a in b]

    #la convertimos en nparray para trabajar con ello
    normalizado2 = np.asarray(normalizado2)
    
    #para multiplicar ese array lo convertimos momentáneamente en lista
    normalizado2 = list(normalizado2) * 750
    
    return normalizado2

def formaDos(normalizado):
    normalizado2 = np.asarray(normalizado)
    normalizado2 = np.sort(normalizado)
    normalizado2 = list(normalizado) * 3000
    return normalizado2