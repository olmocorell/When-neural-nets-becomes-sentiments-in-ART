import pandas as pd 
import numpy as np


#Recibimos el array del análisis de texto y lo normalizamos
def normalizaNeutro(array):
    normalizado = ((array-array.max())*((0-150)/(array.min()-array.max())))+150
    #convierto a enteros
    normalizado = normalizado.astype(int)
    return normalizado

def normalizaPositivo(array):
    normalizado = ((array-array.max())*((0-184)/(array.min()-array.max())))+184
    #convierto a enteros
    normalizado = normalizado.astype(int)
    return normalizado

def normalizaNegativo(array):
    normalizado = ((array-array.max())*((0-48)/(array.min()-array.max())))+48
    #convierto a enteros
    normalizado = normalizado.astype(int)
    return normalizado


#creamos 3 capas obteniendo los colores de la columna rgb del dataset
def capa(x,param,clave):
    if clave == "positivos":
        data = pd.read_csv("data/positivos.csv")
    elif clave == "negativos":
        data = pd.read_csv("data/negativos.csv")

    else:
        data = pd.read_csv("data/neutros.csv")

    nuevo = []
    param = str(f"{param}")
    for a in x:
        b = data.iloc[a][f"{param}"]
        nuevo.append(b)
    return nuevo


def arrayReshape(lista):
    array = np.asarray(lista)
    
    otro = array.reshape(300,300)
    #--------------
    # SI ACTIVO ESTO SE ORDENA EL ARRAY POR GAMAS, DIFERENTES RESULTADOS
    #--------------
    #otro = np.sort(otro)
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
