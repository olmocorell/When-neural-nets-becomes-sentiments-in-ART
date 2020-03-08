import requests 
from bs4 import BeautifulSoup
import pandas as pd
import re

#hago webscrapping en la página de pantone
url = "https://www.logorapid.com/pantone"
datos = requests.get(f"{url}").text
soup = BeautifulSoup(datos, 'html.parser')
extraigo = str(soup.select('div.colortable'))

#me quedo con los datos que necesito
divido = extraigo.split("<p>")
todos = divido[1::2]

def matches(string):
    nombre = re.findall(r"^\w+\s*\w*",string)
    codigo = re.findall(r"(\#\w*)",string)
    return nombre[0],codigo[0]

#genero un dataframe
columnas = ["color","codigo"]
colores = pd.DataFrame(list(map(matches,todos)),columns = columnas)

#fórmula para convertir los colores de hexadecimales a RGB
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

#Genero una columna con los colores en RGB
colores["RGB"] = colores.codigo.map(hex_to_rgb)

#Extraigo RGB y genero 3 columnas, una con cada valor.

def extraeR(x):
    b = x[0]
    return b 
def extraeG(x):
    b = x[1]
    return b 
def extraeB(x):
    b = x[2]
    return b 
colores["R"] = colores.RGB.apply(extraeR)
colores["G"] = colores.RGB.apply(extraeG)
colores["B"] = colores.RGB.apply(extraeB)

#Exporto el dataframe como csv y ya tengo mi propio dataset de colores
colores.to_csv("../data/colores.csv")

#Le doy otra vuelta y saco 3 gamas. Positivo negativo y neutro.
#De momento...a mano.

#positivos
pos0= colores.iloc[176:251]
pos1 = colores.iloc[0:3]
pos2= colores.iloc[8:11]
pos3 = colores.iloc[15:18]
pos4 = colores.iloc[477:479]
pos5 = colores.iloc[976:977]
pos6 = colores.iloc[976:977]
positivo = pd.concat([pos0,pos1,pos2,pos3,pos4,pos5,pos6])
positivo = positivo.reset_index(drop=True)
positivo.to_csv("../data/positivos.csv")

#neutros
neu0 = colores.iloc[535:662]
neu1 = colores.iloc[523:517]
neu2 = colores.iloc[504:509]
neutros = pd.concat([neu0,neu1,neu2])
neutros = neutros.reset_index(drop=True)
neutros.to_csv("../data/neutros.csv")

#negativos
neg0 = colores.iloc[483:553]
neg1 = colores.iloc[589:593]
neg2 = colores.iloc[597:600]
neg3 = colores.iloc[604:607]
neg4 = colores.iloc[618:620]
neg5 = colores.iloc[625:627]
neg6 = colores.iloc[949:963]
neg7 = colores.iloc[928:931]
neg8 = colores.iloc[111:139]
negativos = pd.concat([neg0,neg1,neg2,neg3,neg4,neg5,neg6,neg7,neg8])
negativos = negativos.reset_index(drop=True)
negativos.to_csv("../data/negativos.csv")