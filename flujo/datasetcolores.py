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
