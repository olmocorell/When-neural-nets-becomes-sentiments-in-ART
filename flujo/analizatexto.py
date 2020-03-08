import nltk
import pandas as pd
import numpy as np
from PIL import Image
from nltk.tokenize import sent_tokenize, word_tokenize 
import gensim 
from gensim.models import Word2Vec 
from textblob import TextBlob
#import funtexto as funt  
import random


def analizaPolaridad(texto):
    polaridad = 0
    sent = []
    trad = TextBlob(f"{texto}")
    en = trad.translate(from_lang="es",to="en")
    sent.append(en.sentiment)
    for s in sent:
        polaridad = s[0]
    return (polaridad,str(en))


def generaContexto(frase):
    """
    Esta función devuelve el array en función del contexto del texto
    """
    # Reemplaza los saltos de línea por espacios en blanco
    f = frase.replace("\n", " ") 
  
    data = [] 

    # itera sobre el texto
    for i in sent_tokenize(f): 
        temp = [] 
      
    # añade palabras tokenizadas a una lista temporal
        for j in word_tokenize(i): 
            temp.append(j.lower()) 
          
        data.append(temp) 
    # crea el SkipGram model
    model2 = gensim.models.Word2Vec(data, min_count = 1, size = 30, 
                                             window = 5, sg = 1) 


    #Generamos un array de similaridad de features con una palabra
    choice = random.choice(temp)
    palabra = model2[f"{choice}"]
    
    return palabra
