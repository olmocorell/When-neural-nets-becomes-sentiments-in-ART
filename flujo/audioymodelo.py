from sklearn.externals import joblib
import pickle
from pydub import AudioSegment
import numpy as np
from scipy.io.wavfile import read
from scipy.fftpack import fft
import pydub
from scipy import signal
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def dataSetAudios(nombre):
    lista = []
    listafin = []
    for a in range(1,6):
        audio = AudioSegment.from_file(f'../audios/{nombre}{a}.m4a', format='mp4')
        lista.append(audio)
    for a in lista:
        array = a.get_array_of_samples()[:528320]
        four = abs(fft(array))
        listafin.append(four)
    
    return listafin       


def haceDf(elemento):
    temp = []
    for a in elemento:
        df = pd.DataFrame(elemento)
        temp.append(df)
    return pd.concat(temp).reset_index(drop=True)

#El procesado de los datos de audio es un poco engorroso
af = "Afecto"
ale = "Alegria"
enf = "Enfado"
trs = "Tristeza"
#en esta parte obtengo la fft de los audios y la almaceno en unas listas
afecto = dataSetAudios(af)
alegria = dataSetAudios(ale)
enfado = dataSetAudios(enf)
tristeza = dataSetAudios(trs)

#esto es para etiquetarlos
alegriad = {"transformada": [a for a in alegria], "etiqueta": 1}
enfadod = {"transformada": [a for a in enfado], "etiqueta": 2}
tristezad = {"transformada": [a for a in tristeza], "etiqueta":3}
afectod = {"transformada": [a for a in afecto], "etiqueta": 4}
todos = [alegriad,enfadod,tristezad,afectod]
#genero un dataframe de cada emoción
al = pd.DataFrame(alegriad)
tr= pd.DataFrame(tristezad)
en = pd.DataFrame(enfadod)
af = pd.DataFrame(afectod)
todos = pd.concat([al,tr,en,af]).reset_index(drop=True)
#Saco la x y la y con np.vstack para entrenar el modelo
X=np.concatenate((np.vstack(al.transformada),np.vstack(tr.transformada),np.vstack(en.transformada),np.vstack(af.transformada)))
y=np.concatenate((al.etiqueta,tr.etiqueta,en.etiqueta,af.etiqueta))

#entreno los modelos 
gbc = GradientBoostingClassifier()
rfc = RandomForestClassifier(n_estimators = 200)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
gbc.fit(X_train, y_train)
rfc.fit(X_train, y_train)


y_pred = gbc.predict(X_test)
h_pred = rfc.predict(X_test)

accrandomf = accuracy_score(y_test, y_pred)
accgradient = accuracy_score(y_test,h_pred)
print(accrandomf, accgradient)

joblib.dump(gbc, '../modelos/model_gbc2.pkl')
joblib.dump(rfc, '../modelos/model_rfc2.pkl')