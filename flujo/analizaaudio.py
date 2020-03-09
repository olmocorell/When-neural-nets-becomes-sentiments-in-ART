from sklearn.externals import joblib
import pickle
from scipy.fftpack import fft


rfc = joblib.load('modelos/model_rfc.pkl')


def procesaYpredice(audio):
    lista = []
    array = audio.get_array_of_samples()[:528320]
    four = abs(fft(array))
    lista.append(four)
    prediccion = rfc.predict(lista)
    return prediccion

