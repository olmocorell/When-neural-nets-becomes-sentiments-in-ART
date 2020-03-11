import speech_recognition as sr
from tkinter import messagebox as mb
r = sr.Recognizer()


def recogeTexto():
    with sr.Microphone() as source:
        mb.showinfo("Voy a grabar", "Expresa lo que quieras crear")
        audio1 = r.record(source, duration=13)
        mb.showinfo("Grabando", "Grabación terminada, voy a procesarla")
        texto = r.recognize_google(audio1, language='es_ES')        
        with open("audio.wav", "wb") as f:
            f.write(audio1.get_wav_data())
    return texto
