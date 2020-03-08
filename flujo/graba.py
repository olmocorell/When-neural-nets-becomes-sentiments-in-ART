import speech_recognition as sr
r = sr.Recognizer()


def recogeTexto():
    with sr.Microphone() as source:
        print("Expresa algo")
        audio1 = r.record(source, duration=5)
        print("Grabación terminada. Procesando...")
        texto = r.recognize_google(audio1, language='es_ES')        
        return texto

