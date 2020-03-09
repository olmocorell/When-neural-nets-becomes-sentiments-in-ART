import speech_recognition as sr
r = sr.Recognizer()


def recogeTexto():
    with sr.Microphone() as source:
        print("Expresa algo")
        audio1 = r.record(source, duration=13)
        print("Grabación terminada. Procesando...")
        texto = r.recognize_google(audio1, language='es_ES')        
        with open("audio.wav", "wb") as f:
            f.write(audio1.get_wav_data())
    return texto
