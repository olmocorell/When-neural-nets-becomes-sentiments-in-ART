import flujo.graba as gr
import flujo.analizatexto as an
import flujo.imagen as imagen
import os
import flujo.analizaaudio as aud 
from pydub import AudioSegment

def recogeAudioyEtiqueta():
    frase = gr.recogeTexto()
    print(f"Tu creatividad ha sido: {frase}")
    audio = AudioSegment.from_file(f'audio.wav', format='mp4')


    etiqueta = aud.procesaYpredice(audio)[0]
    if etiqueta == 1:
        print("Lo has expresado con alegría")
    elif etiqueta == 2:
        print("Lo has expresado con enfado")
    elif etiqueta == 3:
        print("Lo has expresado con tristeza")
    else:
        print("Lo has expresado con afecto")



    #Analizatexto devuelve una tupla con la polaridad y la frase traducida
    polaridad = an.analizaPolaridad(frase)
    pol = polaridad[0]
    frasetrad = polaridad[1]
    print(f"La polaridad es de: {pol}")

    array = an.generaContexto(frasetrad)


    generandoimagen = imagen.generaLaImagen(array,pol)

    print("Empiezo a fusionar el contexto con las formas...")
    #os.system("python3 transferencia.py")

    return etiqueta