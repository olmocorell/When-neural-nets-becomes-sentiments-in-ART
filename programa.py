import flujo.graba as gr
import flujo.analizatexto as an
import flujo.imagen as imagen
import os

frase = gr.recogeTexto()
print(f"Tu creatividad ha sido: {frase}")

#Analizatexto devuelve una tupla con la polaridad y la frase traducida
polaridad = an.analizaPolaridad(frase)
pol = polaridad[0]
frasetrad = polaridad[1]
print(f"La polaridad es de: {pol}")

array = an.generaContexto(frasetrad)
print(f"El array generado es:\n {array}")

generandoimagen = imagen.generaLaImagen(array,pol)

os.system("python3 transferencia.py")