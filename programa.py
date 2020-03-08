import flujo.graba as gr
import flujo.analizatexto as an

frase = gr.recogeTexto()
print(f"Tu creatividad ha sido: {frase}")

#Analizatexto devuelve una tupla con la polaridad y la frase traducida
polaridad = an.analizaTexto(frase)
frasetrad = polaridad[1]
print(f"La polaridad es de: {polaridad[0]}")

array = an.generaContexto(frasetrad)
print(f"El array generado es:\n {array}")
