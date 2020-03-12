from tkinter import *
from flujo.paralamezcla import *
import functools
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = False
from random import sample
from PIL import Image
import flujo.graba as gr
import flujo.imagen as imagen
from pydub import AudioSegment
import flujo.analiza as an
import IPython.display
import tensorflow as tf
from tkinter import messagebox as mb



def primeraParte():
  global etiqueta
  #Lo primero que hace el programa es recoger el mensaje
  frase = gr.recogeTexto()
  frase = frase.capitalize()
  print(f"La creatividad ha sido {frase}")
  mb.showinfo("Procesado", f"Tu creatividad ha sido: {frase}")
  audio = AudioSegment.from_file(f'audio.wav', format='mp4')

  #Clasificamos el sentimiento de cómo se ha dicho el mensaje.
  etiqueta = an.procesaYpredice(audio)[0]
  if etiqueta == 1:
      print("Lo has expresado con alegría")
      mb.showinfo("Sentimientos", "Lo has expresado con alegría")
  elif etiqueta == 2:
      print("Lo has expresado con enfado")
      mb.showinfo("Sentimientos", "Lo has expresado con enfado")
  elif etiqueta == 3:
      print("Lo has expresado con tristeza")
      mb.showinfo("Sentimientos", "Lo has expresado con tristeza")
  else:
      print("Lo has expresado con afecto")
      mb.showinfo("Sentimientos", "Lo has expresado con afecto")

  #Analizatexto devuelve una tupla con la polaridad del TEXTO y la frase traducida
  polaridad = an.analizaPolaridad(frase)
  pol = polaridad[0]
  frasetrad = polaridad[1]
  print(f"La polaridad es de: {pol}")
  array = an.generaContexto(frasetrad)
  generandoimagen = imagen.generaLaImagen(array,pol)
  mb.showinfo("Completado", "Fase 1 completada, pulsa el botón fase 2 para terminar la obra")

  return "Fase1 completada"

def segundaParte():
  mb.showinfo("Atención", "No cierres el programa, voy a tardar un rato. Te avisaré cuando acabe.")
  
  # Imagen estilo (creada por el texto)
  style_path = "creada.jpg"

  # imagen contenido (formas base, se eligen en función de los sentimientos)
  n = str(sample(range(1, 11), 1)[0])
  path = f"sentimientos/{etiqueta}/"
  end = ".jpg"
  content_path = path+n+end
  print(f'tu style_path es: {content_path}')

  print(f"Formas de base: {n}")
  tf.compat.v1.enable_eager_execution(
      config=None, device_policy=None, execution_mode=None
  )
  print("Eager execution: {}".format(tf.executing_eagerly()))


  # Función que realiza la mezcla final y transfiere el estilo
  def run_style_transfer(content_path, 
                        style_path,
                        num_iterations=1000,
                        content_weight=1e3, 
                        style_weight=1e-2): 
    #No queremos entrenar el modelo, ponemos trainable false
    model = get_model() 
    for layer in model.layers:
      layer.trainable = False
    
    # Sacamos las feature de la capa estilo
    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
    
    # Establecemos imagen inicial
    init_image = load_and_process_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)
    # Añade el optimizador
    opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

    # Para mostrar las imágenes intermedias
    iter_count = 1
    
    # Puntúa el mejor resultado
    best_loss, best_img = float('inf'), None
    
    # Configuración
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    # Para la visualización
    num_rows = 2
    num_cols = 5
    display_interval = num_iterations/(num_rows*num_cols)
    start_time = time.time()
    global_start = time.time()
    
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means   
    
    imgs = []
    for i in range(num_iterations):
      grads, all_loss = compute_grads(cfg)
      loss, style_score, content_score = all_loss
      opt.apply_gradients([(grads, init_image)])
      clipped = tf.clip_by_value(init_image, min_vals, max_vals)
      init_image.assign(clipped)
      end_time = time.time() 
      
      if loss < best_loss:
        # Actualiza la mejor pérdida y la mejor imagen de la pérdida total.
        best_loss = loss
        best_img = deprocess_img(init_image.numpy())

      if i % display_interval== 0:
        start_time = time.time()
        
        # Usa numpy para obtener el array concreto
        
        plot_img = init_image.numpy()
        plot_img = deprocess_img(plot_img)
        imgs.append(plot_img)
        IPython.display.clear_output(wait=True)
        IPython.display.display_png(Image.fromarray(plot_img))
        
        print('Iteration: {}'.format(i))        
        print('Total loss: {:.4e}, ' 
              'style loss: {:.4e}, '
              'content loss: {:.4e}, '
              'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
    print('Total time: {:.4f}s'.format(time.time() - global_start))
    
    IPython.display.clear_output(wait=True)
    plt.figure(figsize=(14,4))
    for i,img in enumerate(imgs):
        plt.subplot(num_rows,num_cols,i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    return best_img, best_loss

  best, best_loss = run_style_transfer(content_path,
                                      style_path, num_iterations=200)

  imagen = Image.fromarray(best)
  imagen.save("mi_obra.jpg")
  mb.showinfo("Terminado", "La obra está terminada")