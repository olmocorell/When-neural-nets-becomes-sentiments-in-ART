from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras.preprocessing import image as kp_image
import tensorflow as tf
import functools
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = False
from random import sample
from PIL import Image
import flujo.graba as gr
import flujo.imagen as imagen
import os
from pydub import AudioSegment
import flujo.analiza as an

#Lo primero que hace el programa es recoger el mensaje
frase = gr.recogeTexto()
print(f"Tu creatividad ha sido: {frase}")
audio = AudioSegment.from_file(f'audio.wav', format='mp4')

#Clasificamos el sentimiento de cómo se ha dicho el mensaje.
etiqueta = an.procesaYpredice(audio)[0]
if etiqueta == 1:
     print("Lo has expresado con alegría")
elif etiqueta == 2:
    print("Lo has expresado con enfado")
elif etiqueta == 3:
    print("Lo has expresado con tristeza")
else:
    print("Lo has expresado con afecto")

#Analizatexto devuelve una tupla con la polaridad del TEXTO y la frase traducida
polaridad = an.analizaPolaridad(frase)
pol = polaridad[0]
frasetrad = polaridad[1]
print(f"La polaridad es de: {pol}")
array = an.generaContexto(frasetrad)
generandoimagen = imagen.generaLaImagen(array,pol)
print("Empiezo a fusionar el contexto con las formas...")
# imagen contenido (creada por el texto)
#voy a invertirlo
style_path = "creada.jpg"

# imagen estilo (temporal)
n = str(sample(range(1, 11), 1)[0])
path = f"sentimientos/{etiqueta}/"
end = ".jpg"
content_path = path+n+end
print(f'tu content_path es: {content_path}')

print(f"Formas de base: {n}")
tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)
print("Eager execution: {}".format(tf.executing_eagerly()))

# Elegimos, de la imagen de contenido, las capas de contenido
content_layers = ['block5_conv2']

# Elegimos,de la imagen de estilo, las capas de estilo
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def load_img(path_to_img):
    """
    Carga la imagen, la convierte en array y la reescala para 
    que tenga el formato adecuado.
    """
    max_dim = 500
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim/long
    img = img.resize(
        (round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)

    img = kp_image.img_to_array(img)

    img = np.expand_dims(img, axis=0)
    return img


def load_and_process_img(path_to_img):
    """
    Las redes de VGG se entrenan sobre la imagen con 
    cada canal normalizado por media = [103.939, 116.779, 123.68]
    Eso es lo que hace esta función, con vgg19 de keras
    preprocesamos la imagen.
    """
    img = load_img(path_to_img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


def deprocess_img(processed_img):
    """
    Proceso inverso para normalizar 
    nuestros valores dentro del rango de 0-255.
    """
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # deshace el paso de preprocessing anterior
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


def get_content_loss(base_content, target):
    """
    Reduce las dimensiones del tensor
    """
    return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(input_tensor):
    """
    separa la imagen en canales y hace la 
    Matriz de Gram
    """
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_style_loss(base_style, gram_target):
    """Expects two images of dimension h, w, c"""
    # height, width, num filters of each layer
    # We scale the loss at a given layer by the size of the feature map and the number of filters
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)

    # / (4. * (channels ** 2) * (width * height) ** 2)
    return tf.reduce_mean(tf.square(gram_style - gram_target))


def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    """
    Calcula la pérdida de las imágenes
    """
    style_weight, content_weight = loss_weights

    # Alimenta la imagen inicial a través del modelo
    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    # Acumula la pérdida del estilo en todas las capas
    # Pondera la contribución de las capas de pérdida
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * \
            get_style_loss(comb_style[0], target_style)

    # Acumula la pérdida de todas las capas content
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * \
            get_content_loss(comb_content[0], target_content)

    style_score *= style_weight
    content_score *= content_weight

    # Obtenemos la pérdida total
    loss = style_score + content_score
    return loss, style_score, content_score


def get_model():
    """ 
    Cargamos el modelo VGG19 que accede a las capas intermedias
    de la imagen.
    """
    # Cargamos modelo preentrenado VGG19
    vgg = tf.keras.applications.vgg19.VGG19(
        include_top=False, weights='imagenet')
    vgg.trainable = False
    # Obtenemos las capas intermedias que hemos decidido antes.
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    # Devuelve el modelo
    return models.Model(vgg.input, model_outputs)

def get_feature_representations(model, content_path, style_path):
    """
    Esta función carga y preprocesa tanto el contenido como el estilo 
    Luego los alimentará a través de la red para obtener
    las salidas de las capas intermedias. 
    Recibe:
    modelo: El modelo que estamos usando.
    ruta_de_contenido: Ruta a la imagen de contenido.
    style_path: Ruta a la imagen de estilo
    Devuelve:
    Características de estilo y las características de contenido. 
    """

    # Carga la imagen
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)

    # Computa el estilo y el contenido
    style_outputs = model(style_image)
    content_outputs = model(content_image)

    # Obtiene las features
    style_features = [style_layer[0]
                      for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0]
                        for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features



def compute_grads(cfg):
  with tf.GradientTape() as tape: 
    all_loss = compute_loss(**cfg)
  # Computa gradientes de imagen de entrada
  total_loss = all_loss[0]
  return tape.gradient(total_loss, cfg['init_image']), all_loss


# Función que realiza la mezcla final y transfiere el estilo
import IPython.display

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
                                     style_path, num_iterations=195)

imagen = Image.fromarray(best)
imagen.save("mi_obra.jpg")