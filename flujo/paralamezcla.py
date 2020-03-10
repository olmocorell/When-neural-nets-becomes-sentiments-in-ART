import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras.preprocessing import image as kp_image
from PIL import Image
import numpy as np

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