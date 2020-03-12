![Portada](https://github.com/agalvezcorell/When-neural-nets-becomes-sentiments-in-ART/blob/master/readme/portada.jpg)

# When neural nets becomes sentiments in ART

En este proyecto he creado un script con una interfaz gráfica donde un usuario puede crear una obra de arte a través de sus sentimientos.

Tecnologías utilizadas:

- Beautiful Soup
- Word2vec
- Sklearn (sklearn machine learning algorithms)
- NLP (natural language processing)
- Tensorflow
- Deep convolutional neural networks

### Funcionamiento de la interfaz

En primer lugar, la interfaz le pide al usuario que diga lo que quiere expresar para generar la obra. La interfaz recoge la voz, la graba y la analiza.
Por una parte genera una imagen de estilo.
Esta imagen de estilo depende del contexto de lo que diga el usuario, del tipo de mensaje que sea (positivo, negativo) y de la emoción con la que lo diga (con alegria, tristeza, enfado...)
En función de estas variables analizadas con machine learning, nlp y Word2Vec (red neuronal de dos capas) el programa elige una tonalidad, reordena el vector generado por la red neuronal y genera un np.array que convierte en imagen.

Después, el programa le pide al usuario que pulse el segundo botón para iniciar la segunda fase.
En la segunda fase se van a asociar unas formas a cómo el usuario ha dicho el mensaje (elegidas en función de la clasificación de sentimientos que haya realizado el modelo de ML).

Por último, utilizamos una red neuronal profunda convolucional ya entrenada VGG19 a la que, previo procesado, le pasamos a la entrada lo siguiente:

- Una imagen inicial (las formas escogidas)
- Capas intermedias de dicha imagen 
- Capas intermedias de la imagen obtenida del np.array generado por el contexto

Con todo esto, la red neuronal, nos devuelve a la salida una transferencia de estilo que queda como se puede observar a continuación.

