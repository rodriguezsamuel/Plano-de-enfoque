import numpy as np
import os
import skimage.io as io
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as k
import tensorflow as tf

def red():
    altura = 200  # Altura de la carta.
    anchura = 256  # Anchura de la carta.
    canales = 1  # Canales de la carta.
    input=Input((altura,anchura,canales))#Método para crear la red neuronal
    # Define las dimenciones del dato de entrada como una "tupla"
    # El método Conv2D tiene como primer argumento el número de canales de salida de la capa.
    # El segundo argumento puede ser una tupla o un entero que especifica las dimenciones horizontal y
    # vertical del filtro convolutivo. El número de canales del filtro coincidirá con los canales de la
    # capa anterior. El tercer argumento es la función de activación. En la documentación de Keras se
    # pueden consultar más funciones de activación. El cuarto argumento son los bordes. Aquí "same" especifica
    # que se le agreguen los bordes necesarios para que las dimenciones horizontal y vertical del dato
    # permanezcan igual. El quinto argumento son los valores iniciales de los parámetros de la capa.
    # he_normal es un método que da parámetros aleatorios. Más argumentos opcionales se pueden consultar en la
    # documentación oficial de Keras. Al final entre paréntesis se llama a la capa anterior, en este caso, los
    # datos de entrada, para construir el grafo.
    conv1 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(input)
    conv1 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv1)
    # El método Maxpooling2D contrae la imagen seleccionando el pixel de mayor intensidad. El argumento es el
    # tamaño de las ventanas. Al final entre paréntesis se llama a la capa que se le efectuará esta operación.
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool3)
    conv4 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv4)
    # El método UpSampling2D agranda la imagen. El argumento es el tamaño de ventanas en las que se irán copiando
    # los pixeles. Al final entre paréntesis se llama a la capa que se le efectuará la operación.
    up5 = UpSampling2D((2, 2))(conv4)
    up5 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(up5)
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge5)
    conv5 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv5)
    up6 = UpSampling2D((2, 2))(conv5)
    up6 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(up6)
    # concatenate es un método de numpy para concatenar arreglos. En el argumento se ponen, como arreglo de
    # arreglos los arreglos a concatenar. El siguiente argumento especifica el eje en el que se concatenarán
    # 3 corresponde al eje de los canales.
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge6)
    conv6 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv6)
    up7 = UpSampling2D((2, 2))(conv6)
    up7 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(up7)
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge7)
    conv7 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv7)
    conv7 = Conv2D(2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv7)
    conv8 = Conv2D(1, 1, activation="sigmoid")(conv7)
    # El método Model es el que creará el grafo a partir de los argumentos inputs que son la entrada,
    # Y outputs para la salida.
    modelo=Model(inputs=input, outputs=conv8)
    # La función debe arrogar el grafo del modelo para poder llamarlo.
    return modelo