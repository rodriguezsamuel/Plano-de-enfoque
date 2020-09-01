import datos as datos
import postprocesamiento as pp
from wnet import *
import os
import skimage.io as io
import time

ruta_imagenes="./train/image/" #Ruta imágenes de entrenamiento.
ruta_mascaras="./train/mask/" #Ruta máscaras de entrenamiento.
ruta_test="./test/image/" #Ruta imágenes de prueba.
ruta_testmasc="./test/mask/" #Ruta máscaras para comparar resultados.
ruta_predicciones="./test/pred/" #Ruta de las predicciones.
corta_imagenes="./train/crop/image/" #Ruta de las cartas de las imágenes de entrenamiento.
corta_mascaras="./train/crop/mask/" #Ruta de las cartas de las máscaras de entrenamiento.
corta_test="./test/crop/image/" #Ruta de las cartas de las imágenes de prueba.
corta_testmasc="./test/crop/mask/" #Ruta de las cartas de las máscaras de prueba.
corta_predicciones="./test/crop/pred/" #Ruta de las cartas predichas.
depurado="./test/depurado/"#Ruta de las imágenes depuradas.
superpon="./test/s/"#Ruta de la superposición de las imágenes con las predicciones.

def renombrar(ruta):
    lista=os.listdir(ruta)
    for i in lista:
        n=len(i)
        os.rename(ruta+i,ruta+i[n-9:n-3]+"png")

def cortar(): #Función auxiliar para recortar todas las imágenes y máscaras.
    archivos=datos.nombres(ruta_imagenes,5)
    for i in archivos:
        datos.cortar(ruta_imagenes,corta_imagenes,i,altura=200)
        datos.cortar(ruta_mascaras,corta_mascaras,i,altura=200)

    archivos=datos.nombres(ruta_test,5)
    for i in archivos:
        datos.cortar(ruta_test,corta_test,i,altura=200)
        datos.cortar(ruta_testmasc,corta_testmasc,i,altura=200)

def entrenar():
    X_entrenamiento=[] #Lote de entradas de entrenamiento.
    Y_entrenamiento=[] #Lote de salidas de entrenamiento.
    x_nombres=os.listdir(corta_imagenes) #Arreglo de nombres de las cartas.
    for i in x_nombres:
        X_entrenamiento.append(io.imread(corta_imagenes+i,as_gray=True))
        Y_entrenamiento.append(io.imread(corta_mascaras+i,as_gray=True))

    #Tensorización de las entradas y salidas de entrenamiento.
    X_entrenamiento=np.array(X_entrenamiento)
    Y_entrenamiento=np.array(Y_entrenamiento)

    #Normalización de las entradas y salidas de entrenamiento.
    X_entrenamiento=X_entrenamiento/255.
    Y_entrenamiento=Y_entrenamiento/255.

    modelo = red() #Creación del grafo de la red neuronal
    # Compilación del grafo de la red neurnal
    modelo.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    #modelo.load_weights("unet.hdf5") #Carga pesos si es que ya hubo un entrenamiento previo.
    modelo.summary() #Muestra el resumen de las capas de la red neuronal.
    #Función para mostrar la ejecución del entrenamiento.
    model_checkpoint = ModelCheckpoint("wnet.hdf5", monitor="loss", verbose=1, save_best_only=True,save_freq="epoch")
    #Para evitar el calenamiento del ordenador, el conjunto se parte en dos, y las épocas de entrenamiento se
    #indican fuera de la función de ajuste para poder suspender el entrenamiento unos minutos.
    for i in range(5):
        n=len(X_entrenamiento)
        randomizar=np.arange(n) #Se crea un arreglo de índices para los conjuntos de entrenamiento.
        np.random.shuffle(randomizar) #Se reordena aleatoriamente
        X_entrenamiento=X_entrenamiento[randomizar] #Se permutan las imágenes.
        Y_entrenamiento=Y_entrenamiento[randomizar] #Se permutan las máscaras.
        for j in range(2):
            modelo.fit(x=X_entrenamiento[j*900:(j+1)*900], y=Y_entrenamiento[j*900:(j+1)*900], epochs=1, batch_size=5, callbacks=[model_checkpoint],
                       shuffle=True)
            modelo.save_weights("wnet.hdf5") #Guarda los pesos.
            time.sleep(100) #Suspención del entrenamiento.
        # modelo.save("mod.h5") #El modelo y pesos de la red neuronal pueden ser guardados y cargados.

def testear(): #Función para ejecutar la prueba de la red.
    x_test_nombres = os.listdir(corta_test) #Arreglo con los nombres de las cartas de prueba.
    X_test = [] #Lote de datos de entrada.

    for i in x_test_nombres:
        X_test.append(io.imread(corta_test + i, as_gray=True))

    X_test = np.array(X_test) #Tensorización.
    X_test = X_test / 255. #Normalización.
    modelo = red() #Creación del grafo de la red neuronal.
    #Compilación de la red neuronal.
    modelo.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    modelo.load_weights("wnet.hdf5") #Carga los pesos obtenidos en el entrenamiento.
    resultados = modelo.predict(X_test, batch_size=5,verbose=1) #Lote de predicciones.
    #Binarización de la imagen.
    """resultados=np.array(resultados)
    resultados=resultados+0.1 
    resultados=resultados.astype(int)"""
    for j, item in enumerate(resultados):
        imgen = item[:, :, 0] * 255
        io.imsave("./test/crop/pred/" + x_test_nombres[j], imgen) #Guarda el lote de predicciones como imágenes.

renombrar(ruta_imagenes)
renombrar(ruta_mascaras)
renombrar(ruta_test)
renombrar(ruta_testmasc)
cortar()
entrenar()#Entrenamiento de la red.
testear()#Prueba de la red.
archivos=datos.nombres(ruta_test,5) #Arreglo de los nombres de las imágenes de entrada.
for i in archivos:
    datos.reconstruir(corta_predicciones,ruta_predicciones,i,1280,800) #Generación de las máscaras a partir de sus cartas.

pp.depurar(ruta_predicciones,depurado,0.3,20,0.9)
pp.superponer(ruta_test,depurado,superpon,ruta_testmasc)