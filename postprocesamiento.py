import numpy as np
import skimage.io as io
from skimage import measure,color
import os as os
#Skimage es una paquetería para el procesamiento de imágenes.

def depurar(ruta,destino,umbral_de_binarizacion,area_minima,criterio_de_convexidad,mostrar_propiedades=False):
    """Depurado de los resultados obtenidos por la red neuronal

        :param ruta: (str)Ruta de la carpeta origen de los resultados por depurar.
        :param destino: (str)Ruta de la carpeta destino para guardar los resultados depurados.
        :param umbral_de_binarizacion: (float)Umbral <1 que se le suma al valor de pixel que se encuentra
        entre [0,1] para posteriormente aplicar función piso.
        :param area_minima: (int)Área mínima en pixeles que deben tener las regiones. Cualquier región con
        área menor, será descartada.
        :param criterio_de_convexidad: (float)Umbral <1 de la relación entre el área de la región con el
        área del polígono convexo mínimo que contenga la región.
        :param mostrar_propiedades: (bool)Muestra las propiedades de cada región

    """
    archivos=os.listdir(ruta)
    for i in archivos:
        image=io.imread(ruta+i,as_gray=True)
        image=np.array(image)
        image=image/255+umbral_de_binarizacion
        image=image.astype(int) #Binarización de la imagen
        etiquetado=measure.label(image) #Etiqueta regiones disconexas
        regions=measure.regionprops(etiquetado) #Encuentra propiedades de éstas regiones.
        for prop in regions: #Analiza cada región.
            convexidad=prop.area/prop.convex_area
            #Descarta las partículas por convexidad y tamaño:
            if(prop.area<area_minima or convexidad<criterio_de_convexidad):
                coordenadas = np.array(prop.coords)
                image[coordenadas[:,0],coordenadas[:,1]]=0
            else:
                if(mostrar_propiedades==True):
                    print("centroides:")
                    print(prop.centroid)
                    print("area:")
                    print(prop.area)
                    print("area convexa:")
                    print(prop.convex_area)
                    print("coeficiente de convexidad")
                    print(convexidad)
        io.imsave(destino + i, image)

def superponer(ruta_imagenes,ruta_predicciones,ruta_guardado,ruta_mascaras=""):
    """Superpone la imagen sin procesar, la predicción y opcionalmente, la máscara de evaluación.
    Las predicciones y máscaras deben tener los mismos nombres que las imágenes.

    :param ruta_imagenes: (str) Ruta de las imágenes sin procesar.
    :param ruta_predicciones: (str) Ruta de las predicciones.
    :param ruta_guardado: (str) Ruta de guardado de las imagenes superpuestas.
    :param ruta_mascaras: (str) (opcional) Ruta de las máscaras de evaluación.
    """
    archivos=os.listdir(ruta_imagenes)
    for i in archivos:
        imagen=io.imread(ruta_imagenes+i,as_gray=True)
        imagen=color.gray2rgb(imagen)
        imagen = np.array(imagen)
        p = io.imread(ruta_predicciones + i)
        p = np.array(p)
        prediccion=np.zeros((800,1280))
        prediccion[:,:]=p
        q = np.zeros((800, 1280, 3))
        if(ruta_mascaras==""): #Si no hay máscara, las predicciones aparecerán en azul
            q[:,:,2]=prediccion
        else: #Si hay máscara de evaluación, los verdaderos positivos aparecerán en verde,
              #los falsos positivos en azul, y los falsos negativos en rojo.
            mascara=io.imread(ruta_mascaras+i)
            mascara=np.array(mascara)
            mascara=mascara[:,:,0]
            q[:,:,1]=np.where(prediccion+mascara>255,255,0)
            q[:,:,0]=np.where(prediccion-mascara<0,255,0)
            q[:, :, 2] = np.where(mascara-prediccion < 0, 255, 0)
            imagen=imagen/2+q/2
            io.imsave(ruta_guardado+i,imagen)