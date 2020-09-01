import skimage.transform as trans
import skimage.io as io
import os
import numpy as np


def nombres(ruta,caracteres):
    """
    :param ruta (str): Carpeta de los archivos.
    :param caracteres (str): Número de caracteres de los nuevos nombre del archivo.
    :return: Regresa una lista de (str) con los nombres de los archivos recortados hasta el
    número de caracteres especificado.
    """
    archivos=os.listdir(ruta)
    archivo=[]
    for i in range(len(archivos)):
        archivo.append(archivos[i][0:caracteres])
    return archivo


def cortar(origen, destino, archivo, anchura=256, altura=256):
    """Copia y recorta la imagen en cartas de tamaño especificado. Si la imágen tiene dimenciones que
    no son múltiplos de la anchura y altura especificada, se generarán cartas de la altura y anchura
    especificadas, llenando los pixeles faltantes con ceros.

    Args:
        :param origen (str): Ruta de la carpeta de origen de la imagen.
        :param destino (str): Ruta de la carpeta destino de las cartas.
        :param archivo (str): Nombre de la imagen.
        :param anchura (int): Tamaño horizontal de la carta.
        :param altura (int): Tamaño vertical de la carta.
    """
    imgn = io.imread(origen + archivo + ".png", as_gray=True)
    imgn = np.array(imgn)
    vertical,horizontal=imgn.shape
    x= horizontal / anchura
    x=int(x)
    sobrax= horizontal - anchura * x
    y= vertical / altura
    y=int(y)
    sobray= vertical - altura * y
    faltax=0
    faltay=0
    if(sobrax>0):
        x=x+1
        faltax= anchura - sobrax
    if(sobray>0):
        y=y+1
        faltay= altura - sobray
    imgn=np.pad(imgn,((0,faltay),(0,faltax)),"median")
    for i in range(x):
        for j in range(y):
            img = imgn[j * altura:(j + 1) * altura, i * anchura:(i + 1) * anchura]
            io.imsave(destino+archivo+"x"+str(i)+"y"+str(j)+".png", img)

def reconstruir(origen,destino,archivo,anchura,altura):
    """Reconstruye una imagen a partir de sus cartas

    Args:
        :param origen (str): Ruta de la carpeta de origen de las cartas.
        :param destino (str): Ruta de la carpeta destino de la imagen.
        :param archivo (str): Prefijo de las cartas; también será el nombre de la imagen reconstruida.
        :param anchura (int): Tamaño horizontal de la imagen reconstruida.
        :param altura (int): Tamaño vertical de la imagen reconstruida.
    """
    #Toma una carta para conocer las dimensiones de las cartas:
    primera=io.imread(origen+archivo+"x" + str(0) + "y" + str(0) + ".png",as_gray=True)
    primera=np.array(primera)
    cartay, cartax = primera.shape
    reconstruccion=np.empty((altura,1))
    horizontal=anchura/cartax
    vertical=altura/cartay
    horizontal=int(horizontal)
    vertical=int(vertical)
    if(altura%vertical==0):
        vertical=vertical-1
    if(anchura%horizontal==0):
        horizontal=horizontal-1
    for i in range(horizontal+1):
        columna = np.empty((1, cartax))
        for j in range(vertical+1):
            img=io.imread(origen+archivo+"x" + str(i) + "y" + str(j) + ".png",as_gray=True)
            columna = np.concatenate((columna, img))
        reconstruccion=np.concatenate((reconstruccion,columna[1:]),axis=1)
    reconstruccion=reconstruccion[0:altura,0:anchura]
    io.imsave(destino+archivo+".png",reconstruccion)