import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def getNbPixelBlanc(imgBin):
    nbPixelB = 0
    for i in range(0, imgBin.shape[0]):
        for j in range(0, imgBin.shape[1]):
            if imgBin[i, j] == 1:
                nbPixelB += 1
    return nbPixelB

# fait un truc pour faire l'union des deux et l'inter des deux images binaires
# entre json et le png/jpg ?
