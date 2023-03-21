import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv


def compare(image1, image2):

    #resize
    width1 = image1.shape[1]
    width2 = image2.shape[1]

    if width1 > width2:
        ratio = width2 / width1
        dim = (width2, int(image1.shape[0] * ratio))
        image1 = cv.resize(image1, dim)
    else:
        ratio = width1 / width2
        dim = (width1, int(image2.shape[0] * ratio))
        image2 = cv.resize(image2, dim)

    sumUnion = 0
    sumInter = 0
    for i in range(0, image1.shape[0]):
        for j in range(0, image1.shape[1]):
            if image1[i, j] != 0 or image2[i, j] != 0:
                sumUnion += 1
            if image1[i, j] != 0 and image2[i, j] != 0:
                sumInter += 1

    return sumInter / sumUnion
