import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import json


def getLabeling(nomImage: str):
    f = open('../Ressources/labeling/' + nomImage + '.json')
    data = json.load(f)
    shapes = data['shapes']

    board = []
    lignes = []
    schema = []

    for i in range(len(shapes)):
        s = shapes[i]
        if s['label'] == 'Board':
            board.append(s['points'])
        elif s['label'] == 'Ligne':
            lignes.append(s['points'])
        elif s['label'] == 'Schema':
            schema.append(s['points'])

    print(data)
    return data['imageHeight'], data['imageWidth'], board, lignes, schema


def getpixelSegment(point1, point2):
    x1 = point1[0]
    x2 = point2[0]
    y1 = point1[1]
    y2 = point2[1]

    pixelinSegment = []

    # Bresenham's line algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        if x1 >= 0 and y1 >= 0:
            pixelinSegment.append((x1, y1))

        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err = err - dy
            x1 = x1 + sx
        if e2 < dx:
            err = err + dx
            y1 = y1 + sy

    return pixelinSegment


def castRay(image, point, max_x):

    nbInter = 0

    x, y = point
    while x <= max_x:

        if image[x, y] == 1:
            nbInter += 1
        x += 1

    if nbInter % 2 == 1:
        return 255
    else:
        return 0


def pixelinpolygones(height, width, polygone):
    image = np.zeros((height, width))
    # find max and min
    max_x = 0
    max_y = 0
    min_x = width
    min_y = height

    for p in polygone:
        p[0] = int(p[0])
        p[1] = int(p[1])

        if p[0] > max_x:
            max_x = p[0]
        if p[1] > max_y:
            max_y = p[1]
        if p[0] < min_x:
            min_x = p[0]
        if p[1] < min_y:
            min_y = p[1]

    contours = []
    for i in range(1, len(polygone)):
        contours += getpixelSegment(polygone[i - 1], polygone[i])

    contours += getpixelSegment(polygone[0], polygone[len(polygone) - 1])

    for p in contours:
        image[p[0], p[1]] = 255

    for i in range(int(min_x), int(max_x)):
        for j in range(int(min_y), int(max_y)):
            if image[i, j] == 0:
                image[i, j] = castRay(image, (i, j), max_x)

    return image


height, width, b, l, s = getLabeling('0')
image = pixelinpolygones(height, width, b[0])
plt.imshow(image, cmap='gray')
plt.show()

print(len(b))
print(len(l))
print(len(s))
print(b)


def getNbPixelBlanc(imgBin):
    nbPixelB = 0
    for i in range(0, imgBin.shape[0]):
        for j in range(0, imgBin.shape[1]):
            if imgBin[i, j] == 1:
                nbPixelB += 1
    return nbPixelB

# fait un truc pour faire l'union des deux et l'inter des deux images binaires
# entre json et le png/jpg ?
