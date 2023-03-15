import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
from PIL import Image

def getLabeling(nomImage: str):
    #f = open('../Ressources/labeling/' + nomImage + '.json')
    f = open('../Ressources/testsimple/petittest.json')
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
        image[p[1], p[0]] = 255

    for x in range(max_y-1, min_y-1, -1):
        for y in range(min_x, max_x):
            if image[x, y] == 0:
                image[x, y] = castRay(image, (x, y), max_x)


    return image


def castRay(image, point, max_x):

    nbInter = 0
    x, y = point
    x_init = x

    while x < max_x:
        if image[x, y] == 255:
            nbInter += 1
            while image[x, y] == 255:
                x += 1

        x += 1

    if nbInter % 2 == 1:
        return 254
    else:
        return 0


height, width, b, l, s = getLabeling('33')
image = pixelinpolygones(height, width, b[0])
im = Image.fromarray(image)
im = im.convert('RGB')
im.save("resultat_Validation.png")

print(len(b))
print(len(l))
print(len(s))

