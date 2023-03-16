import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
from PIL import Image
from time import time
from random import shuffle, randint
def getLabeling(nomImage: str):
    f = open('../Ressources/labeling/' + nomImage + '.json')
    #f = open('../Ressources/testsimple/petittest.json')
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


def getNeighbours(image, point):
    neighbours = []
    x=point[0]
    y=point[1]

    try:
        if image[x-1, y] == 0:
            neighbours.append((x-1, y))
    except:
        pass
    try:
        if image[x+1, y] == 0:
            neighbours.append((x+1, y))
    except:
        pass
    try:
        if image[x, y-1] == 0:
            neighbours.append((x, y-1))
    except:
        pass
    try:
        if image[x, y+1] == 0:
            neighbours.append((x, y+1))
    except:
        pass

    #shuffle(neighbours)
    return neighbours


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

    #mean of the contour
    mean_x = (max_x+min_x)/2
    mean_y = (max_y+min_y)/2
    print(mean_x, mean_y)

    pixelQueue = [(int(mean_y), int(mean_x), True, True)]
    cpt = 0

    try:
        while len(pixelQueue) > 0:
            cpt += 1
            #if cpt % 1000 == 0:
                #print(len(pixelQueue), 'pixels left')

            x, y, hory, verti = pixelQueue.pop(0)
            pixel = (x, y)
            if hory:
                pixelQueue.append(castRayTop(image, pixel))
                pixelQueue.append(castRayBottom(image, pixel))
            if verti:
                pixelQueue.append(castRayLeft(image, pixel))
                pixelQueue.append(castRayRight(image, pixel))

    except KeyboardInterrupt:
        pass

    # try:
    #     for x in range(max_y-1, min_y-1, -1):
    #         print(((x-min_y)/(max_y-min_y)*100), '%')
    #         for y in range(min_x, max_x):
    #             if image[x, y] == 0:
    #                 image[x, y] = castRay(image, (x, y), max_x)
    #
    # except KeyboardInterrupt:
    #     pass
    #
    # for x in range(min_x, max_x):
    #     for y in range(min_x, max_x):
    #         if image[x, y] == 128:
    #             image[x, y] = 255
    return image

def castRayTop(image, point):
    x, y = point
    x_init = x

    try:
        while image[x-1, y] != 255:
            x -= 1
            image[x, y] = 128
    except:
        pass
    x += (x_init-x) // 2
    return x, y, False, True

def castRayBottom(image, point):
    x, y = point
    x_init = x
    try:
        while image[x+1, y] != 255:
            x += 1
            image[x, y] = 128
    except:
        pass
    x +=( x_init-x) // 2
    return x, y, False, True

def castRayLeft(image, point):
    x, y = point
    y_init = y
    try:
        while image[x, y-1] != 255:
            y -= 1
            image[x, y] = 128
    except:
        pass
    y += (y_init-y) // 2
    return x, y, True, False

def castRayRight(image, point):
    x, y = point
    y_init = y
    try:
        while image[x, y+1] != 255:
            y += 1
            image[x, y] = 128
    except:
        pass
    y += (y_init-y) // 2
    return x, y, True, False
def castRay(image, point, max_x):

    nbInter = 0
    x, y = point
    while x < max_x:
        if image[x, y] == 255:
            nbInter += 1
            while image[x, y] == 255 and x < max_x:
                x += 1

        if x < max_x-1 and image[x+1, y] == 128:
            return 128

        x += 1

    if nbInter % 2 == 1:
        return 128
    else:
        return 0


height, width, b, l, s = getLabeling('33')
timeStart = time()
image = pixelinpolygones(height, width, b[1])
timeEnd = time()
print('time: ', timeEnd - timeStart)
im = Image.fromarray(image)
im = im.convert('RGB')
im.save("resultat_Validation.png")

print(len(b))
print(len(l))
print(len(s))

