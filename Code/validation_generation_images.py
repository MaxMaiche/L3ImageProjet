import json
import os
from time import time

import numpy as np
from PIL import Image


def getLabeling(nomImage: str):
    f = open('../Ressources/labeling/' + nomImage + '.json')
    # f = open('../Ressources/testsimple/petittest.json')
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
    x = point[0]
    y = point[1]

    try:
        if image[x - 1, y] == 0:
            neighbours.append((x - 1, y))
    except:
        pass
    try:
        if image[x + 1, y] == 0:
            neighbours.append((x + 1, y))
    except:
        pass
    try:
        if image[x, y - 1] == 0:
            neighbours.append((x, y - 1))
    except:
        pass
    try:
        if image[x, y + 1] == 0:
            neighbours.append((x, y + 1))
    except:
        pass

    # shuffle(neighbours)
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

    # try:
    #     for x in range(max_y-1, min_y-1, -1):
    #         cpt += 1
    #         if cpt % 100 == 0:
    #             print(((x - min_y) / (max_y - min_y) * 100), '%   x:', x, 'cpt:', cpt)
    #         for y in range(min_x, max_x):
    #             if image[x, y] == 0:
    #                 image[x, y] = castRay(image, (x, y), max_x)
    #
    # except KeyboardInterrupt:
    #     pass

    # for x in range(min_y, max_y):
    #     for y in range(min_x, max_x):
    #         if image[x, y] == 128:
    #             if getNeighbours(image, (x, y)):
    #                 image[x, y] = 0
    #             else:
    #                 image[x, y] = 255

    points = []

    y = min_x
    for x in range(min_y, max_y):
        points.extend(castRayRight(image, (x, y), max_x))

    y = max_x
    for x in range(min_y, max_y):
        points.extend(castRayLeft(image, (x, y), min_x))

    x = max_y
    for y in range(min_x, max_x):
        points.extend(castRayUp(image, (x, y), min_y))

    x = min_y
    for y in range(min_x, max_x):
        points.extend(castRayDown(image, (x, y), max_x))

    for p in points:
        image[p[0], p[1]] = 128

    for x in range(min_y, max_y + 1):
        for y in range(min_x, max_x + 1):
            if image[x, y] == 128:
                image[x, y] = 0
            elif image[x, y] == 0:
                image[x, y] = 254

    for x in range(min_y, max_y + 1):
        for y in range(min_x, max_x + 1):
            if image[x, y] == 254:
                if getNeighbours(image, (x, y)):
                    image[x, y] = 0

    return image


def castRayRight(image, point, max_x):
    x, y = point
    points = []
    while x < max_x:
        if image[x, y] == 255:
            return points
        points.append((x, y))
        y += 1

    return points


def castRayLeft(image, point, min_x):
    x, y = point
    points = []
    while x > min_x:
        if image[x, y] == 255:
            return points
        points.append((x, y))
        y -= 1

    return points


def castRayUp(image, point, min_y):
    x, y = point
    points = []
    while y > min_y:
        if image[x, y] == 255:
            return points
        points.append((x, y))
        x -= 1

    return points


def castRayDown(image, point, max_y):
    x, y = point
    points = []
    while y < max_y:
        if image[x, y] == 255:
            return points
        points.append((x, y))
        x += 1

    return points


def castRay(image, point, max_x):
    nbInter = 0
    x, y = point
    while x < max_x:
        if image[x, y] == 255:
            nbInter += 1
            while image[x, y] == 255 and x < max_x:
                x += 1

        if x < max_x - 1 and image[x + 1, y] == 128:
            return 128

        x += 1

    if nbInter % 2 == 1:
        return 128
    else:
        return 0


def unionBinaryImage(image1, image2, height, width):
    for x in range(height):
        for y in range(width):
            if image1[x, y] >= 128 or image2[x, y] >= 128:
                image1[x, y] = 255
            else:
                image1[x, y] = 0

    return image1


# def getAllLines(image):

def main():
    folder_path = '../Ressources/labeling'
    allowed_extensions = '.json'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(allowed_extensions):
            board = []
            lignes = []
            schema = []

            print(filename)
            filename = filename.split('.')[0]

            height, width, b, l, s = getLabeling(filename)
            print(height, width)
            timeStart = time()
            for i in range(len(b)):
                board.append(pixelinpolygones(height, width, b[i]))

            for i in range(len(l)):
                lignes.append(pixelinpolygones(height, width, l[i]))

            for i in range(len(s)):
                schema.append(pixelinpolygones(height, width, s[i]))

            if len(board) == 0:
                board.append(np.zeros((height, width), dtype=np.uint8))
            else:
                boardImage = board[0]
                for i in range(1, len(board)):
                    boardImage = unionBinaryImage(boardImage, board[i], height, width)

            if len(lignes) == 0:
                lignes.append(np.zeros((height, width), dtype=np.uint8))
            else:
                lignesImage = lignes[0]
                for i in range(1, len(lignes)):
                    lignesImage = unionBinaryImage(lignesImage, lignes[i], height, width)

            if len(schema) == 0:
                schema.append(np.zeros((height, width), dtype=np.uint8))
            else:
                schemaImage = schema[0]
                for i in range(1, len(schema)):
                    schemaImage = unionBinaryImage(schemaImage, schema[i], height, width)

            timeEnd = time()
            print('time: ', timeEnd - timeStart)
            if not os.path.exists("../Validation/labeling/" + filename):
                os.makedirs("../Validation/labeling/" + filename)

            im = Image.fromarray(boardImage)
            im = im.convert('RGB')
            im.save("../Validation/labeling/" + filename + "/board.png")

            im = Image.fromarray(lignesImage)
            im = im.convert('RGB')
            im.save("../Validation/labeling/" + filename + "/lignes.png")

            im = Image.fromarray(schemaImage)
            im = im.convert('RGB')
            im.save("../Validation/labeling/" + filename + "/schema.png")


if __name__ == '__main__':
    main()
