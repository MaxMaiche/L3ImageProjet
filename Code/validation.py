import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json


def getLabeling(nomImage: str) -> tuple[list[tuple[float]]]:
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

    return board, lignes, schema


b, l, s = getLabeling('0')

print(len(b))
print(len(l))
print(len(s))
