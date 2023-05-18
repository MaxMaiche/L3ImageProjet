import math
import os
import random

import cv2 as cv
import numpy as np

from Code import validation
from main import getEdges, getBoardLines, getIntersection

# sys.setrecursionlimit(100000)

bordureW = 30
bordureH = 10


def get_lignes_rectangles(board):
    board_edges = cv.Canny(board, 50, 150, apertureSize=3)

    # kernel = np.ones((2, 2), np.uint8)
    # board_edges = cv.dilate(board_edges, kernel, iterations=1)
    # board_edges = cv.erode(board_edges, kernel, iterations=1)

    cv.imwrite('Resultats/board_edges.jpg', board_edges)
    cc = get_composantes_connexes(board_edges, board.copy())
    lignes = get_lignes(cc, board.copy())

    # On affiche les lignes sur l'image d'output
    out = board.copy()
    for ligne in lignes:
        couleur = (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255))
        for cc in ligne:
            x, y, w, h, surface = cc
            cv.rectangle(out, (x, y), (x + w, y + h), couleur, 2)
    cv.imwrite('Resultats/lignesconnexes.jpg', out)

    lignes_simples = simplifier_lignes(lignes)

    # On affiche les lignes sur l'image d'output
    for ligne in lignes_simples:
        x1, y1, x2, y2 = ligne
        cv.rectangle(board, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv.imwrite('Resultats/lignes.jpg', board)

    # On crée une image binaire avec les lignes
    lignes_bin = np.zeros((board.shape[0], board.shape[1]), dtype=np.uint8)
    for ligne in lignes_simples:
        x1, y1, x2, y2 = ligne
        cv.rectangle(lignes_bin, (x1, y1), (x2, y2), (255, 255, 255), -1)

    cv.imwrite('Resultats/lignes_bin.jpg', lignes_bin)


def get_composantes_connexes(img, img_output, tolerance=0.1):
    nb_composantes, labels, stats, centroids = cv.connectedComponentsWithStats(img, 8, cv.CV_32S)

    # On affiche les composantes connexes sur l'image d'output
    for icomposante in range(1, nb_composantes):
        x, y, w, h, surface = stats[icomposante]
        if w > tolerance * img.shape[1] or h > tolerance * img.shape[0]:
            stats[icomposante] = [0, 0, 0, 0, 0]
            continue
        stats[icomposante] = [x - bordureW, y - bordureH, w + 2 * bordureW, h + 2 * bordureH, surface]
        x, y, w, h, surface = stats[icomposante]
        cv.rectangle(img_output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imwrite('Resultats/composantes_connexes.jpg', img_output)

    stats = [x for x in stats if x[0] > 0 and x[1] > 0]

    return stats


def get_lignes(composantes_connexes, img):
    composantes_connexes.sort(key=lambda x: x[0])
    lignes = []

    while len(composantes_connexes) > 0:

        cc = composantes_connexes.pop(0)
        x, y, w, h, surface = cc

        ligne = enchainement(cc, composantes_connexes, img, [])

        if len(ligne) > 3:
            lignes.append(ligne)

    return lignes


def enchaine(cc1, cc2, tolerance=0):
    x1, y1, w1, h1, surface1 = cc1
    x2, y2, w2, h2, surface2 = cc2
    if x1 + w1 < x2:
        return False
    dm = y1 + h1 / 2
    dh, db = dm - h1 * tolerance, dm + h1 * tolerance
    if y2 + h2 < dh or y2 > db:
        return False
    return True


def enchainement(cc, composantes_connexes, img, ligne=[], icc=0):
    x, y, w, h, surface = cc

    if len(ligne) == 0:
        ligne.append(cc)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.imwrite('Resultats/CC1.jpg', img)

    for icc2 in range(icc + 1, len(composantes_connexes)):
        if icc2 >= len(composantes_connexes):
            break
        if enchaine(cc, composantes_connexes[icc2]):
            if icc == 0:
                x2, y2, w2, h2, surface2 = composantes_connexes[icc2]
                cv.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)
                cv.imwrite('Resultats/CC1.jpg', img)
            ligne.append(composantes_connexes[icc2])
            composantes_connexes.pop(icc)
            np.concatenate((ligne,
                            enchainement(composantes_connexes[icc2 - 1], composantes_connexes, img, ligne, icc2 - 1)))

    return ligne


def simplifier_lignes(lignes):
    lignes_simples = []
    for ligne in lignes:
        ligne_simple = [math.inf, math.inf, 0, 0]
        for cc in ligne:
            x, y, w, h, surface = cc
            if x < ligne_simple[0]:
                ligne_simple[0] = x
            if y < ligne_simple[1]:
                ligne_simple[1] = y
            if x + w > ligne_simple[2]:
                ligne_simple[2] = x + w
            if y + h > ligne_simple[3]:
                ligne_simple[3] = y + h

        ligne_simple[0] += bordureW
        ligne_simple[1] += bordureH
        ligne_simple[2] -= bordureW
        ligne_simple[3] -= bordureH
        lignes_simples.append(ligne_simple)

    return lignes_simples


def main_test(nomImage):
    # Import image
    # nomImage = input("Nom de l'image : ")
    # img = cv.imread("../Ressources/Images/" + nomImage)
    img = cv.imread("../Ressources/Images/" + nomImage)

    # Resize image
    width = img.shape[1]
    height = img.shape[0]
    ratio = 1024 / width
    dim = (1024, int(height * ratio))

    img = cv.resize(img, dim)

    # Make image grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Get the edges
    edges = getEdges(img)

    # Get the lines
    lineTop, lineBottom, lineLeft, lineRight = getBoardLines(img, edges, minWidth=675)

    # Get the corners of the board
    ptTopLeft = getIntersection(lineTop, lineLeft)
    ptTopRight = getIntersection(lineTop, lineRight)
    ptBottomLeft = getIntersection(lineBottom, lineLeft)
    ptBottomRight = getIntersection(lineBottom, lineRight)

    # Apply perspective transform
    pts1 = np.float32([ptTopLeft, ptTopRight, ptBottomRight, ptBottomLeft])
    ratio = (((ptTopRight[0] - ptTopLeft[0]) + (ptBottomRight[0] - ptBottomLeft[0])) / 2) / (
            ((ptBottomLeft[1] - ptTopLeft[1]) + (ptBottomRight[1] - ptTopRight[1])) / 2)

    boardW = 1024
    boardH = int(boardW / ratio)

    pts2 = np.float32([[0, 0], [boardW, 0], [boardW, boardH], [0, boardH]])

    M = cv.getPerspectiveTransform(pts1, pts2)

    transformed = np.zeros((int(boardW), int(boardH)), dtype=np.uint8)
    board = cv.warpPerspective(img, M, transformed.shape)

    # Save images
    cv.imwrite('Resultats/gray.jpg', gray)
    cv.imwrite('Resultats/edges.jpg', edges)
    cv.imwrite('Resultats/transformed.jpg', board)

    board_edges = cv.Canny(board, 50, 150, apertureSize=3)

    # kernel = np.ones((2, 2), np.uint8)
    # board_edges = cv.dilate(board_edges, kernel, iterations=1)
    # board_edges = cv.erode(board_edges, kernel, iterations=1)

    cv.imwrite('Resultats/board_edges.jpg', board_edges)
    cc = get_composantes_connexes(board_edges, board.copy())
    lignes = get_lignes(cc, board.copy())

    # On affiche les lignes sur l'image d'output
    out = board.copy()
    for ligne in lignes:
        couleur = (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255))
        for cc in ligne:
            x, y, w, h, surface = cc
            cv.rectangle(out, (x, y), (x + w, y + h), couleur, 2)
    cv.imwrite('Resultats/lignesconnexes.jpg', out)

    lignes_simples = simplifier_lignes(lignes)

    imglignes = board.copy()
    # On affiche les lignes sur l'image d'output
    for ligne in lignes_simples:
        x1, y1, x2, y2 = ligne
        cv.rectangle(imglignes, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv.imwrite('Resultats/lignes.jpg', imglignes)

    imgLignesOriginales = img.copy()

    # On crée une image binaire avec les lignes
    lignes_bin = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for ligne in lignes_simples:
        x1, y1, x2, y2 = ligne
        pointsTransformes = np.array([[x1, y1]], dtype=np.float32).reshape(-1, 1, 2), np.array([[x2, y1]], dtype=np.float32).reshape(-1, 1, 2), np.array([[x2, y2]], dtype=np.float32).reshape(-1, 1, 2), np.array([[x1, y2]], dtype=np.float32).reshape(-1, 1, 2)
        # pointsDetransformes = [np.matmul(inverse_M, np.array([p[0], p[1], 1])) for p in pointsTransformes]
        # pointsOriginaux = [(int(p[0]), int(p[1])) for p in pointsDetransformes]
        pointsOriginaux = [cv.perspectiveTransform(p, np.linalg.inv(M)).reshape(-1, 2) for p in pointsTransformes]
        pointsOriginaux = [(int(p[0][0]), int(p[0][1])) for p in pointsOriginaux]
        cv.fillConvexPoly(lignes_bin, np.array(pointsOriginaux), 255)
        cv.drawContours(imgLignesOriginales, [np.array(pointsOriginaux)], 0, (0, 0, 255), 2)

    cv.imwrite('Resultats/lignes_bin.jpg', lignes_bin)
    cv.imwrite('Resultats/lignes_originales.jpg', imgLignesOriginales)

    nomImage = nomImage.split(".")[0]
    valid = cv.imread("../Validation/labeling/" + nomImage + "/lignes.png", cv.IMREAD_GRAYSCALE)

    score = validation.compare(valid, lignes_bin)
    print(f"Image : {nomImage}")
    print("Score : {:.2f}%".format(score * 100))
    print(f"Validation {'réussie' if score > 0.9 else 'échouée'}")
    if score > 0.9:
        return 1
    return 0


# main_test("35.jpg")

def doALL():
    folder_path = '../Ressources/Images'
    allowed_extensions = '.jpg', '.png', '.jpeg'
    cpt = 0
    cptTotal = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(allowed_extensions):
            if os.path.exists("../Validation/labeling/" + filename.split(".")[0]):
                cpt += main_test(filename)
                cptTotal += 1
                print()

    print("Score total : ", cpt, "/", cptTotal)


main_test("0.jpg")
#doALL()
