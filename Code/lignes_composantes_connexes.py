import math
import random

import cv2 as cv
import numpy as np

bordure_width = 40
bordure_height = 0


def get_lines(board, base_image, M):
    board_edges = cv.Canny(board, 50, 150, apertureSize=3)

    # kernel = np.ones((2, 2), np.uint8)
    # board_edges = cv.dilate(board_edges, kernel, iterations=1)
    # board_edges = cv.erode(board_edges, kernel, iterations=1)

    cv.imwrite('Resultats/board_edges.jpg', board_edges)

    composantes_connexes = get_composantes_connexes(board_edges, board.copy())
    lignes = get_lignes(composantes_connexes, board.copy())

    # On affiche les lignes sur l'image d'output
    board_lines_chains = board.copy()
    for ligne in lignes:
        couleur = (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255))
        for composante_connexe in ligne:
            x, y, w, h, surface = composante_connexe
            cv.rectangle(board_lines_chains, (x, y), (x + w, y + h), couleur, 2)
    cv.imwrite('Resultats/board_lines_chains.jpg', board_lines_chains)

    lignes_simples = simplifier_lignes(lignes)

    # On affiche les lignes sur l'image d'output
    board_lines = board.copy()
    for ligne in lignes_simples:
        x1, y1, x2, y2 = ligne
        cv.rectangle(board_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv.imwrite('Resultats/board_lines.jpg', board_lines)

    base_lines = base_image.copy()

    # On crée une image binaire avec les lignes
    base_lines_binary = np.zeros((base_image.shape[0], base_image.shape[1]), dtype=np.uint8)

    for ligne in lignes_simples:
        x1, y1, x2, y2 = ligne
        points_board = (np.array([[x1, y1]], dtype=np.float32).reshape(-1, 1, 2),
                        np.array([[x2, y1]], dtype=np.float32).reshape(-1, 1, 2),
                        np.array([[x2, y2]], dtype=np.float32).reshape(-1, 1, 2),
                        np.array([[x1, y2]], dtype=np.float32).reshape(-1, 1, 2))

        # pointsDetransformes = [np.matmul(inverse_M, np.array([p[0], p[1], 1])) for p in pointsTransformes]
        # pointsOriginaux = [(int(p[0]), int(p[1])) for p in pointsDetransformes]

        points_base = [cv.perspectiveTransform(p, np.linalg.inv(M)).reshape(-1, 2) for p in points_board]
        points_base = [(int(p[0][0]), int(p[0][1])) for p in points_base]

        cv.fillConvexPoly(base_lines_binary, np.array(points_base), 255)
        cv.drawContours(base_lines, [np.array(points_base)], 0, (0, 0, 255), 2)

    cv.imwrite('Resultats/base_lines_binary.jpg', base_lines_binary)
    cv.imwrite('Resultats/base_lines.jpg', base_lines)

    return base_lines, base_lines_binary


def get_composantes_connexes(img, img_output, tolerance=0.1):
    nb_composantes, labels, stats, centroids = cv.connectedComponentsWithStats(img, 8, cv.CV_32S)

    # On affiche les composantes connexes sur l'image d'output
    for icomposante in range(1, nb_composantes):
        x, y, w, h, surface = stats[icomposante]
        if w > tolerance * img.shape[1] or h > tolerance * img.shape[0]:
            stats[icomposante] = [0, 0, 0, 0, 0]
            continue
        stats[icomposante] = [x - bordure_width, y - bordure_height, w + 2 * bordure_width, h + 2 * bordure_height,
                              surface]
        x, y, w, h, surface = stats[icomposante]
        cv.rectangle(img_output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imwrite('Resultats/board_composantes_connexes.jpg', img_output)

    stats = [x for x in stats if x[0] > 0 and x[1] > 0]

    return stats


def get_lignes(composantes_connexes, img):
    composantes_connexes.sort(key=lambda x: x[0])
    lignes = []

    while len(composantes_connexes) > 0:

        start_cc = composantes_connexes.pop(0)
        x, y, w, h, surface = start_cc

        ligne = enchainement(start_cc, composantes_connexes, img, [])

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


def enchainement(current_cc, composantes_connexes, img, ligne, icc=0):
    x, y, w, h, surface = current_cc

    if len(ligne) == 0:
        ligne.append(current_cc)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.imwrite('Resultats/board_cc_racines.jpg', img)

    for icc2 in range(icc + 1, len(composantes_connexes)):
        if icc2 >= len(composantes_connexes):
            break
        if enchaine(current_cc, composantes_connexes[icc2]):
            if icc == 0:
                x2, y2, w2, h2, surface2 = composantes_connexes[icc2]
                cv.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)
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

        ligne_simple[0] += bordure_width
        ligne_simple[1] += bordure_height
        ligne_simple[2] -= bordure_width
        ligne_simple[3] -= bordure_height
        lignes_simples.append(ligne_simple)

    return lignes_simples
