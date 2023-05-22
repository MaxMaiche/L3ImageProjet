import cv2 as cv
import numpy as np

from Code.lignes_composantes_connexes import get_composantes_connexes

horizontal = 10
vertical = 10


def get_lines(board, base_image, M):
    board_edges = cv.Canny(board, 50, 150, apertureSize=3)

    # kernel = np.ones((2, 2), np.uint8)
    # board_edges = cv.dilate(board_edges, kernel, iterations=1)
    # board_edges = cv.erode(board_edges, kernel, iterations=1)

    cv.imwrite('Resultats/board_edges.jpg', board_edges)

    image_rlsa = rlsa(board_edges, 10, 10)
    cv.imwrite('Resultats/board_rlsa.jpg', image_rlsa)

    composantes_connexes = get_composantes_connexes(image_rlsa, board.copy())

    lignes = []
    for composante_connexe in composantes_connexes:
        x, y, w, h, surface = composante_connexe
        if x + w < board.shape[1] - 1:
            w -= horizontal
        if y + h < board.shape[0] - 1:
            h -= vertical
        points = (x, y, x + w, y + h)
        lignes.append(points)

    # On affiche les lignes sur l'image d'output
    board_lines = board.copy()
    for ligne in lignes:
        x1, y1, x2, y2 = ligne
        cv.rectangle(board_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv.imwrite('Resultats/board_lines.jpg', board_lines)

    base_lines = base_image.copy()

    # On crée une image binaire avec les lignes
    base_lines_binary = np.zeros((base_image.shape[0], base_image.shape[1]), dtype=np.uint8)

    for ligne in lignes:
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


def rlsa(image, horizontal, vertical):
    result = np.copy(image)  # Create a copy of the input image

    # Horizontal RLSA
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == 0:
                for k in range(1, horizontal + 1):
                    if j + k < image.shape[1]:
                        result[i, j + k] = 0
                    if j - k >= 0:
                        result[i, j - k] = 0

    # Vertical RLSA
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            if image[j, i] == 0:
                for k in range(1, vertical + 1):
                    if j + k < image.shape[0]:
                        result[j + k, i] = 0
                    if j - k >= 0:
                        result[j - k, i] = 0

    return result
