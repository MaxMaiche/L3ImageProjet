import cv2 as cv
import numpy as np

from Code.traitements_basiques import getEdges, getBoardLines, getIntersection


def get_board(nom_image):
    base_image = cv.imread("../Ressources/Images/" + nom_image)

    # Resize image
    width = base_image.shape[1]
    height = base_image.shape[0]
    ratio = 1024 / width
    dimensions = (1024, int(height * ratio))

    base_image = cv.resize(base_image, dimensions)

    # Make image grayscale
    base_grayscale = cv.cvtColor(base_image, cv.COLOR_BGR2GRAY)

    # Get the edges
    base_edges = getEdges(base_image)

    # Get the lines
    line_top, line_bottom, line_left, line_right = getBoardLines(base_image, base_edges, minWidth=675)

    # Get the corners of the board
    point_top_left = getIntersection(line_top, line_left)
    point_top_right = getIntersection(line_top, line_right)
    point_bottom_left = getIntersection(line_bottom, line_left)
    point_bottom_right = getIntersection(line_bottom, line_right)

    # Apply perspective transform
    points_depart = np.float32([point_top_left, point_top_right, point_bottom_right, point_bottom_left])
    ratio = (((point_top_right[0] - point_top_left[0]) + (point_bottom_right[0] - point_bottom_left[0])) / 2) / (
            ((point_bottom_left[1] - point_top_left[1]) + (point_bottom_right[1] - point_top_right[1])) / 2)

    board_width = 1024
    board_height = int(board_width / ratio)

    points_arrivee = np.float32([[0, 0], [board_width, 0], [board_width, board_height], [0, board_height]])

    M = cv.getPerspectiveTransform(points_depart, points_arrivee)

    board = cv.warpPerspective(base_image, M, (board_width, board_height))

    # Save images
    cv.imwrite('Resultats/base_gray.jpg', base_grayscale)
    cv.imwrite('Resultats/base_edges.jpg', base_edges)
    cv.imwrite('Resultats/board.jpg', board)

    # Compare to validation
    binary_board = np.zeros((base_image.shape[0], base_image.shape[1]), dtype=np.uint8)
    board_points = np.array([point_top_left, point_top_right, point_bottom_right, point_bottom_left], np.int32)

    cv.fillPoly(binary_board, pts=[board_points], color=(255, 255, 255))
    cv.imwrite('Resultats/board_binary.jpg', binary_board)

    return board, binary_board, M
