import random

import cv2 as cv
import numpy as np

from Code.traitements_basiques import get_edges, get_intersection


def get_board(nom_image):
    base_image = cv.imread("../Ressources/VALIDATIONNEPASTOUCHER/" + nom_image)

    # Resize image
    width = base_image.shape[1]
    height = base_image.shape[0]
    ratio = 1024 / width
    dimensions = (1024, int(height * ratio))

    resized_base_image = cv.resize(base_image, dimensions)

    # Make image grayscale
    base_grayscale = cv.cvtColor(resized_base_image, cv.COLOR_BGR2GRAY)

    # Get the edges
    base_edges = get_edges(resized_base_image)

    # Get the lines
    line_top, line_bottom, line_left, line_right = get_board_lines(resized_base_image, base_edges, min_width=675)

    # Get the corners of the board
    point_top_left = get_intersection(line_top, line_left)
    point_top_right = get_intersection(line_top, line_right)
    point_bottom_left = get_intersection(line_bottom, line_left)
    point_bottom_right = get_intersection(line_bottom, line_right)

    # Scale points back to original size
    point_top_left = (int(point_top_left[0] / ratio), int(point_top_left[1] / ratio))
    point_top_right = (int(point_top_right[0] / ratio), int(point_top_right[1] / ratio))
    point_bottom_left = (int(point_bottom_left[0] / ratio), int(point_bottom_left[1] / ratio))
    point_bottom_right = (int(point_bottom_right[0] / ratio), int(point_bottom_right[1] / ratio))

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


def get_board_lines(img, edges, min_width=650, delta=0.1, tolerance=0.05):
    # Identify lines
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=5)

    line_top = (0, 0, img.shape[1], 0)
    line_bottom = (0, img.shape[0], img.shape[1], img.shape[0])

    if lines is None:
        print(":/")
        exit()

    image_result = img.copy()
    image_lines = img.copy()

    # Identify the board's top and bottom lines
    for line in lines:
        x1, y1, x2, y2 = line[0]  # Coordinates of the line
        if (y1 + y2) / 2 < img.shape[0] / 2:  # If the line is on the top of the image
            if (y1 + y2) / 2 > (line_top[1] + line_top[
                3]) / 2 and abs(x1 - x2) > min_width:  # If the line is lower than the last one and is long enough
                # Print the conditions
                if (y1 + y2) / 2 - (line_top[1] + line_top[3]) / 2 > 50 or line_top == (0, 0, img.shape[1], 0) or abs(
                        line_top[0] - line_top[2]) * (
                        1 - tolerance) < abs(x1 - x2):  # If the line is significantly lower or longer than the last one
                    line_top = (x1, y1, x2, y2)  # Set the line as the top line
        else:  # If the line is on the bottom of the image
            if (y1 + y2) / 2 < (line_bottom[1] + line_bottom[
                3]) / 2 and abs(x1 - x2) > min_width:  # If the line is higher than the last one and is long enough
                if (line_bottom[1] + line_bottom[3]) / 2 - (y1 + y2) / 2 > 50 or line_bottom == (
                        0, img.shape[0], img.shape[1], img.shape[0]) or abs(line_top[0] - line_top[2]) * (
                        1 - tolerance) < abs(x1 - x2):  # If the line is significantly higher or longer than the last one
                    line_bottom = (x1, y1, x2, y2)  # Set the line as the bottom line

        # Draw the line for the lines image
        cv.line(image_lines, (x1, y1), (x2, y2), (
            random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1)

    # Draw the top and bottom lines on the results image
    cv.line(image_result, (line_top[0], line_top[1]), (line_top[2], line_top[3]), (0, 0, 255), 10)
    cv.line(image_result, (line_bottom[0], line_bottom[1]), (line_bottom[2], line_bottom[3]), (0, 0, 255), 10)

    # Identify the left and right lines
    line_left = (img.shape[1] // 2, line_top[1], img.shape[1] // 2, line_bottom[1])
    line_right = (img.shape[1] // 2, line_top[3], img.shape[1] // 2, line_bottom[3])

    d_left = abs(line_top[1] - line_bottom[1])
    d_right = abs(line_top[3] - line_bottom[3])

    for line in lines:
        x1, y1, x2, y2 = line[0]  # Coordinates of the line
        if (x1 + x2) / 2 < img.shape[1] / 2:  # If the line is on the left of the image
            if (x1 + x2) / 2 < (line_left[0] + line_left[
                2]) / 2 and 1 - delta < abs(y1 - y2) / d_left < 1 + delta:  # If the line is on the left of the last one and is around the same height as the board
                if (x1 + x2) / 2 - (line_left[0] + line_left[2]) / 2 < -50 or abs(line_left[1] - line_left[
                    3]) < abs(y1 - y2):  # If the line is significantly on the left or longer than the last one
                    line_left = (x1, y1, x2, y2)  # Set the line as the left line
        else:  # If the line is on the right of the image
            if (x1 + x2) / 2 > (line_right[0] + line_right[
                2]) / 2 and 1 - delta < abs(y1 - y2) / d_right < 1 + delta:  # If the line is on the right of the last one and is around the same height as the board
                if (line_right[0] + line_right[2]) / 2 - (x1 + x2) / 2 < -50 or abs(line_right[1] - line_right[
                    3]) < abs(y1 - y2):  # If the line is significantly on the right or longer than the last one
                    line_right = (x1, y1, x2, y2)  # Set the line as the right line

    # Check if the left and right points of the top and bottom lines are close to the edges of the image
    if line_top[0] < 10 and line_bottom[0] < 10 and (line_left[0] + line_left[2]) / 2 > 100:
        line_left = (0, line_top[1], 0, line_bottom[1])
    if line_top[2] > img.shape[1] - 10 and line_bottom[2] > img.shape[1] - 10 and (line_right[0] + line_right[2]) / 2 < \
            img.shape[1] - 100:
        line_right = (img.shape[1], line_top[3], img.shape[1], line_bottom[3])

    # If the left and right lines are defaut value, set them to the edges of the image
    if line_left == (img.shape[1] // 2, line_top[1], img.shape[1] // 2, line_bottom[1]):
        line_left = (0, line_top[1], 0, line_bottom[1])
    if line_right == (img.shape[1] // 2, line_top[3], img.shape[1] // 2, line_bottom[3]):
        line_right = (img.shape[1], line_top[3], img.shape[1], line_bottom[3])

    # Draw the left and right lines on the results image
    cv.line(image_result, (line_left[0], line_left[1]), (line_left[2], line_left[3]), (0, 0, 255), 10)
    cv.line(image_result, (line_right[0], line_right[1]), (line_right[2], line_right[3]), (0, 0, 255), 10)

    # Get the corners of the board
    point_top_left = get_intersection(line_top, line_left)
    point_top_right = get_intersection(line_top, line_right)
    point_bottom_left = get_intersection(line_bottom, line_left)
    point_bottom_right = get_intersection(line_bottom, line_right)

    # Draw the corners on the results image
    to_int = lambda x: (int(x[0]), int(x[1]))
    cv.circle(image_result, to_int(point_top_left), 10, (0, 255, 0), -1)
    cv.circle(image_result, to_int(point_top_right), 10, (0, 255, 0), -1)
    cv.circle(image_result, to_int(point_bottom_left), 10, (0, 255, 0), -1)
    cv.circle(image_result, to_int(point_bottom_right), 10, (0, 255, 0), -1)

    base_detected_top_bottom_lines = img.copy()
    cv.line(base_detected_top_bottom_lines, (line_top[0], line_top[1]), (line_top[2], line_top[3]), (0, 0, 255), 10)
    cv.line(base_detected_top_bottom_lines, (line_bottom[0], line_bottom[1]), (line_bottom[2], line_bottom[3]), (0, 0, 255), 10)

    base_detected_left_right_lines = base_detected_top_bottom_lines.copy()
    cv.line(base_detected_left_right_lines, (line_left[0], line_left[1]), (line_left[2], line_left[3]), (255, 0, 0), 10)
    cv.line(base_detected_left_right_lines, (line_right[0], line_right[1]), (line_right[2], line_right[3]), (255, 0, 0), 10)

    cv.imwrite('Resultats/base_detected_left_right_lines.jpg', base_detected_left_right_lines)
    cv.imwrite('Resultats/base_detected_top_bottom_lines.jpg', base_detected_top_bottom_lines)
    cv.imwrite('Resultats/base_detected_hough_lines.jpg', image_lines)
    cv.imwrite('Resultats/base_detected_board.jpg', image_result)

    return line_top, line_bottom, line_left, line_right
