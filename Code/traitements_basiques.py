import cv2 as cv
import numpy as np


def get_intersection(line1, line2):
    """
    Get the intersection of two lines
    :param line1: the first line
    :param line2: the second line
    :return: [x, y] the coordinates of the intersection
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Avoid division by zero
    if x1 == x2:
        x1 += 0.0000001
    if x3 == x4:
        x3 += 0.0000001

    # Compute the two lines' equations

    a1 = (y2 - y1) / (x2 - x1)
    b1 = y1 - a1 * x1

    a2 = (y4 - y3) / (x4 - x3)
    b2 = y3 - a2 * x3

    # Compute the intersection

    x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1

    return [x, y]


def get_edges(gray):
    # Do edge detection
    edges = cv.Canny(gray, 50, 150, apertureSize=3)

    cv.imwrite('./Resultats/base_non_dilated_edges.jpg', edges)

    # Closure
    kernel = np.ones((15, 15), np.uint8)
    edges = cv.dilate(edges, kernel, iterations=1)
    edges = cv.erode(edges, kernel, iterations=1)
    # cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    # Dilate edge detection image so the lines are more readable
    kernel = np.ones((10, 5), np.uint8)
    edges = cv.dilate(edges, kernel, iterations=1)

    return edges


def get_all_lines(image):
    mask = np.ones(image.shape[:2], dtype="uint8") * 255  # create a white image
    lines = lines_extraction(image, 100, 50)  # extract lines -> a changer pour une meilleur ligne IMPORTANT
    try:
        for line in lines:  # write lines to mask
            x1, y1, x2, y2 = line[0]
            cv.line(mask, (x1, y1), (x2, y2), (0, 255, 0), 3)
    except TypeError:
        pass
    (_, contours, _) = cv.findContours(~image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # find contours
    areas = [cv.contourArea(c) for c in contours]  # find area of contour
    avgArea = sum(areas) / len(areas)  # finding average area
    for c in contours:  # average area heuristics
        if cv.contourArea(c) > 20 * avgArea:  # 20% de la longeur = ligne -> à modifier pour meilleur résultat IMPORTANT
            cv.drawContours(mask, [c], -1, 0, -1)
            binary = cv.bitwise_and(binary, binary, mask=mask)  # subtracting the noise
    # cv.imwrite('noise.png', mask)
    # cv.imshow('mask', mask)
    # cv.imshow('binary_noise_removal', ~binary)
    # cv.imwrite('binary_noise_removal.png', ~binary)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


def lines_extraction(gray, minLineLength, maxLineGap):
    edges = cv.Canny(gray, 75, 150)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    return lines
