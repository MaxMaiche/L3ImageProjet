import random

import cv2 as cv
import numpy as np


def getIntersection(line1, line2):
    """
    Get the intersection of two lines
    :param line1: the first line
    :param line2: the second line
    :return: [x, y] the coordinates of the intersection
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    if x1 == x2:
        x1 += 0.0000001
    if x3 == x4:
        x3 += 0.0000001

    a1 = (y2 - y1) / (x2 - x1)
    b1 = y1 - a1 * x1

    a2 = (y4 - y3) / (x4 - x3)
    b2 = y3 - a2 * x3

    x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1

    return [x, y]


def getEdges(gray):
    # Do edge detection
    edges = cv.Canny(gray, 50, 150, apertureSize=3)

    # Closure
    kernel = np.ones((15, 15), np.uint8)
    edges = cv.dilate(edges, kernel, iterations=1)
    edges = cv.erode(edges, kernel, iterations=1)
    # cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    # Dilate edge detection image so the lines are more readable
    kernel = np.ones((10, 5), np.uint8)
    edges = cv.dilate(edges, kernel, iterations=1)

    return edges


def getBoardLines(img, edges, minWidth=650, delta=0.1, tolerance=0.05):
    # # Add image borders to the edges image
    # cv.line(edges, (0, 0), (0, img.shape[0]), (255, 255, 255), 10)
    # cv.line(edges, (0, 0), (img.shape[1], 0), (255, 255, 255), 10)
    # cv.line(edges, (img.shape[1], 0), (img.shape[1], img.shape[0]), (255, 255, 255), 10)
    # cv.line(edges, (0, img.shape[0]), (img.shape[1], img.shape[0]), (255, 255, 255), 10)

    # Identify lines
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=5)

    lineTop = (0, 0, img.shape[1], 0)
    lineBottom = (0, img.shape[0], img.shape[1], img.shape[0])

    if lines is None:
        print(":/")
        exit()

    imgResult = img.copy()
    imgLines = img.copy()

    # Identify the board's top and bottom lines
    for line in lines:
        x1, y1, x2, y2 = line[0]  # Coordinates of the line
        if (y1 + y2) / 2 < img.shape[0] / 2:  # If the line is on the top of the image
            if (y1 + y2) / 2 > (lineTop[1] + lineTop[
                3]) / 2 and abs(x1 - x2) > minWidth:  # If the line is lower than the last one and is long enough
                # Print the conditions
                if (y1 + y2) / 2 - (lineTop[1] + lineTop[3]) / 2 > 50 or lineTop == (0, 0, img.shape[1], 0) or abs(
                        lineTop[0] - lineTop[2]) * (
                        1 - tolerance) < abs(x1 - x2):  # If the line is significantly lower or longer than the last one
                    lineTop = (x1, y1, x2, y2)  # Set the line as the top line
        else:  # If the line is on the bottom of the image
            if (y1 + y2) / 2 < (lineBottom[1] + lineBottom[
                3]) / 2 and abs(x1 - x2) > minWidth:  # If the line is higher than the last one and is long enough
                if (lineBottom[1] + lineBottom[3]) / 2 - (y1 + y2) / 2 > 50 or lineBottom == (
                        0, img.shape[0], img.shape[1], img.shape[0]) or abs(lineTop[0] - lineTop[2]) * (
                        1 - tolerance) < abs(x1 - x2):  # If the line is significantly higher or longer than the last one
                    lineBottom = (x1, y1, x2, y2)  # Set the line as the bottom line

        # Draw the line for the lines image
        cv.line(imgLines, (x1, y1), (x2, y2), (
            random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1)

    # Draw the top and bottom lines on the results image
    cv.line(imgResult, (lineTop[0], lineTop[1]), (lineTop[2], lineTop[3]), (0, 0, 255), 10)
    cv.line(imgResult, (lineBottom[0], lineBottom[1]), (lineBottom[2], lineBottom[3]), (0, 0, 255), 10)

    # Identify the left and right lines
    lineLeft = (img.shape[1] // 2, lineTop[1], img.shape[1] // 2, lineBottom[1])
    lineRight = (img.shape[1] // 2, lineTop[3], img.shape[1] // 2, lineBottom[3])

    dLeft = abs(lineTop[1] - lineBottom[1])
    dRight = abs(lineTop[3] - lineBottom[3])

    for line in lines:
        x1, y1, x2, y2 = line[0]  # Coordinates of the line
        if (x1 + x2) / 2 < img.shape[1] / 2:  # If the line is on the left of the image
            if (x1 + x2) / 2 < (lineLeft[0] + lineLeft[
                2]) / 2 and 1 - delta < abs(y1 - y2) / dLeft < 1 + delta:  # If the line is on the left of the last one and is around the same height as the board
                if (x1 + x2) / 2 - (lineLeft[0] + lineLeft[2]) / 2 < -50 or abs(lineLeft[1] - lineLeft[
                    3]) < abs(y1 - y2):  # If the line is significantly on the left or longer than the last one
                    lineLeft = (x1, y1, x2, y2)  # Set the line as the left line
        else:  # If the line is on the right of the image
            if (x1 + x2) / 2 > (lineRight[0] + lineRight[
                2]) / 2 and 1 - delta < abs(y1 - y2) / dRight < 1 + delta:  # If the line is on the right of the last one and is around the same height as the board
                if (lineRight[0] + lineRight[2]) / 2 - (x1 + x2) / 2 < -50 or abs(lineRight[1] - lineRight[
                    3]) < abs(y1 - y2):  # If the line is significantly on the right or longer than the last one
                    lineRight = (x1, y1, x2, y2)  # Set the line as the right line

    # Check if the left and right points of the top and bottom lines are close to the edges of the image
    if lineTop[0] < 10 and lineBottom[0] < 10 and (lineLeft[0] + lineLeft[2]) / 2 > 100:
        lineLeft = (0, lineTop[1], 0, lineBottom[1])
    if lineTop[2] > img.shape[1] - 10 and lineBottom[2] > img.shape[1] - 10 and (lineRight[0] + lineRight[2]) / 2 < \
            img.shape[1] - 100:
        lineRight = (img.shape[1], lineTop[3], img.shape[1], lineBottom[3])

    # If the left and right lines are defaut value, set them to the edges of the image
    if lineLeft == (img.shape[1] // 2, lineTop[1], img.shape[1] // 2, lineBottom[1]):
        lineLeft = (0, lineTop[1], 0, lineBottom[1])
    if lineRight == (img.shape[1] // 2, lineTop[3], img.shape[1] // 2, lineBottom[3]):
        lineRight = (img.shape[1], lineTop[3], img.shape[1], lineBottom[3])

    # Draw the left and right lines on the results image
    cv.line(imgResult, (lineLeft[0], lineLeft[1]), (lineLeft[2], lineLeft[3]), (0, 0, 255), 10)
    cv.line(imgResult, (lineRight[0], lineRight[1]), (lineRight[2], lineRight[3]), (0, 0, 255), 10)

    cv.imwrite('Resultats/lines.jpg', imgLines)
    cv.imwrite('Resultats/result.jpg', imgResult)

    return lineTop, lineBottom, lineLeft, lineRight


def getAllLines(image):
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