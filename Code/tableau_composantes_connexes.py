import os

import cv2 as cv
import numpy as np

from Code import validation_comparaison
from Code.traitements_basiques import get_intersection

global totalpourcent
totalpourcent = []


def calculEuclidienne(point1, point2):
    point1 = list(point1[0])
    point2 = list(point2[0])
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def getBoardCC(nomImage):
    image = cv.imread("../Ressources/VALIDATIONNEPASTOUCHER/" + nomImage)

    # Resize image
    width = image.shape[1]
    height = image.shape[0]
    ratio = 1024 / width
    dim = (1024, int(height * ratio))

    img = cv.resize(image, dsize=dim)

    # Make image grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Get the edges
    edges = edges = cv.Canny(gray, 50, 150, apertureSize=3)

    # Dilate the image
    kernel = np.ones((5, 5), np.uint8)
    edges = cv.dilate(edges, kernel, iterations=1)

    # Get complementary image
    edges = cv.bitwise_not(edges)

    # Get the connected components
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(edges, connectivity=8)

    # Find the biggest component in the center of the image
    max_area = 0
    max_label = 0

    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2

    delta = 50

    for i in range(1, nb_components):
        try:
            inCenter = any(
                [output[c_x, c_y] == i
                 for c_x in range(center_x - delta, center_x + delta)
                 for c_y in range(center_y - delta, center_y + delta)
                 ]
            )
        except IndexError:
            inCenter = False

        if inCenter and max_area < stats[i, cv.CC_STAT_AREA]:
            max_label = i
            max_area = stats[i, cv.CC_STAT_AREA]

    max_x = stats[max_label, cv.CC_STAT_LEFT]
    max_y = stats[max_label, cv.CC_STAT_TOP]
    max_width = stats[max_label, cv.CC_STAT_WIDTH]
    max_height = stats[max_label, cv.CC_STAT_HEIGHT]

    # find 4 corners of the biggest components
    biggestcc = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    biggestcc[output == max_label] = [255, 255, 255]

    # Close the edges
    kernel = np.ones((30, 30), np.uint8)
    biggestcc = cv.morphologyEx(biggestcc, cv.MORPH_CLOSE, kernel)

    # Find contours of the current component
    contours, _ = cv.findContours(biggestcc[:, :, 0].astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = contours[0]  # Assuming there's only one contour
    # Approximate the contour to a polygon
    epsilon = 0.01 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)

    # Draw the corners of the trapeze
    for point in approx:
        cv.circle(biggestcc, tuple(point[0]), 5, (0, 0, 255), -1)

    # keep the 4 edges with the biggest distance between consécutive points
    approx_edges = [(approx[i],
                     approx[(i + 1) % len(approx)],
                     calculEuclidienne(approx[i], approx[(i + 1) % len(approx)]))
                    for i in range(len(approx))]

    # # separate the edges in vertical and horizontal
    # vert_edges = [edge for edge in approx_edges if abs(edge[0][0][0] - edge[1][0][0]) > abs(edge[0][0][1] - edge[1][0][1])]
    # horiz_edges = [edge for edge in approx_edges if abs(edge[0][0][1] - edge[1][0][1]) > abs(edge[0][0][0] - edge[1][0][0])]
    #
    # # draw vertical edges
    # for edge in vert_edges:
    #     pt1 = tuple(edge[0][0])
    #     pt2 = tuple(edge[1][0])
    #     cv.line(biggestcc, pt1, pt2, (0, 255, 0), 5)
    #
    # # draw horizontal edges
    # for edge in horiz_edges:
    #     pt1 = tuple(edge[0][0])
    #     pt2 = tuple(edge[1][0])
    #     cv.line(biggestcc, pt1, pt2, (0, 0, 255), 5)

    kept_approx_edges = sorted(approx_edges, key=lambda x: x[2], reverse=True)[:4]

    # delete the distance
    approx_edges = [(edge[0], edge[1]) for edge in kept_approx_edges]
    # Get back at the old order
    approx_edges = sorted(approx_edges, key=lambda x: approx.tolist().index(x[0].tolist()))

    # draw the edges
    for edge in approx_edges:
        pt1 = tuple(edge[0][0])
        pt2 = tuple(edge[1][0])
        cv.line(biggestcc, pt1, pt2, (0, 255, 0), 5)

    # get the 4 corners
    corners = []
    # get the intersection of the edges
    for i in range(len(approx_edges)):
        line1 = approx_edges[i]
        line2 = approx_edges[(i + 1) % len(approx_edges)]

        line1 = list(line1[0][0]) + list(line1[1][0])
        line2 = list(line2[0][0]) + list(line2[1][0])

        intersection = get_intersection(line1, line2)
        intersection = tuple([int(intersection[0]), int(intersection[1])])
        corners.append(intersection)
        cv.circle(biggestcc, intersection, 5, (255, 0, 0), -1)

    # Find the corners of the board
    xsorted = sorted(corners, key=lambda x: x[0])
    left = xsorted[:2]
    right = xsorted[2:]
    ysorted = sorted(left, key=lambda x: x[1])
    ptTopLeft = ysorted[0]
    ptBottomLeft = ysorted[1]
    ysorted = sorted(right, key=lambda x: x[1])
    ptTopRight = ysorted[0]
    ptBottomRight = ysorted[1]

    # Apply perspective transform
    pts1 = np.float32([ptTopLeft, ptTopRight, ptBottomRight, ptBottomLeft])

    # transform pts1 into point before resize
    pts1 = np.float32([[int(pt[0] / ratio), int(pt[1] / ratio)] for pt in pts1])

    ratio2 = (((ptTopRight[0] - ptTopLeft[0]) + (ptBottomRight[0] - ptBottomLeft[0])) / 2) / (
            ((ptBottomLeft[1] - ptTopLeft[1]) + (ptBottomRight[1] - ptTopRight[1])) / 2)

    boardW = 1024
    boardH = int(boardW / ratio2)

    pts2 = np.float32([[0, 0], [boardW, 0], [boardW, boardH], [0, boardH]])

    M = cv.getPerspectiveTransform(pts1, pts2)

    transformed = np.zeros((int(boardW), int(boardH)), dtype=np.uint8)
    board = cv.warpPerspective(image, M, transformed.shape)

    # Show the image
    # cv.imshow("Board", board)
    # cv.imshow("Connected Components", biggestcc)
    # cv.imshow("Image", img)
    # cv.imshow("Edges", edges)

    # Compare to validation
    binaryImage = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    points = np.array([ptTopLeft, ptTopRight, ptBottomRight, ptBottomLeft], np.int32)

    cv.fillPoly(binaryImage, pts=[points], color=(255, 255, 255))
    cv.imwrite('Resultats/binary.jpg', binaryImage)
    # nomImage = nomImage.split(".")[0]
    # valid = cv.imread("../Validation/labeling/" + nomImage + "/board.png", cv.IMREAD_GRAYSCALE)
    #
    # score = validation_comparaison.compare(valid, binaryImage)
    # global totalpourcent
    # totalpourcent.append(score * 100)
    # print(f"Image : {nomImage}")
    # print("Score : {:.2f}%".format(score * 100))
    # print(f"Validation {'réussie' if score > 0.9 else 'échouée'}")
    # if score > 0.9:
    #     return 1
    # return 0

    return board, binaryImage


def doALL():
    folder_path = '../Ressources/Images'
    allowed_extensions = '.jpg', '.png', '.jpeg'
    cpt = 0
    cptTotal = 0
    total = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(allowed_extensions):
            if os.path.exists("../Validation/labeling/" + filename.split(".")[0]):
                cpt += getBoardCC(filename)
                cptTotal += 1
                print()

    print("Score total : ", cpt, "/", cptTotal)
    global totalpourcent
    print(totalpourcent)


if __name__ == "__main__":
    doALL()

    # getBoardCC("79.jpg")
    # cv.waitKey(0)
    # cv.destroyAllWindows()
