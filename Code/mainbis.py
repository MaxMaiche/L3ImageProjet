import cv2 as cv
import numpy as np
import validation
import os
def getBoard(nomImage):
    # # Import image
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

    # Do edge detection
    edges = cv.Canny(gray, 50, 150, apertureSize=3)


    # Dilate edge detection image so the lines are more readable
    kernel = np.ones((3, 3), np.uint8)
    edges = cv.dilate(edges, kernel, iterations=1)


    # Closure
    kernel = np.ones((15, 15), np.uint8)
    edges = cv.dilate(edges, kernel, iterations=1)
    edges = cv.erode(edges, kernel, iterations=1)
    #cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    # Identify lines
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=5)

    lineTop = (0, 0, img.shape[1], 0)
    lineBottom = (0, img.shape[0], img.shape[1], img.shape[0])

    if lines is None:
        print(":/")
        exit()


    imgLines = img.copy()

    # Display each line on the original image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if (y1 + y2) / 2 < img.shape[0] / 2:
            if (y1 + y2) / 2 > (lineTop[1] + lineTop[3]) / 2 and abs(x1 - x2) > 700:
                if y1 - lineTop[1] > 50 or abs(lineTop[0] - lineTop[2]) < abs(x1 - x2):
                    lineTop = (x1, y1, x2, y2)
        else:
            if (y1 + y2) / 2 < (lineBottom[1] + lineBottom[3]) / 2 and abs(x1 - x2) > 700:
                if lineBottom[1] - y1 > 50 or abs(lineTop[0] - lineTop[2]) < abs(x1 - x2):
                    lineBottom = (x1, y1, x2, y2)

        cv.line(imgLines, (x1, y1), (x2, y2), (0, 200, 200), 2)



    if lineTop[1] > lineBottom[1]:
        lineTop, lineBottom = lineBottom, lineTop

    cv.line(img, (lineTop[0], lineTop[1]), (lineTop[2], lineTop[3]), (0, 0, 255), 10)
    cv.line(img, (lineBottom[0], lineBottom[1]), (lineBottom[2], lineBottom[3]), (0, 0, 255), 10)

    pts1 = np.float32([[lineTop[0], lineTop[1]], [lineTop[2], lineTop[3]], [lineBottom[2], lineBottom[3]], [lineBottom[0], lineBottom[1]]])
    ratio = ((abs(lineTop[0] - lineTop[2]) + abs(lineBottom[0] - lineBottom[2])) / 2) / ((abs(lineTop[1] - lineBottom[1]) + abs(lineTop[3] - lineBottom[3])) / 2)
    # print("Ligne1 : ", lineTop)
    # print("Ligne2 : ", lineBottom)
    # print("ratio: ", ratio)

    boardW = 1024
    boardH = int(boardW/ratio)

    pts2 = np.float32([[0, 0], [boardW, 0], [boardW, boardH], [0, boardH]])

    M = cv.getPerspectiveTransform(pts1, pts2)

    transformed = np.zeros((int(boardW), int(boardH)), dtype=np.uint8)
    board = cv.warpPerspective(img, M, transformed.shape)


    # Save images
    cv.imwrite('gray.jpg', gray)
    cv.imwrite('edges.jpg', edges)
    cv.imwrite('resultat.jpg', img)
    cv.imwrite('transformed.jpg', board)

    binaryImage = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    points = np.array([[lineTop[0], lineTop[1]],
                       [lineTop[2], lineTop[3]],
                       [lineBottom[2], lineBottom[3]],
                       [lineBottom[0], lineBottom[1]]], np.int32)

    cv.fillPoly(binaryImage, pts=[points], color=(255, 255, 255))
    cv.imwrite('binary.jpg', binaryImage)
    nomImage = nomImage.split(".")[0]
    valid = cv.imread("../Validation/labeling/"+nomImage+"/board.png", cv.IMREAD_GRAYSCALE)

    score = validation.compare(valid, binaryImage)
    print("Image : ", nomImage)
    print("Score : {:.2f}%".format(score*100))
    print(f"Validation {'réussie' if score > 0.9 else 'échouée'}")
    if score > 0.9:
        return 1
    return 0
def doALL():
    folder_path = '../Ressources/Images'
    allowed_extensions = '.jpg', '.png', '.jpeg'
    cpt = 0
    cptTotal = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(allowed_extensions):
            if os.path.exists("../Validation/labeling/"+filename.split(".")[0]):
                cpt += getBoard(filename)
                cptTotal += 1
                print()

    print("Score total : ", cpt, "/", cptTotal)

def doOne():
    getBoard("12.jpg")

if __name__ == "__main__":
    #doALL()
    doOne()

