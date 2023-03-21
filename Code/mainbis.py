import cv2 as cv
import numpy as np

# Import image
ok = False
while not ok:
    try:
        img = cv.imread("../Ressources/Images/" + input("Nom de l'image : "))
        ok = True
    except:
        pass

# Make image grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Do edge detection
edges = cv.Canny(gray, 50, 150, apertureSize=3)

# Dilate edge detection image so the lines are more readable
kernel = np.ones((11, 5), np.uint8)
edges = cv.dilate(edges, kernel, iterations=1)

# Identify lines
lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=400, maxLineGap=30)

maxLine1 = (0, 0, 0, 0)
maxLine2 = (0, 0, 0, 0)

# Display each line on the original image
for line in lines:
    x1, y1, x2, y2 = line[0]
    if abs(x1 - x2) > abs(maxLine1[0] - maxLine1[2]):
        maxLine1 = line[0]
    elif abs(y1 - maxLine1[1]) > 100 and abs(x1 - x2) > abs(maxLine2[0] - maxLine2[2]):
        maxLine2 = line[0]

    cv.line(img, (x1, y1), (x2, y2), (0, 200, 200), 2)

cv.line(img, (maxLine1[0], maxLine1[1]), (maxLine1[2], maxLine1[3]), (0, 0, 255), 10)
cv.line(img, (maxLine2[0], maxLine2[1]), (maxLine2[2], maxLine2[3]), (0, 0, 255), 10)

# Save images
cv.imwrite('gray.jpg', gray)
cv.imwrite('edges.jpg', edges)
cv.imwrite('resultat.jpg', img)
