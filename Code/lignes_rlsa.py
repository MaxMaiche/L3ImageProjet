import numpy as np


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
