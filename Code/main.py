import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy
from PIL import Image

nomImage = "../Ressources/Images/" + input("Nom de l'image : ")
image = mpimg.imread(nomImage).copy()
# print(image)

# formatImage = nomImage.split('.')[-1]

# if formatImage == "png":
if image.max() <= 1:
    image = np.uint8(255 * image)

print(image)

imageGrayscale = np.array([[round(sum(image[i, j]) / 3) for j in range(image.shape[1])] for i in range(image.shape[0])])

sobel = np.array([[-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]])

imageConvolution = scipy.signal.convolve2d(imageGrayscale, sobel, mode='same', boundary='fill', fillvalue=0)
# imageConvolution = np.array([[int(imageConvolution[i, j]) for j in range(imageConvolution.shape[1])] for i in
#                              range(imageConvolution.shape[0])])

print(imageConvolution)

# jpeg : plt.imread(image,format="jpeg")images = []
# for i in range(100):
#     try:
#         image = mpimg.imread('Ressources/' + str(i) + '.jpg').copy()
#         images.append(image)
#         con
#     except:
#         pass

imageConvolution = np.array([[abs(imageConvolution[i, j]) for j in range(imageConvolution.shape[1])] for i in
                             range(imageConvolution.shape[0])])

mi, ma = np.min(imageConvolution), np.max(imageConvolution)

imageConvolution = np.uint8([[int(255 * (imageConvolution[i, j] - mi) / (ma - mi)) for j in
                              range(imageConvolution.shape[1])] for i in range(imageConvolution.shape[0])])

imageConvolution = np.uint8([[255 if imageConvolution[i, j] > 30 else 0 for j in range(imageConvolution.shape[1])] for i in range(imageConvolution.shape[0])])

print(np.max(imageConvolution))

plt.imshow(imageGrayscale, cmap='gray', vmin=0, vmax=255)
plt.show()

plt.imshow(imageConvolution, cmap='gray', vmin=0, vmax=255, )

im = Image.fromarray(imageConvolution)
im.save("resultat_convolution.jpeg")


