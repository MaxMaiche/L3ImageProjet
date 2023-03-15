import scipy
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

nomImage = "../Ressources/Images/" + input("Nom de l'image : ")
image = mpimg.imread(nomImage).copy()
print(image)

formatImage = nomImage.split('.')[-1]

if formatImage == "png":
    image = np.uint8(255 * image)

imageGrayscale = [[round(sum(image[i, j]) / 3) for j in range(image.shape[1])] for i in range(image.shape[0])]

sobel = [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]]

imageConvolution = scipy.signal.fftconvolve(imageGrayscale, sobel)

#jpeg : plt.imread(image,format="jpeg")images = []
# for i in range(100):
#     try:
#         image = mpimg.imread('Ressources/' + str(i) + '.jpg').copy()
#         images.append(image)
#         con
#     except:
#         pass

plt.imshow(imageGrayscale, cmap='gray', vmin=0, vmax=255)
plt.show()

plt.imshow(imageConvolution, cmap='gray')
plt.show()
