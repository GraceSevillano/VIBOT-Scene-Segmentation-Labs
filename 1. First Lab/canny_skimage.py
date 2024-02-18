import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
from skimage import filters


# Generate noisy image of a square
image = np.zeros((128, 128), dtype=float)
image[:, :-64] = 1

#image1 = ndi.rotate(image, 15, mode='constant')
#image_noise = ndi.gaussian_filter(image, 4)

lvar = filters.gaussian(image, sigma=4) + 1e-10
image_random = random_noise(np.zeros((128, 128), dtype=float), mode='localvar', local_vars=lvar*0.5)
image_random1 = random_noise(image, mode='localvar', local_vars=lvar*0.5)
image_canny= image_random + image

image_canny= image_random + image
#image_random = random_noise(image, mode='localvar', mean=0.1)

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(image_canny)
edges2 = feature.canny(image, sigma=3)

def fom(img, img_gold_std, alpha = DEFAULT_ALPHA):
   

    # To avoid oversmoothing, we apply canny edge detection with very low
    # standard deviation of the Gaussian kernel (sigma = 0.1).
    edges_img = canny(img, 0.1, 20, 50)
    edges_gold = canny(img_gold_std, 0.1, 20, 50)
    
    # Compute the distance transform for the gold standard image.
    dist = distance_transform_edt(np.invert(edges_gold))

    fom = 1.0 / np.maximum(
        np.count_nonzero(edges_img),
        np.count_nonzero(edges_gold))

    N, M = img.shape

    for i in range(0, N):
        for j in range(0, M):
            if edges_img[i, j]:
                fom += 1.0 / ( 1.0 + dist[i, j] * dist[i, j] * alpha)

    fom /= np.maximum(
        np.count_nonzero(edges_img),
        np.count_nonzero(edges_gold))    

    return fom








# display results
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(8, 3))

ax[0].imshow(image, cmap='gray')
ax[0].set_title('image', fontsize=20)

ax[1].imshow(image_random, cmap='gray')
ax[1].set_title('ruido', fontsize=20)

ax[1].imshow(image_canny, cmap='gray')
ax[1].set_title('suma de ruido', fontsize=20)

ax[2].imshow(edges1, cmap='gray')
ax[2].set_title('canny', fontsize=20)

#ax[3].imshow(edges1, cmap='gray')
#ax[3].set_title('Canny filter', fontsize=20)

#ax[4].imshow(image_, cmap='gray')
#ax[4].set_title('noise', fontsize=20)


for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()