import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from canny  import canny
import kuan
from scipy import ndimage   
from scipy.ndimage import distance_transform_edt
from skimage.util import random_noise
from skimage import feature


DEFAULT_ALPHA = 1.0 / 9

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
"""
test_index=1
im_orig = Image.open('imgs/fig_geom_speckled.tif').convert('L')
im_orig_array = np.array(im_orig)

plt.figure(test_index * 10 + 1)
plt.subplot(131)
plt.imshow(im_orig_array, cmap=plt.cm.gray)

plt.axis('off')
plt.title('noisy')
plt.show()
"""
fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
#ax1 = fig.add_subplot(121)  # left side
#ax2 = fig.add_subplot(122) 



ax1=fig.add_subplot(221)   #top left
ax2=fig.add_subplot(222)   #top right
ax3=fig.add_subplot(223)   #bottom left
ax4=fig.add_subplot(224)

im_orig = Image.open('imgs/fig_geom_speckled.tif').convert('L')
im_orig_array = np.array(im_orig)

#cu = kuan.variation(im_orig_array[50:100,300:350])

im_filtered_array = kuan.filter(im_orig_array)

im_filtered_edges = canny(im_filtered_array, 0.1, 20, 50)



im_gold = Image.open('imgs/fig_geom_goldStd.tif').convert('L')
im_gold_array = np.array(im_gold)
im_gold_edges = canny(im_gold_array, 0.1, 20, 50)

im_gold_dist = distance_transform_edt(np.invert(im_gold_edges))
f = fom(im_filtered_array, im_gold_array)




original = Image.open('imgs/fig_geom_speckled.tif').convert('L')




sobel = ndimage.sobel(original)
prewitt = ndimage.prewitt(original)
ax1.imshow(original)
ax2.imshow(sobel)
ax3.imshow(prewitt)




ax4.imshow(im_gold_dist)
plt.show()

"""

cu = kuan.variation(im_orig_array[50:100,300:350])
im_filtered_array = kuan.filter(im_orig_array,2,cu)

plt.figure(test_index * 10 + 1)
plt.subplot(132)
plt.imshow(im_filtered_array, cmap=plt.cm.gray)
plt.axis('off')
plt.title('filtered, Cu=%s' % cu)
plt.show()
"""







"""


im_gold_edges = canny(im_gold_array, 0.1, 20, 50)

plt.figure(test_index * 10 + 2)
plt.subplot(132)
plt.imshow(im_gold_edges, cmap=plt.cm.gray)
plt.axis('off')
plt.title('gold std edges')

im_gold_dist = distance_transform_edt(np.invert(im_gold_edges))
f = fom(im_filtered_array, im_gold_array)
plt.figure(test_index * 10 + 2)
plt.subplot(133)
plt.imshow(im_gold_dist)
plt.axis('off')
plt.title('dist transf gold,fom=%s' % f)

plt.show()

"""