import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage import io, filters, feature, morphology, color
from skimage.measure import label

# Define the Canny-Deriche smoothing filter
def canny_deriche_filter(s, alpha):
    n = 1
    si = np.arange(-s, s+1)
    si = si.astype(np.float64)
    h = np.zeros(si.shape)

    eps = 1e-10  # small positive number to avoid division by zero
    for i in range(len(si)):
        h[i] = np.exp(-alpha*si[i]) * ((alpha*si[i])**n) * ((1-np.exp(-alpha*si[i])) / ((si[i]**2) + eps))

    h = h / np.sum(h)
    return h,si

# Load the image
img = io.imread('C:/Users/ksevi/OneDrive/Desktop/MASTER/SCENE_LALIGANT/fourth report/fruits.png')
gray_img = color.rgb2gray(img)

# Define the filter parameters
alpha = 2
s = 10
h, si = canny_deriche_filter(s, alpha)

# Normalize the filter in the discrete domain
h /= np.sum(h)

# Plot the filter in the discrete domain
plt.plot(si, h)
plt.xlabel('s')
plt.ylabel('h(s)')
plt.title('Canny-Deriche filter')
plt.show()

# Truncate the support of the filter according to s
support = np.where(h > h[0]/100)[0]
h = h[support]
s = si[support]

# Apply the filter to the image
smoothed_img = signal.convolve2d(gray_img, np.transpose([h]), mode='same')
smoothed_img = signal.convolve2d(smoothed_img, [h], mode='same')

# Compute the gradient components
Gx = signal.convolve2d(smoothed_img, [[-1, 1]], mode='same')
Gy = signal.convolve2d(smoothed_img, [[-1], [1]], mode='same')

# Compute the magnitude and direction of the gradient
G = np.sqrt(Gx**2 + Gy**2)
theta = np.arctan2(Gy, Gx)

# Compute the non-maximum suppressed gradient magnitude
nms = filters.unsharp_mask(G)

# Threshold the gradient magnitude
sh = 0.2
sb = 0.05
edges = (nms > sh*np.max(nms))

# Refine the segmentation with hysteresis thresholding
edges = morphology.dilation(edges)
edges = feature.canny(nms, sb*np.max(nms), sh*np.max(nms))

# Remove small objects
edges = morphology.remove_small_objects(edges, 50)

# Label connected components
labels = label(edges)

# Plot the results
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
ax = axes.ravel()
ax[0].imshow(img)
ax[0].set_title('Original image')
ax[1].imshow(smoothed_img, cmap=plt.cm.gray)
ax[1].set_title('Smoothed image')
ax[2].imshow(G, cmap=plt.cm.gray)
ax[2].set_title('Gradient magnitude')
ax[3].imshow(edges, cmap=plt.cm.gray)
ax[3].set_title('Edges')
plt.show()
