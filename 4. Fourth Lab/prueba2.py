import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage import io, filters, feature, morphology, color
from skimage.measure import label

# Define the Canny-Deriche smoothing filter
def canny_deriche_filter(s, alpha):
    n = 1
    si = np.linspace(-s, s, 2*s+1)
    h = np.exp(-alpha*np.abs(si)) * ((alpha*np.abs(si))**n) * ((1-np.exp(-alpha*np.abs(si))) / ((si**2) + 1e-10))
    h /= np.sum(h)
    return h, si

# Load the image
img = io.imread('C:/Users/ksevi/OneDrive/Desktop/MASTER/SCENE_LALIGANT/fourth report/fruits.png')
gray_img = color.rgb2gray(img)

# Define the filter parameters
alpha = 2
s = 10
h, si = canny_deriche_filter(s, alpha)

# Plot the filter in the discrete domain
plt.plot(si, h)
plt.xlabel('s')
plt.ylabel('h(s)')
plt.title('Canny-Deriche filter')
plt.show()

# Truncate the support of the filter according to s
support = np.abs(si) <= s
h = h[support]
s = si[support]

# Apply the filter to the image
smoothed_img = signal.convolve2d(gray_img, np.transpose([h]), mode='same')
smoothed_img = signal.convolve2d(smoothed_img, [h], mode='same')

# Compute the gradient components
kernel = np.array([[-1, 1], [-1, 1]])
G = signal.convolve2d(smoothed_img, kernel, mode='same')
theta = np.arctan2(G[:, 1:], G[:, :-1])

# Compute the non-maximum suppressed gradient magnitude
nms = filters.unsharp_mask(G)

# Threshold the gradient magnitude
sh = 0.2
edges = (nms > sh*np.max(nms))

# Refine the segmentation with hysteresis thresholding
edges = feature.canny(nms, 0, sh*np.max(nms))

# Remove small objects
edges = morphology.remove_small_objects(edges, 50)

# Label connected components
labels = label(edges)

# Keep only the connected components which contain a point above sh
sh_labels = np.unique(labels[edges])
sh_labels = sh_labels[1:] # exclude label 0 (background)
for l in sh_labels:
    l_mask = (labels == l)
    if np.max(nms[l_mask]) <= sh*np.max(nms):
        edges[l_mask] = False

# Plot the results
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
ax = axes.ravel()
ax[0].imshow(img)
ax[0].set_title('Original image')
ax[1].imshow(smoothed_img, cmap=plt.cm.gray)
ax[1].set_title('Smoothed image')
ax[2].imshow(nms, cmap=plt.cm.gray)
ax[2].set_title('Non-maximum suppressed gradient magnitude')
ax[3].imshow(edges, cmap=plt.cm.gray)
ax[3].set_title('Edges')
plt.show()
