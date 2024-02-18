import matplotlib.pyplot as plt
from skimage import io, color
# Load the image
img = io.imread('C:/Users/ksevi/OneDrive/Desktop/MASTER/SCENE_LALIGANT/fourth report/fruits.png')
gray_img = color.rgb2gray(img)

# Plot the first row of the image as a signal
plt.plot(gray_img[0,:])
plt.show()