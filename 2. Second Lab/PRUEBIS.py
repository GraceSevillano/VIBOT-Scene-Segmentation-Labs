import cv2
import numpy as np

def connected_component_labeling(img):
    # Finding the height and width of the image
    height, width = img.shape
    # Initializing the label image with zeros
    label_image = np.zeros((height, width), dtype=np.uint32)
    # Initializing the label with 1
    label = 1

    for row in range(height):
        for col in range(width):
            if img[row][col] == 255 and label_image[row][col] == 0:
                # New component found, perform BFS to label all its pixels
                q = [(row, col)]
                while len(q) > 0:
                    r, c = q.pop(0)
                    # Check if the current pixel is not already labeled
                    if label_image[r][c] == 0:
                        label_image[r][c] = label
                        # Adding its neighbors to the queue
                        if r > 0:
                            q.append((r - 1, c))
                        if c > 0:
                            q.append((r, c - 1))
                        if r < height - 1:
                            q.append((r + 1, c))
                        if c < width - 1:
                            q.append((r, c + 1))
                label += 1
    return label_image

# Reading an image
img = cv2.imread('C:/Users/ksevi/OneDrive/Desktop/MASTER/SCENE_LALIGANT/second REPORT/vibot.png', 0)

# Thresholding the image to binary
_, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

# Calling the connected component labeling function
labeled_image = connected_component_labeling(img)

# Displaying the labeled image
cv2.imshow('Labeled Image', labeled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()