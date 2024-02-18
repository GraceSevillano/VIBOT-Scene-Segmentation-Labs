import numpy as np
import matplotlib.pyplot as plt

def connected_components(image):
    m, n = image.shape
    #I locate the coordinates that only have data (1) in the image
    coor_y, coor_x = np.where(image != 0)
    listita = []
    image_copy = np.zeros((m, n))
    label = 0
    for elem in range(len(coor_x)):
        i = coor_y[elem]
        j = coor_x[elem]
        if image_copy[i, j] == 0:
            image_copy[i, j] = label+1
            listita.append((i, j))
            while len(listita) != 0:
                current = listita.pop(0)              
                i, j = current
                neighbors = [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j-1), (i+1, j-1), (i+1, j)]              
                for neighbor in neighbors:
                    x, y = neighbor
                    if x >= 0 and x < m and y >= 0 and y < n and image[x, y] != 0 and image_copy[x, y] == 0:
                        image_copy[x, y] = label +1
                        listita.append((x, y))                 
            label += 1
    return image_copy, label

def ploting():
    image_matriz = np.array([
        [0,0,0,0,0,0,0,0],
        [0,1,1,0,0,1,1,1],
        [0,1,0,0,0,1,1,0],
        [0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,0,0],
        [0,1,1,1,1,1,0,0],
        [0,0,0,0,0,0,0,0]
    ])
    print("Matrix original")
    print(image_matriz)
    print(" ")
    
    matriz_label, count_label = connected_components(image_matriz)
    print("Matrix with labels")                                               
    print(matriz_label)
    print(" ")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax[0].imshow(image_matriz)
    ax[0].set_title('Original image ', fontsize=10)

    ax[1].imshow(matriz_label)
    ax[1].set_title('Image with labels', fontsize=10)
    plt.show()

ploting()

