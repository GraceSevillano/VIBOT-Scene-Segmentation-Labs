import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import feature,filters
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

imag = np.zeros((128, 128), dtype=float)
imag[:, :-64] = 1
noise =  np.random.normal(loc=0, scale=1, size=imag.shape)
noise2 =noise*20

# noise overlaid over image
noisy = np.clip((imag + noise),0,1)
noisy2 = np.clip((imag + noise2),0,1)

Canny_1 = feature.canny(noisy,sigma=2)
Canny_img = feature.canny(imag,sigma=2) # CANNY REAL IMAGE
Canny_2 = feature.canny(noisy2,sigma=2)

Sobel_1 = filters.sobel(noisy)
Sobel_2 = filters.sobel(imag)
Sobel_3 = filters.sobel(noisy2)

Roberts_1 = filters.roberts(noisy)
Roberts_2 = filters.roberts(imag)
Roberts_3 = filters.roberts(noisy2)
#--------- propuestos------------
Prewitt_1 = filters.prewitt(noisy)
Prewitt_2 = filters.prewitt(imag)
Prewitt_3 = filters.prewitt(noisy2)

Laplace_1 = ndimage.laplace(noisy)
Laplace_2 = ndimage.laplace(imag)
Laplace_3 = ndimage.laplace(noisy2)

DEFAULT_ALPHA = 1.0 / 9

def fom(img, img2, alpha = DEFAULT_ALPHA):
    dist = distance_transform_edt(img)
    fom = 1.0 / np.maximum(
        np.count_nonzero(img),
        np.count_nonzero(img2))
    N, M = img.shape
    for i in range(0, N):
        for j in range(0, M):
            if img[i, j]:
                fom += 1.0 / ( 1.0 + dist[i, j] * dist[i, j] * alpha)
    fom /= np.maximum(
        np.count_nonzero(img),
        np.count_nonzero(img2))    
    return fom

f1 = fom(Canny_img,Canny_1)

f9 = fom(Prewitt_1,Prewitt_3)
f10 = fom(Sobel_1, Sobel_3)
f11 = fom(Roberts_1,Roberts_3)
f12 = fom(Laplace_1, Laplace_3)
f2 = fom(Canny_1, Canny_2)

fig4, ax4 = plt.subplots( figsize=(10, 7))
values=[f9,f10,f11,f12,f2 ]
index=['Prewitt', 'Sobel', 'Roberts', 'Laplace','Canny']
#total = data.sum(axis=1)
ax4.bar(index, values)

plt.show()

print("Comparison Prewitt tiny vs Prewitt-big noise: ",f9)
print("Comparison Sobel tiny vs Sobel-big noise: ",f10)
print("Comparison Roberts tiny vs Roberts-big noise: ",f11)
print("Comparison Laplace tiny vs Laplace-big noise: ",f12)
print("Comparison Canny tiny vs Canny-big noise: ",f2)
print(" ")
fig1, ax1 = plt.subplots(nrows=2, ncols=5, figsize=(10, 5))
####----CANNY------------------
ax1[0][0].imshow(imag,cmap='gray')
ax1[0][0].set_title('image', fontsize=7)

ax1[0][1].imshow(noise,cmap='gray')
ax1[0][1].set_title('Noise', fontsize=7 )

ax1[0][2].imshow(noisy,cmap='gray')
ax1[0][2].set_title('Sum with noise', fontsize=7 )

ax1[0][3].imshow(Canny_1,cmap='gray')
ax1[0][3].set_title('Canny', fontsize=7 )
ax1[0][3].set_xlabel(f'Comparison Canny vs Canny-tiny noise using PFOM: {f1}',loc='right')

ax1[0][4].imshow(Canny_img,cmap='gray')
ax1[0][4].set_title('Canny image', fontsize=7 )

#--------CANNY with big noise--------------
ax1[1][0].imshow(imag, cmap='gray')
ax1[1][0].set_title('image', fontsize=7 )

ax1[1][1].imshow(noise2, cmap='gray')
ax1[1][1].set_title('Noise2', fontsize=7 )

ax1[1][2].imshow(noisy2, cmap='gray')
ax1[1][2].set_title('Sum with noise2', fontsize=7 )

ax1[1][3].imshow(Canny_2, cmap='gray')
ax1[1][3].set_title('Canny_noise2', fontsize=7 )
ax1[1][3].set_xlabel(f'Comparison Canny vs Canny-big noise using PFOM: {f2}',loc='right')

ax1[1][4].imshow(Canny_img, cmap='gray')
ax1[1][4].set_title('Canny image', fontsize=7 )

f3 = fom(Canny_1,Prewitt_1)
f4 = fom(Sobel_1, Prewitt_1)
f5 = fom(Roberts_1,Prewitt_1)
f6 = fom(Canny_1,Laplace_1)
f7 = fom(Sobel_1,Laplace_1)
f8 = fom(Roberts_1,Laplace_1)

##COMPARACIONES
fig2, ax2 = plt.subplots(nrows=3, ncols=2, figsize=(10, 5))
#-----------------canny vs prewitt
ax2[0][0].imshow(Canny_2, cmap='gray')
ax2[0][0].set_title('Canny', fontsize=7 )

ax2[0][1].imshow(Prewitt_3, cmap='gray')
ax2[0][1].set_title('Prewitt', fontsize=7 )
ax2[0][1].set_xlabel(f'Comparison Canny vs Prewitt using PFOM: {f3}',loc='right')
#-------sobel vs prewitt
ax2[1][0].imshow(Sobel_3, cmap='gray')
ax2[1][0].set_title('Sobel', fontsize=7 )

ax2[1][1].imshow(Prewitt_3, cmap='gray')
ax2[1][1].set_title('Prewitt', fontsize=7 )
ax2[1][1].set_xlabel(f'Comparison Sobel vs Prewitt using PFOM: {f4}',loc='right')
#---------------Robert vs prewitt
ax2[2][0].imshow(Roberts_3, cmap='gray')
ax2[2][0].set_title('Roberts', fontsize=7 )

ax2[2][1].imshow(Prewitt_3, cmap='gray')
ax2[2][1].set_title('Prewitt', fontsize=7 )
ax2[2][1].set_xlabel(f'Comparison Roberts vs Prewitt using PFOM: {f5}',loc='right')

fig3, ax3 = plt.subplots(nrows=3, ncols=2, figsize=(10, 5))
#-----------------CANNY VS LAPLACE -----------
ax3[0][0].imshow(Canny_2, cmap='gray')
ax3[0][0].set_title('Canny', fontsize=7 )
ax3[0][1].imshow(Laplace_3, cmap='gray')
ax3[0][1].set_title('Laplace', fontsize=7 )
ax3[0][1].set_xlabel(f'Comparison Canny vs Laplace using PFOM: {f6}',loc='right')
#-----------------SOBEL VS LAPLACE -----------
ax3[1][0].imshow(Sobel_3, cmap='gray')
ax3[1][0].set_title('Sobel', fontsize=7 )
ax3[1][1].imshow(Laplace_3, cmap='gray')
ax3[1][1].set_title('Laplace', fontsize=7 )
ax3[1][1].set_xlabel(f'Comparison Sobel vs Laplace using PFOM: {f7}',loc='right')
#-----------------ROBERTS  VS LAPLACE -----------
ax3[2][0].imshow(Roberts_3, cmap='gray')
ax3[2][0].set_title('Roberts', fontsize=7 )
ax3[2][1].imshow(Laplace_3, cmap='gray')
ax3[2][1].set_title('Laplace', fontsize=7 )
ax3[2][1].set_xlabel(f'Comparison Roberts vs Laplace using PFOM: {f8}',loc='right')

print("Comparison Canny vs Canny-tiny noise: ",f1)
print("Comparison Canny vs Canny-big noise: ",f2)
print(" ")
print("Comparison Canny vs Prewitt: ",f3)
print("Comparison Sobel vs Prewitt: ",f4)
print("Comparison Roberts vs Prewitt: ",f5)
print(" ")
print("Comparison Canny vs Laplace: ",f6)
print("Comparison Sobel vs Laplace: ",f7)
print("Comparison Roberts vs Laplace: ",f8)

fig1.tight_layout(pad=1, w_pad=1, h_pad=2)
fig2.tight_layout()
fig3.tight_layout()
plt.show()
