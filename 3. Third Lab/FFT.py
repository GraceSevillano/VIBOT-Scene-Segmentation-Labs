import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
#mat = scipy.io.loadmat('')
 #mat = sio.loadmat('C:/Users/ksevi/OneDrive/Desktop/MASTER/SCENE_LALIGANT/tercer report/100m.mat')# load mat-file
#mdata = mat['val']  # variable in mat file 
#si=sio.whosmat(mat)
#print(si)
#
#value = mdata[0][:20]

# Define the frequency and sampling frequency
T  = 0.01
fs = 5
fg = 100
Tg = 1/ fg


# Compute the number of samples
Np = 1
N = int((Np*fg/fs))   
print(N)
if N%2 != 0:
    N += 1

# Generate the sin function
k = np.arange(N)
#print(k.dtype)
#print(k)
#s = np.sin(2*np.pi*fs) #cambiar aqui

s = np.sin(2*np.pi*38*k/fg)

#s = np.sin(2*np.pi*38*k/fg)
#print(value)

dft = np.fft.fft(s)/N
dft1=np.fft.fftshift(dft)

#dft_val = np.fft.fft(value)/N

# Display the amplitude
amplitude = np.abs(dft)
#amplitude_val = np.abs(dft_val)

# Scale the x-axis in frequencies
frequencies = np.fft.fftfreq(N, 1/fg)
print(frequencies)
#dft1=np.fft.fftshift(frequencies)
#print(frequencies)
# Plot the amplitude spectrum
#plt.plot(s)
plt.stem(frequencies,amplitude)
plt.xlabel('Time')
plt.ylabel('Amplitude')
#plt.xlim(0, fg/2)


#A1 = amplitude[0]
#A2 = np.sqrt(np.sum(amplitude[1:]**2))
#THD = A1/A2
#plt.title('Amplitude Spectrum fs=38  THD = {:.3f}'.format(THD))
#print("Total Harmonic Distortion: {:.3f}".format(THD))



plt.show()