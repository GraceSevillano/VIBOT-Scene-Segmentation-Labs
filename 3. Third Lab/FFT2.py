import numpy as np
import matplotlib.pyplot as plt

# Define the frequency and sampling frequency
fs = 1/0.05
fg = 1000

# Compute the number of samples
Np = 10
N = int(np.ceil(Np*fg/fs))
if N%2 != 0:
    N += 1

# Generate the sin function
k = np.arange(0,N-1)
s = np.sin(2*np.pi*fs*k/fg)

# Change the frequency
fs = 1/0.05 + 0.01
s = np.sin(2*np.pi*fs*k/fg)

# Compute the DFT
dft = np.fft.fft(s)/(N-1)


# Display the amplitude
amplitude = np.abs(dft)

# Scale the x-axis in frequencies
frequencies = np.fft.fftfreq(N-1, 1/fg)

# Plot the amplitude spectrum
plt.plot(frequencies, amplitude)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Amplitude Spectrum')
plt.xlim(0, fg/2)
plt.show()

# Compute the Total Harmonic Distortion
A1 = amplitude[0]
A2 = np.sqrt(np.sum(amplitude[1:]**2))
THD = A1/A2

print("Total Harmonic Distortion: {:.3f}".format(THD))
