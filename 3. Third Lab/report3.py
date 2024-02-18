import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Generate sin signal with period Np and frequency fs
fg = 1000
fs = 100
T = 1/fs
Np = 5
N = 2*Np*fg
k = np.arange(N)
s = np.sin(2*np.pi*fs*k*T)

# Compute the DFT and display amplitude
S = np.fft.fft(s)
freq = np.fft.fftfreq(N, T)
plt.stem(freq, np.abs(S))

# Change signal frequency and compute DFT
fs2 = 200
s2 = np.sin(2*np.pi*fs2*k*T)
S2 = np.fft.fft(s2)
plt.figure()
plt.stem(freq, np.abs(S2))

# Adjust N for best THD
Np = 3
N = 2*Np*fg
k = np.arange(N)
s3 = np.sin(2*np.pi*fs2*k*T)
S3 = np.fft.fft(s3)
A1 = np.abs(S3[1])
A2 = np.abs(S3[2])
A3 = np.abs(S3[3])
THD = np.sqrt(A2**2 + A3**2)/A1
plt.figure()
plt.stem(freq[0:N//2], np.abs(S3[0:N//2]))

# Load file and analyze DFT
import scipy.io
data = scipy.io.loadmat('C:/Users/ksevi/OneDrive/Desktop/MASTER/SCENE_LALIGANT/tercer report/100m.mat')
#val= data['val'] 
#value = mdata[0][:20]
val = data['val'][0]
plt.figure()
plt.plot(val)
N = 1000
k = np.arange(N)
s4 = val[0:N]
S4 = np.fft.fft(s4)
freq = np.fft.fftfreq(N, 1/360)
plt.figure()
plt.stem(freq[0:N//2], np.abs(S4[0:N//2]))

# Cancel 60Hz frequency
#b, a = butter(4, 2*np.pi*60, 'high', fs=360)
b, a = butter(4, 2*np.pi*40/360, 'high', fs=360)

s5 = filtfilt(b, a, s4)
S5 = np.fft.fft(s5)
plt.figure()
plt.plot(s5)

# Denoise signal with Butterworth filter
cutoff = 20
b, a = butter(4, 2*np.pi*cutoff, 'low', fs=360)
S6 = S5.copy()
S6[(freq > cutoff)] = 0
s6 = np.fft.ifft(S6)
s6 = np.real(s6)
s6 = filtfilt(b, a, s6)
plt.figure()
plt.plot(s6)
