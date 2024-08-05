import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file
audio_path = 'audio.wav'

#Load the audio as a waveform `y`
#Store the sampling rate as `sr`
y, sr = librosa.load(audio_path)

# Compute the Short-Time Fourier Transform (STFT)
D = librosa.stft(y)

# Convert the complex-valued STFT to magnitude
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Plot the spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.savefig('spectrogram.png')  # Save the spectrogram as an image
plt.show()
