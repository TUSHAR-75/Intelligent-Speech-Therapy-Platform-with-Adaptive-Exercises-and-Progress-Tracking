import librosa
import matplotlib.pyplot as plt

# Path to sample audio
audio_path = "data/sample_audio/sample.wav"

# Load audio
signal, sr = librosa.load(audio_path, sr=None)

# Duration
duration = len(signal) / sr

print("Sample Rate:", sr)
print("Duration (seconds):", round(duration, 2))

# Plot waveform
plt.figure(figsize=(10, 4))
plt.plot(signal)
plt.title("Audio Waveform")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
