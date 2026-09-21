import wave
import pyaudio
import numpy as np
from datasets import load_dataset
import librosa
import tensorflow as tf
print("GPUs available:", tf.config.list_physical_devices('GPU'))
# Load the dataset
dataset = load_dataset("AbstractTTS/IEMOCAP", split="train")

# Extract the audio and metadata for the first item
audio_data = dataset["audio"][0]
audio_samples = audio_data["array"]  # This should be a numpy array
sampling_rate = audio_data["sampling_rate"]

# Ensure the audio samples are in int16 format (or adjust to the format required)
# If the data is not int16, it might need scaling or conversion
if audio_samples.dtype != np.int16:
    audio_samples = np.int16(audio_samples / np.max(np.abs(audio_samples)) * 32767)  # Scale to int16 range



'''
# Write to WAV file
wf = wave.open("./project/dependencies/multimodal_sentiment/output.wav", 'wb')
wf.setnchannels(1)  # Mono audio (set to 2 for stereo)
wf.setsampwidth(2)  # 2 bytes for int16 format
wf.setframerate(sampling_rate)

# Write frames as bytes
wf.writeframes(audio_samples.tobytes())
wf.close()
'''

y, sr = librosa.load('./project/dependencies/multimodal_sentiment/output.wav', sr=None)  # sr=None keeps the original sampling rate

print(y)
print()