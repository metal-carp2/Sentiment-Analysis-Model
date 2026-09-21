import numpy as np
import whisper
import noisereduce as nr
import dependencies.parameters as param
import time
import librosa

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base.en").to(device)

def transcribeAudio(reduced_noise_audio, iter_callback, voiceStartTime=0):
    # Check if the audio array is valid and contains data
    if reduced_noise_audio is None or len(reduced_noise_audio) == 0 or np.allclose(reduced_noise_audio, 0):
        print("Error: Received empty or invalid audio data for transcription. Skipping...")
        return None
    try:
        result = model.transcribe(reduced_noise_audio, task="transcribe", fp16=False)
        for segment in result["segments"]:
            if voiceStartTime == 0:
                iter_callback(segment)
            else:
                iter_callback(segment, voiceStartTime)
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def process_audio_chunk(buffer, callback):

    # Combine the buffer into a single audio stream (bytes)
    if not buffer:  # Ensure the buffer is not empty
        print("Error: Buffer is empty. Skipping audio processing.")
        return
    audio_data = b''.join(buffer)

    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0  # Convert bytes to np.ndarray (16-bit PCM to float32)
    reduced_noise_audio = nr.reduce_noise(y=audio_array, sr=param.SAMPLE_RATE, prop_decrease=0.5)


    # Check if noise reduction yielded usable data
    if (reduced_noise_audio is None or len(reduced_noise_audio) == 0 or np.allclose(reduced_noise_audio, 0)):
        print("Warning: Noise-reduced audio is invalid or silent. Skipping transcription.")
        return

    # Start transcription in a separate thread

    callback(reduced_noise_audio)

def __default(var):
    pass

def analyze_pitch_volume(audio_chunk, start_time, loaded_set_start = __default):
    
    audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0  # Convert bytes to np.ndarray
    reduced_noise_audio = nr.reduce_noise(y=audio_array, sr=param.SAMPLE_RATE, prop_decrease=0.5)


    # Calculate pitch using YIN
    pitch = librosa.yin( 
        reduced_noise_audio, 
        fmin=param.FMIN, 
        fmax=param.FMAX, 
        sr=param.SAMPLE_RATE, 
        frame_length=param.FRAME_LENGTH
    ) 
    avg_pitch = np.mean(pitch[np.isfinite(pitch)])  # Avoid NaN/inf issues
    
    # Calculate MFCC
    mfcc = librosa.feature.mfcc(
        y=reduced_noise_audio, 
        sr=param.SAMPLE_RATE,
        n_mfcc=13, 
        hop_length=param.FRAME_LENGTH // 2
    )
    avg_mfcc = np.mean(mfcc, axis=1)  # Average MFCCs across the frame
    rms = np.sqrt(np.mean(reduced_noise_audio**2))  # Calculate volume (RMS)
    
    if loaded_set_start is not __default:
        start_time = time.time()
        loaded_set_start(start_time)
    #Logarithm of zero is undefined; represents silence in dB || Ref is 1.0 since audio is normalized
    #volume_db = -np.inf if rms == 0 else 20 * np.log10(rms / param.REF)

    # timestamp, pitch, volume, mfcc
    return time.time() - start_time, avg_pitch, rms, avg_mfcc