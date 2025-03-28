import os
import numpy as np
import pandas as pd
import librosa
import zipfile

class AudioDataSet(list):
    def extract_features(self, data, sample_rate):
        features = [
            np.mean(librosa.feature.zero_crossing_rate(y=data), axis=1),  
            np.mean(librosa.feature.chroma_stft(y=data, sr=sample_rate), axis=1), 
            np.mean(librosa.feature.mfcc(y=data, sr=sample_rate), axis=1), 
            np.mean(librosa.feature.rms(y=data), axis=1), 
            np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate), axis=1)  
        ]
        return np.concatenate(features, axis=0)  

    def noise(self, data, noise_level=0.02):
        return data + noise_level * np.random.randn(len(data))

    def stretch(self, data, rate=0.8):
        return librosa.effects.time_stretch(data, rate)

    def pitch(self, data, sample_rate, n_steps=2):
        return librosa.effects.pitch_shift(data, sr=sample_rate, n_steps=n_steps)

    def get_features(self, path):
        data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
        
        res1 = self.extract_features(data, sample_rate)
        result = np.array(res1)
        
        noise_data = self.noise(data)
        res2 = self.extract_features(noise_data, sample_rate)
        result = np.vstack((result, res2))
        
        stretched_data = self.stretch(data)
        pitch_data = self.pitch(stretched_data, sample_rate)
        res3 = self.extract_features(pitch_data, sample_rate)
        result = np.vstack((result, res3))
        
        return result

zip_paths = {
    "RAVDESS": r"C:\Users\vasub\Downloads\RAVDESS.zip",
    "TESS": r"C:\Users\vasub\Downloads\archive (5).zip",
    "CREMA": r"C:\Users\vasub\Downloads\archive (6).zip",
    "SAVEE": r"C:\Users\vasub\Downloads\archive (7).zip"
}

extract_path = r"C:\Users\vasub\Downloads\extracted_datasets"

def extract_zip(zip_path, extract_to):
    """Extracts a ZIP file to the specified directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

for dataset, path in zip_paths.items():
    extract_zip(path, extract_path)

Ravdess = os.path.join(extract_path, "r5")
Tess = os.path.join(extract_path, "TESS Toronto emotional speech set data")
Crema = os.path.join(extract_path, "AudioWAV")
Savee = os.path.join(extract_path, "ALL")

emotion_labels = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fear', '07': 'disgust', '08': 'surprise'
}

audio_processor = AudioDataSet()
all_data = []

# RAVDESS Processing
if os.path.exists(Ravdess):
    print("Processing RAVDESS dataset...")
    for dir in os.listdir(Ravdess):
        actor_path = os.path.join(Ravdess, dir)
        if os.path.isdir(actor_path): 
            for file in os.listdir(actor_path):
                part = file.split('.')[0].split('-')
                if len(part) > 2 and part[2].isdigit():
                    emotion = emotion_labels.get(part[2], "Unknown")
                    file_path = os.path.join(actor_path, file)
                    features = audio_processor.get_features(file_path)
                    for feature_set in features:
                        all_data.append([emotion] + feature_set.tolist())

# CREMA Processing
if os.path.exists(Crema):
    print("Processing CREMA dataset...")
    for file in os.listdir(Crema):
        full_path = os.path.join(Crema, file)
        part = file.split('_')
        emotion = {'SAD': 'sad', 'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear', 'HAP': 'happy', 'NEU': 'neutral'}.get(part[2], 'Unknown')
        features = audio_processor.get_features(full_path)
        for feature_set in features:
            all_data.append([emotion] + feature_set.tolist())

# TESS Processing
if os.path.exists(Tess):
    print("Processing TESS dataset...")
    for dir in os.listdir(Tess):
        dir_path = os.path.join(Tess, dir)
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                part = file.split('.')[0].split('_')
                emotion = 'surprise' if part[2] == 'ps' else part[2]
                file_path = os.path.join(dir_path, file)
                features = audio_processor.get_features(file_path)
                for feature_set in features:
                    all_data.append([emotion] + feature_set.tolist())

# SAVEE Processing
if os.path.exists(Savee):
    print("Processing SAVEE dataset...")
    for file in os.listdir(Savee):
        file_path = os.path.join(Savee, file)
        part = file.split('_')[1]
        emotion = {'a': 'angry', 'd': 'disgust', 'f': 'fear', 'h': 'happy', 'n': 'neutral', 'sa': 'sad'}.get(part[:-6], 'surprise')
        features = audio_processor.get_features(file_path)
        for feature_set in features:
            all_data.append([emotion] + feature_set.tolist())

# Convert to DataFrame and save to CSV
df = pd.DataFrame(all_data)
df.to_csv("Audio_Features_Dataset.csv", index=False)
print("Feature extraction complete. Dataset saved as 'Audio_Features_Dataset.csv'")
