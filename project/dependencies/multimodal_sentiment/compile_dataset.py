
import numpy as np
import random
from dependencies.multimodal_sentiment.extra import extract_dataset2, extract_dataset3, getFeatures
import tensorflow as tf
from datasets import load_dataset, Audio
import dependencies.parameters as param
from datasets import DownloadConfig
from dependencies.multimodal_sentiment.model_data import serialized_emotions
import json
import time
import pickle
import datetime
from dependencies.livedebug import show_live_debug_display
import math
import librosa
import pandas as pd
import csv
from keras.preprocessing.sequence import pad_sequences


version = "3.0"
download_config = DownloadConfig(cache_dir="./cache_dir")
dataset = load_dataset("AbstractTTS/IEMOCAP", split="train", download_config=download_config)
dataset.cast_column("audio", Audio(sampling_rate=param.SAMPLE_RATE))
audio_data = dataset["audio"][:]
transcription_data = dataset["transcription"][:]
major_emotion = dataset["major_emotion"][:]
n = len(audio_data)
import base64
import json

def encode_large_data(cell):
    json_string = json.dumps(cell)  # Convert list to JSON string
    encoded = base64.b64encode(json_string.encode('utf-8')).decode('utf-8')
    return encoded

def compile_training_data():
    # Shuffle indexes to ensure randomness in training
    indexes = random.sample(range(len(dataset)), n)
    process_times = np.array([])
    first = True
    df = pd.DataFrame(columns=["text", "mfcc", "text_embedding", "vader", "label", "major_emotion"])
    for i in indexes:
        last_time = round(time.time(), 1)
        # Extract audio features for each row in the batch
        array = audio_data[i]
        text = transcription_data[i]
        mfcc, text_embedding, vader = extract_dataset2(array, text)
        appended_labels = [dataset[emotion][i] for emotion in serialized_emotions]

        df.loc[len(df)] = [text, mfcc.tolist(), text_embedding.tolist(), vader, appended_labels, major_emotion[i]]
        if not first:
            process_times = np.append(process_times,round(time.time(), 1)-last_time)
        print(f"Estimated Time Remaining -  {datetime.timedelta(seconds=round((n-indexes.index(i)-1)*np.mean(process_times))) if not first else '--:--:--'}  -", end="\r")
        first = False
    print("Encoding Data...                                   ", end="\r")
    df['mfcc'] = df['mfcc'].apply(encode_large_data)
    df['text_embedding'] = df['text_embedding'].apply(encode_large_data)
    df['label'] = df['label'].apply(encode_large_data)
    print("Success!                                           ", end="\r")
    return df

print("\n\n-----------------------------------------------\n\n")
from pathlib import Path
path = Path(__file__).resolve().parent / 'datasets' / f'v{version}.csv'
print("--> Starting Compilation...")
dataframe = compile_training_data()
print("--> Writing to file...                                  ")
dataframe.to_csv(path, index=True, header=True, sep=',', quoting=csv.QUOTE_ALL)
print("--> Done!")