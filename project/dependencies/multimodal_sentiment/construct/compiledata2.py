
import numpy as np
import random
from dependencies.multimodal_sentiment.extra import extract_dataset2, getFeatures
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


from numba import jit, cuda 


version = "1.1"
download_config = DownloadConfig(cache_dir="./cache_dir")
dataset = load_dataset("AbstractTTS/IEMOCAP", split="train", download_config=download_config)
dataset.cast_column("audio", Audio(sampling_rate=param.SAMPLE_RATE))
audio_data = dataset["audio"][:]
transcription_data = dataset["transcription"][:]
major_emotion = dataset["major_emotion"][:]
n = 5000 #math.floor(len(audio_data)/10)

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
        print(f"Estimated Time Remaining -  {datetime.timedelta(seconds=round((n-indexes.index(i)-1)*np.mean(process_times))) if not first else "--:--:--"}", end="\r")
        first = False
    return df

print("\n\n-----------------------------------------------\n\n")
path = f'./project/dependencies/multimodal_sentiment/datasets/v{version}.csv'
print("--> Starting Compilation...")
dataframe = compile_training_data()
print("--> Writing to file...                                  ")
dataframe.to_csv(path, index=True, header=True, sep=',', quoting=csv.QUOTE_ALL)
print("--> Done!")