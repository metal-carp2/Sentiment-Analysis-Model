import numpy as np
import random
from dependencies.multimodal_sentiment.extra import extract_dataset, getFeatures
import tensorflow as tf
from datasets import load_dataset, Audio
import dependencies.parameters as param
from datasets import DownloadConfig
from model_data import serialized_emotions
import json
import time
import pickle
import datetime
from dependencies.livedebug import show_live_debug_display
import math
import librosa



download_config = DownloadConfig(cache_dir="./cache_dir")
dataset = load_dataset("AbstractTTS/IEMOCAP", split="train", download_config=download_config)
dataset.cast_column("audio", Audio(sampling_rate=param.SAMPLE_RATE))
audio_data = dataset["audio"][:]
transcription_data = dataset["transcription"][:]
n = 100 #math.floor(len(audio_data)/10)


def compile_training_data():
    """
    Prepares batches of training data and compiles the dataset.
    dataset: The full training dataset (IEMOCAP or similar).
    batch_size: The number of samples per batch.
    serialized_emotions: List or dict that maps emotions to labels.
    sample_rate: The sample rate of the audio data.
    n_epochs: The number of training epochs (default: 1).
    """

    # Shuffle indexes to ensure randomness in training
    indexes = random.sample(range(len(dataset)), n)
    
    output = []
    process_times = np.array([])
    first = True
    print(f"Starting Compilation - {n}")
    for i in indexes:
        last_time = round(time.time(), 1)
        # Extract audio features for each row in the batch

        aud_set, array = extract_dataset(audio_data[i], transcription_data[i])  # Assuming 'audio' field holds data

        extracting_time = round(time.time(), 1)
        features = getFeatures(aud_set.ranges)
        features_time = round(time.time(), 1)
        appended_features = {}
        # Append features to the batch features dictionary
        appended_features["pitch_data"] = (features["pitch_normalized"])
        appended_features["pitch_mean"] = (features["pitch_mean"])
        appended_features["pitch_std"] = (features["pitch_std"])
        appended_features["pitch_range"] = (features["pitch_range"])
        appended_features["volume_data"] = (features["volume_normalized"])
        appended_features["volume_mean"] = (features["volume_mean"])
        appended_features["volume_std"] = (features["volume_std"])
        appended_features["volume_range"] = (features["volume_range"])
        appended_features["mfcc_data"] = (features["mfcc_normalized"])
        appended_features["mfcc_mean"] = (features["mfcc_mean"])
        appended_features["mfcc_std"] = (features["mfcc_std"])
        #appended_features["dialogue_embedding"] = (features["dialogue_embedding"])
        mfcc = librosa.feature.mfcc(y=array, sr=param.SAMPLE_RATE, n_mfcc=40)
        appended_features["mfcc_revised0"] = np.mean(mfcc, axis=0)
        appended_features["mfcc_revised1"] = np.mean(mfcc, axis=1)
        appended_features["mfcc_revised2"] = mfcc
        appended_features["mfcc_revised3"] = mfcc.T


        # Assuming 'emotion' field contains the correct label for the sample
        assigning_time = round(time.time(), 1)
        appended_labels = [dataset[emotion][i] for emotion in serialized_emotions]
        y_labels_time = round(time.time(), 1)
        # Once we have a full batch, convert to numpy arrays for efficiency

        # Return this batch for training
        output.append([appended_features, appended_labels])
        batching_time = round(time.time(), 1)
        print(f"Finished in {round(time.time(), 1)-last_time}s - " + str(i))
        print(f"Stall Time: (debug): \n   Extraction - {extracting_time-last_time}\n   Features - {features_time-extracting_time}\n   Assigning - {assigning_time-features_time}\n   Y_Labels - {y_labels_time-assigning_time}\n   Batching - {batching_time-y_labels_time}\n\n")
        if not first:
            process_times = np.append(process_times,round(time.time(), 1)-last_time)
        print(f"Estimated Time Remaining -  {datetime.timedelta(seconds=(n-indexes.index(i)-1)*np.mean(process_times)) if not first else "--:--:--"}\n")
        first = False
    return output



output = open('./project/dependencies/multimodal_sentiment/compiled_training_data2.pkl', 'wb')
pickle.dump(list(compile_training_data()), output)
output.close()