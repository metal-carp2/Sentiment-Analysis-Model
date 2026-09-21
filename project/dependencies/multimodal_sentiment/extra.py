from scipy.interpolate import interp1d
from typing import List
from sentence_transformers import SentenceTransformer  # Or any preferred NLP model
import dependencies.audiodata as aud  # Assuming this module is accessible
import librosa
import numpy as np
import librosa.display
import noisereduce as nr
import dependencies.parameters as param
from dependencies.recording import transcribeAudio
from joblib import Parallel, delayed
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import threading
from sklearn.preprocessing import StandardScaler

# Ensure VADER lexicon is downloaded
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import tokenize
nltk.download('vader_lexicon')
nltk.download('punkt_tab')

import re

def preprocess_for_sentence_transformers(text):
    """
    Preprocesses and splits long text into a list of sentences for Sentence Transformers.
    
    Args:
        text (str): The input text to preprocess and split.
    
    Returns:
        list: A list of preprocessed sentences suitable for encoding.
    """
    # Clean up text: remove extra spaces and unwanted characters
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s.,!?]", "", text)  # Remove unwanted special characters
    
    # Regular expression to split text into sentences
    sentence_endings = r"(?<=[.!?]) +"
    sentences = re.split(sentence_endings, text)
    
    # Post-process sentences: strip whitespace and lowercase
    preprocessed_sentences = [sentence.strip().lower() for sentence in sentences]
    
    return preprocessed_sentences



# Initialize VADER Sentiment Intensity Analyzer
def sentiment_percentages(text):
    # Initialize VADER Sentiment Intensity Analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Tokenize text into sentences
    sentences = tokenize.sent_tokenize(text)
    
    # Counters for positive, neutral, and negative sentiments
    positive_count = 0
    neutral_count = 0
    negative_count = 0

    # Analyze each sentence
    for sentence in sentences:
        scores = sia.polarity_scores(sentence)
        
        # Classify based on the compound score
        if scores['compound'] > 0.05:  # Positive sentiment threshold
            positive_count += 1
        elif scores['compound'] < -0.05:  # Negative sentiment threshold
            negative_count += 1
        else:  # Neutral sentiment
            neutral_count += 1
    
    # Total sentences
    total_sentences = len(sentences)

    # Calculate percentages
    positive_percent = (positive_count / total_sentences) * 100
    neutral_percent = (neutral_count / total_sentences) * 100
    negative_percent = (negative_count / total_sentences) * 100

    # Return the results
    return {
        'positive': round(positive_percent, 2),
        'neutral': round(neutral_percent, 2),
        'negative': round(negative_percent, 2),
    }

# Function definition
def getFeatures(ranges: List[aud.AudioDataRange]):
    # Flatten the input ranges
    flattened = aud.joinRanges(ranges)

    timestamps = np.array([r.timestamp for r in flattened])  # Seconds
    pitch_data = np.array([r.pitch for r in flattened])  # Hz
    volume_data = np.array([r.volume for r in flattened])  # dB
    mfcc_data = np.array([r.mfcc for r in flattened])  # Average MFCC values
    dialogues = [r.dialogue for r in flattened.ranges]  # Dialogue texts

    # Reduce redundant computation of the Sentence Transformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Calculate the embeddings for all dialogues in one batch
    dialogue_embeddings = model.encode(dialogues, show_progress_bar=False, batch_size=64)  # Batching for speed

    vader_ratings = [sentiment_percentages(dialogue).values() for dialogue in dialogues]
    # Aggregate embeddings (Optional: averaging across ranges)
    avg_dialogue_embedding = np.mean(dialogue_embeddings, axis=0)

    # Pitch, Volume, and MFCC Calculations (same as previous)
    pitch_baseline = np.median(pitch_data)
    normalized_pitch = (pitch_data - pitch_baseline) / pitch_baseline
    pitch_mean = np.mean(normalized_pitch)
    pitch_std = np.std(normalized_pitch)
    pitch_range = np.max(normalized_pitch) - np.min(normalized_pitch)

    volume_baseline = np.median(volume_data)
    normalized_volume = (volume_data - volume_baseline) / volume_baseline
    volume_mean = np.mean(normalized_volume)
    volume_std = np.std(normalized_volume)
    volume_range = np.max(normalized_volume) - np.min(normalized_volume)

    mfcc_baseline = np.median(mfcc_data)
    normalized_mfcc = (mfcc_data - mfcc_baseline) / mfcc_baseline
    mfcc_mean = np.mean(normalized_mfcc, axis=0)
    mfcc_std = np.std(normalized_mfcc, axis=0)

    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Collect VADER sentiment scores for each dialogue
    vader_scores = [sia.polarity_scores(dialogue) for dialogue in dialogues]

    # Extract compound scores (overall sentiment)
    compound_scores = [score['compound'] for score in vader_scores]
    compound_mean = np.mean(compound_scores)
    compound_std = np.std(compound_scores)

    
    # Add extracted features to the dictionary
    features = {
        "pitch_normalized": normalized_pitch,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "pitch_range": pitch_range,
        "volume_normalized": normalized_volume,
        "volume_mean": volume_mean,
        "volume_std": volume_std,
        "volume_range": volume_range,
        "mfcc_normalized": normalized_mfcc,
        "mfcc_mean": mfcc_mean.tolist(),
        "mfcc_std": mfcc_std.tolist(),
        "dialogue_embedding": avg_dialogue_embedding.tolist(),  # Aggregated dialogue embedding
        "vader_compound_scores": vader_ratings,  # List of compound scores
    }

    return features

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_dataset3(audio, text):
    # Resample audio and apply noise reduction
    y = librosa.resample(librosa.util.normalize(audio["array"].astype('float32')), orig_sr=audio["sampling_rate"], target_sr=param.SAMPLE_RATE)
    
    # Precompute MFCCs once
    mfcc = librosa.feature.mfcc(y=y, sr=param.SAMPLE_RATE, n_mfcc=param.N_MFCC)

    # Calculate the embeddings for all dialogues in one batch
    dialogue_embeddings = model.encode(preprocess_for_sentence_transformers(text), show_progress_bar=False, batch_size=64)  # Batching for speed

    vader_ratings = sentiment_percentages(text)
    avg_mfcc = np.mean(mfcc.T, axis=1)
    return avg_mfcc, dialogue_embeddings, vader_ratings

def extract_dataset2(audio, text):
    # Resample audio and apply noise reduction
    y = librosa.resample(librosa.util.normalize(audio["array"].astype('float32')), orig_sr=audio["sampling_rate"], target_sr=param.SAMPLE_RATE)
    #y_denoised = nr.reduce_noise(y=y, sr=param.SAMPLE_RATE, prop_decrease=0.5).astype(np.float32)
    
    # Precompute MFCCs once
    mfcc = librosa.feature.mfcc(y=y, sr=param.SAMPLE_RATE, n_mfcc=param.N_MFCC)

    # Normalize the MFCC coefficients
    avg_mfcc = np.mean(mfcc, axis=1)
    # Calculate the embeddings for all dialogues in one batch
    dialogue_embeddings = model.encode(preprocess_for_sentence_transformers(text), show_progress_bar=False, batch_size=64)  # Batching for speed

    vader_ratings = sentiment_percentages(text)

    return avg_mfcc, dialogue_embeddings, vader_ratings


def extract_dataset(audio, text):
    # Resample audio and apply noise reduction
    y = librosa.resample(audio["array"], orig_sr=audio["sampling_rate"], target_sr=param.SAMPLE_RATE)
    y_denoised = nr.reduce_noise(y=y, sr=param.SAMPLE_RATE, prop_decrease=0.5).astype(np.float32)

    # Split audio into chunks
    chunk_length = int(param.SAMPLE_RATE * param.CHUNK_DURATION)
    num_chunks = len(y_denoised) // chunk_length + (1 if len(y_denoised) % chunk_length else 0)
    timestamps = np.arange(num_chunks) * param.CHUNK_DURATION
    
    # Precompute MFCCs once
    mfcc = librosa.feature.mfcc(y=y_denoised, sr=param.SAMPLE_RATE, n_mfcc=param.N_MFCC, hop_length=param.FRAME_LENGTH // 2)

    # Helper to process a single chunk
    def process_chunk(index):
        start = index * chunk_length
        end = start + chunk_length
        frame = y_denoised[start:end]

        # Skip empty chunks
        if len(frame) == 0:
            return None

        # Pitch and RMS
        pitch = librosa.yin(frame, fmin=param.FMIN, fmax=param.FMAX, sr=param.SAMPLE_RATE)
        avg_pitch = np.mean(pitch[np.isfinite(pitch)])
        rms = np.sqrt(np.mean(frame**2))
        
        start_frame = start // (param.FRAME_LENGTH // 2)
        end_frame = min(mfcc.shape[1], end // (param.FRAME_LENGTH // 2))
        avg_mfcc = np.mean(mfcc[:, start_frame:end_frame], axis=1)

        return timestamps[index], avg_pitch, rms, avg_mfcc

    # Process chunks in parallel
    results = Parallel(n_jobs=-1)(delayed(process_chunk)(i) for i in range(num_chunks))

    # Create the dataset
    stamps = aud.AudioDataSet()
    for res in results:
        if res:
            stamps.add_frame(*res)

    stamps.add_range(0, len(stamps), text)
    return stamps, y_denoised
