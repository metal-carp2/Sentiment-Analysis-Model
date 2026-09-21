"""Shared audio preprocessing and prediction for file and microphone input."""
import json
from pathlib import Path
import numpy as np
import librosa

MODEL_DIR = Path(__file__).resolve().parent/'project/dependencies/multimodal_sentiment/model'
SAMPLE_RATE = 16000


def features(audio, sample_rate=SAMPLE_RATE):
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.ndim != 1 or len(y) == 0 or not np.isfinite(y).all():
        raise ValueError('Expected nonempty finite mono or stereo audio.')
    if sample_rate != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    if len(y) < 2048:
        raise ValueError('At least 0.13 seconds of audio is required.')
    # Matches extract_dataset2, used for the recovered 40-value feature data.
    y = librosa.util.normalize(y)
    return librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=40).mean(axis=1).astype(np.float32)


class Predictor:
    def __init__(self, model_dir=MODEL_DIR):
        import tensorflow as tf
        model_dir = Path(model_dir)
        self.metadata = json.loads((model_dir/'metadata.json').read_text())
        self.labels = self.metadata['labels']
        self.model = tf.keras.models.load_model(model_dir, compile=False)
        if self.model.input_shape[-1] != 40 or self.model.output_shape[-1] != len(self.labels):
            raise ValueError('Model and metadata shapes disagree.')

    def predict(self, audio, sample_rate=SAMPLE_RATE):
        probabilities = self.model(features(audio, sample_rate)[None], training=False).numpy()[0]
        if not np.isfinite(probabilities).all():
            raise ValueError('Model returned nonfinite scores.')
        index = int(np.argmax(probabilities))
        return {'emotion': self.labels[index], 'score': float(probabilities[index]),
                'scores': dict(zip(self.labels, map(float, probabilities)))}
