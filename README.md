# Speech Emotion Sessions

An installable terminal application for separate recording sessions of up to one minute, with an emotion timeline and end-of-session report. Processing runs locally. The bundled model is experimental: held-out accuracy is **32.3%**, so reports are estimates, not reliable measurements of feelings.

## Install

Python **3.10 or 3.11** is required (3.11 recommended). From a terminal with Python and Git installed:

```sh
python -m pip install "git+https://github.com/metal-carp2/Sentiment-Analysis-Model.git"
emotion-session --help
```

The model is included; no training or separate model download is needed. This installs from GitHub, not PyPI. Tested on Windows with Python 3.11; macOS/Linux are not yet verified. Linux may require the system PortAudio package for microphone capture. TensorFlow installation is large and availability depends on platform/CPU.

For your existing checkout and virtual environment:

```powershell
git pull
uv pip install --python .venv\Scripts\python.exe .
.\.venv\Scripts\emotion-session.exe record --name first-session --seconds 60
```

## Record and review sessions

```sh
emotion-session record --name morning --seconds 60
emotion-session record --name afternoon --seconds 30
emotion-session list
emotion-session report morning
```

Wait for **Recording. Speak now**. Ctrl+C ends early and analyzes what was captured. Duration is limited to 1-60 seconds. Each name must be unique; omitting `--name` generates one. The microphone is closed before analysis starts.

Each session creates a directory under `emotion-sessions/` in the current working directory:

- `audio.wav`: your recording.
- `report.txt`: readable timeline and summary.
- `report.json`: structured results, model metadata, and scores.

Use `--sessions-dir PATH` on any session command to choose a shared location. Audio and reports stay local and may contain private information; they are ignored by Git by default.

Reports analyze non-overlapping three-second windows, include any usable final partial window, and report quiet periods separately. Emotion percentages are the share of **analyzed time assigned to each label**, not the probability that you felt that emotion. Loud background sounds can exceed the simple RMS gate. Scores are not calibrated confidence.

```sh
emotion-session devices
emotion-session record --seconds 30 --device 1
emotion-session analyze samples/03-01-05-01-01-01-01.wav --name sample
```

Device indexes depend on your computer. `analyze` accepts audio up to 60 seconds. Sample files are in the Git checkout and have a [separate audio license](samples/README.md). Use `--window 5` to adjust report resolution or `--silence-threshold 0.0005` for quiet recordings.

## Replace the model

Recording and reporting depend only on a predictor interface. They do not contain model architecture or feature extraction logic.

**Same 40-MFCC feature pipeline:** supply another TensorFlow SavedModel bundle with `metadata.json`, unique `labels`, `sample_rate: 16000`, `n_mfcc: 40`, `aggregation: "mean_over_time"`, and `audio_normalization: "peak"`. It must accept `(batch, 40)` MFCC means, include its own training normalization, and return one probability per label. Use the bundled metadata as a reference.

```sh
emotion-session record --seconds 60 --model /path/to/model-bundle
```

**Different features, labels, or framework:** provide an installed Python module with a factory, then select it explicitly:

```sh
emotion-session record --seconds 60 --backend my_emotion_backend:create --model /path/to/artifacts
```

The factory receives a `Path` (or `None`) and returns an object with:

```python
class Predictor:
    labels = ["neutral", "happy", "sad"]
    metadata = {"model_id": "my-model-v1"}  # JSON-serializable

    def predict(self, audio, sample_rate):
        # audio is a mono float32 NumPy array; perform your own resampling,
        # feature extraction, normalization, and model inference here.
        return {"scores": {"neutral": 0.7, "happy": 0.2, "sad": 0.1}}

def create(model_path):
    return Predictor()  # Load your artifacts here.
```

This is an interface example, not a trained classifier. Scores must be finite values in [0, 1], sum to one, and match the declared labels. Only use backend modules you trust: selecting one executes Python code. Audio+text models can transcribe inside their backend. No changes to session storage or reports are required. Incompatible preprocessing metadata is rejected by the default backend.

For training, `construct_model.py` defines the model and `train.py` handles the supplied CSV. Saved normalization, feature extraction, and label order must stay consistent when retraining. Architecture changes alone do not establish improved accuracy; compare on held-out data.

## Development and validation

```sh
python -m pip install -e .
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Wheel installation and bundled-model inference were tested outside the source repository. Session tests cover duration limits, no overwrites, early stopping with partial audio, quiet windows, invalid backend scores, and report persistence. Physical microphone capture has not been exercised during automated validation.

## What was trained

The supplied CSV has 1,024 examples. Two `other` examples were excluded, leaving eight labels: angry, excited, fear, frustrated, happy, neutral, sad, and surprise. Training used 721 examples, validation 146, and held-out testing 155. Normalized transcript groups are disjoint across these splits. Speaker identities are absent, so this is **not a speaker-independent evaluation**.

The model takes 40 MFCC means and uses a small dense neural network suited to that fixed-size input. It replaces the untrained sequence-model placeholder. Audio is resampled to 16 kHz, peak-normalized, and converted to mean MFCCs. Training-only feature normalization is embedded in the saved model. The preprocessing was reconstructed from the recovered `extract_dataset2`; original CSV-generation provenance is not fully documented. Live windows also differ from the complete utterances in the training CSV.

Training stopped after 31 epochs, restoring the best validation-loss weights. Test results:

| Metric | Result |
| --- | --- |
| Accuracy | 32.3% |
| Majority-label baseline accuracy | 26.5% |
| Balanced accuracy | 24.6% |
| Macro F1 | 22.9% |

**This is a functioning prototype, not a reliable emotion detector.** Fear and surprise had zero recall in the small held-out set. Scores are not calibrated confidence, and microphone/domain differences may reduce performance further. File replay success does not establish live accuracy. Model scores were not used to choose the data split.

Full evaluation, label order, preprocessing settings, source-data hash, and optimizer step count are saved in `speech_emotion/model/metadata.json` alongside the model. No separate scaler file is needed.

## Code and retraining

- `live_emotion.py`: bounded microphone buffer, quiet-window gating, and WAV replay.
- `emotion_model.py`: shared preprocessing and model loading/prediction.
- `test_audio.py`: recording and whole-file testing.
- `project/dependencies/multimodal_sentiment/construct_model.py`: single model definition.
- `project/dependencies/multimodal_sentiment/train.py`: reproducible training/evaluation on the supplied CSV.
- `speech_emotion/model/`: single trained model and metadata, shipped with the package.
- `project/dependencies/multimodal_sentiment/compile_dataset.py`: optional original IEMOCAP/text compiler, now using compatible 40-MFCC means; not needed to run or retrain from the supplied CSV. It still requires additional research dependencies.
- `project/main.py` and the other recovered utilities: original Whisper/transcription experiments; `emotion-session record` is the supported session entry point; `live_emotion.py` remains available for continuous estimates.

To reproduce training without overwriting the shipped model:

```powershell
.\.venv\Scripts\python.exe project/dependencies/multimodal_sentiment/train.py --output work/retrained-model
```

The output must not already exist. Raw dataset archives were not used for this training run; the supplied prepared CSV was used directly. Text embeddings are not inputs to this audio-only model.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Validation covers audio input, silence gating, group-disjoint splits, model loading with embedded normalization, and finite prediction scores. WAV replay is tested with real speech; a physical microphone recording was not performed during development.
