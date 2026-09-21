# Live Speech Emotion Recognition

A trained experimental audio emotion model using the supplied `v2.1.csv`, with live microphone inference and WAV replay. The repository keeps one model and one model constructor.

## Run live predictions

In PowerShell, from the repository root:

```powershell
git pull
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\python.exe live_emotion.py
```

For a fresh checkout, first create the environment with `uv venv --python 3.11 .venv` (or `py -3.11 -m venv .venv`).

Speak after **Microphone listening** appears. The program analyzes the latest three seconds of audio and updates about once per second, depending on processing speed. It prints an emotion estimate and model scores in the terminal. Ctrl+C stops. Quiet windows are skipped. Live mode captures audio in memory; it does not save recordings or transcribe speech.

To choose a microphone or stop after a fixed duration:

```powershell
.\.venv\Scripts\python.exe live_emotion.py --list-devices
.\.venv\Scripts\python.exe live_emotion.py --device 1 --seconds 20
```

The device index depends on your computer. Adjust `--silence-threshold` if a quiet microphone is consistently skipped (default RMS threshold: 0.001).

## Test without a microphone

```powershell
.\.venv\Scripts\python.exe live_emotion.py --wav samples/03-01-05-01-01-01-01.wav
.\.venv\Scripts\python.exe test_audio.py --wav samples/03-01-01-01-01-01-01.wav --check-model
```

The first command replays overlapping audio windows through the same predictor used live; the second evaluates the whole clip. Three real neutral/happy/angry clips are included; see [their attribution and separate audio license](samples/README.md).

To save a five-second recording for replay:

```powershell
.\.venv\Scripts\python.exe test_audio.py --record 5
.\.venv\Scripts\python.exe live_emotion.py --wav recordings/test.wav
```

Choose a different `--output` when recording again; existing files are not overwritten.

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

Full evaluation, label order, preprocessing settings, source-data hash, and optimizer step count are saved in `project/dependencies/multimodal_sentiment/model/metadata.json` alongside the model. No separate scaler file is needed.

## Code and retraining

- `live_emotion.py`: bounded microphone buffer, quiet-window gating, and WAV replay.
- `emotion_model.py`: shared preprocessing and model loading/prediction.
- `test_audio.py`: recording and whole-file testing.
- `project/dependencies/multimodal_sentiment/construct_model.py`: single model definition.
- `project/dependencies/multimodal_sentiment/train.py`: reproducible training/evaluation on the supplied CSV.
- `project/dependencies/multimodal_sentiment/model/`: single trained model and metadata.
- `project/dependencies/multimodal_sentiment/compile_dataset.py`: optional original IEMOCAP/text compiler, now using compatible 40-MFCC means; not needed to run or retrain from the supplied CSV. It still requires additional research dependencies.
- `project/main.py` and the other recovered utilities: original Whisper/transcription experiments; the supported live prediction entry point is `live_emotion.py`.

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
