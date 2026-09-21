# Sentiment Analysis Model

Recovered speech emotion recognition research project combining acoustic analysis and speech transcription.

## Recovery

The original `SentimentAnlaysis-main.zip` backup was restored into this repository. It includes microphone capture, Whisper transcription, pitch/volume/MFCC analysis, model construction and training experiments, a feature dataset, and eight TensorFlow SavedModel directories with their weights.

All recovered files are unchanged from the backup. `recovery-manifest.json` records their SHA-256 checksums; the backup README is preserved as `RECOVERED_README.md`. The existing root `model.py` and `Combined_Dataset.csv` are also preserved. Existing Git history is retained; the deleted repository's commit history was not present in the ZIP.

## Project layout

- `project/main.py`: microphone recording and transcription entry point.
- `project/dependencies/`: audio processing, configuration, and plotting.
- `project/dependencies/multimodal_sentiment/construct/`: original dataset and model experiments.
- `project/dependencies/multimodal_sentiment/models/`: eight recovered model versions.
- `project/dependencies/multimodal_sentiment/datasets/v2.1.csv`: recovered feature dataset.
- `project/requirements.txt`: original environment snapshot (UTF-16).
- `model.py`: original standalone RAVDESS, CREMA-D, TESS, and SAVEE preprocessing script.

## Running the recovered code

This is a faithful recovery of research code, not a newly validated release. The archived environment pins TensorFlow 2.10.1 and Keras 2.10.0. Recreate a compatible environment before installing `project/requirements.txt`; the Python 3.14 environment used during recovery cannot run those old pins.

The archived requirements are incomplete: source imports also include Whisper (`openai-whisper`), PyTorch, WebRTC VAD, noisereduce, NumPy, and plotting libraries. Review imports and install dependencies appropriate to your environment. Whisper may download its model on first launch.

Run the microphone entry point from the project directory because configuration paths are relative:

```sh
cd project
python main.py
```

It begins microphone capture immediately. Ctrl+C stops recording and displays analysis. The original code saves WAV-formatted audio under the name `output.mp3`.

Training/preprocessing experiments contain machine-specific dataset paths that must be updated for your local data. `project/server.py` is an empty placeholder. The feature CSV is not a replacement for source audio recordings.

## Validation

Recovery checks verified every archived file byte-for-byte and parsed all 25 Python files for syntax. Model loading, dependency installation, microphone capture, and training were not executed. No accuracy or improvement claims from the accompanying research paper have been independently verified.

