# Speech Emotion Recognition

Record a short audio clip or test a supplied WAV file. The repository contains one saved model, one model constructor, and one dataset compiler.

## Quick start (Windows PowerShell, Python 3.11)

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe test_audio.py --wav samples/03-01-01-01-01-01-01.wav --check-model
```

If using `uv`, create the environment with `uv venv --python 3.11 .venv` instead. The local development checkout already has a tested environment installed.

The command prints duration, audio level, MFCC dimensions, and whether the saved model loads and produces finite output. The first run can take longer while libraries initialize. TensorFlow may print deprecation notices when loading the SavedModel.

**This is an input/model-loading test, not an emotion prediction.** Inspection found zero optimizer steps and unchanged batch-normalization statistics in the retained checkpoint, strongly suggesting it is an initialized, untrained model. The training scaler and verified output-label mapping are also absent. Train and save these artifacts together, or supply a verified trained checkpoint, before interpreting model outputs as emotions.

## Record your microphone

```powershell
.\.venv\Scripts\python.exe test_audio.py --record 5
```

This records five seconds, saves a proper WAV to `recordings/test.wav`, and analyzes it. It does not transcribe or predict emotions. Recording begins only when you run this command. Choose a different filename for subsequent recordings; existing recordings are never overwritten:

```powershell
.\.venv\Scripts\python.exe test_audio.py --record 5 --output recordings/second.wav
.\.venv\Scripts\python.exe test_audio.py --wav recordings/second.wav --check-model
```

## Sample speech

Three real RAVDESS clips are included under `samples/`:

| Filename | Intended acted emotion |
| --- | --- |
| `03-01-01-01-01-01-01.wav` | Neutral |
| `03-01-03-01-01-01-01.wav` | Happy |
| `03-01-05-01-01-01-01.wav` | Angry |

See [sample attribution and license](samples/README.md). These are noncommercial, CC BY-NC-SA 4.0 audio samples; the code license does not replace their license. The clips test the input pipeline, not classification accuracy.

## Code layout

- `test_audio.py`: independent WAV analysis and microphone test CLI; no Whisper downloads.
- `project/dependencies/multimodal_sentiment/model/`: the single TensorFlow SavedModel. Its metadata and variable files form one model and must stay together.
- `project/dependencies/multimodal_sentiment/construct_model.py`: the single CNN/LSTM model builder. Importing it creates no model; running it prints a new model's summary without overwriting the saved model.
- `project/dependencies/multimodal_sentiment/compile_dataset.py`: the single IEMOCAP dataset compiler.
- `project/dependencies/multimodal_sentiment/train.py`: research training script, currently requiring further setup described below.
- `project/main.py`: original continuous recording, Whisper transcription, and plotting workflow.
- `project/dependencies/`: original audio analysis and text feature utilities.
- `model.py`: standalone dataset feature extraction for RAVDESS, CREMA-D, TESS, and SAVEE; not another saved classifier.

## Training and continuous transcription limitations

The root requirements support the new test command. The original research scripts need additional dependencies beyond these (including Whisper, PyTorch, WebRTC VAD, PyAudio, noisereduce, matplotlib, pandas, datasets, sentence-transformers, and NLTK). `project/requirements.txt` is an incomplete older environment snapshot, not the installation path for the tested CLI.

The original training script expects `datasets/v3.0.csv`, while the recovered data is `v2.1.csv`; they must not be treated as interchangeable. It also needs preprocessing/label persistence and saving of trained weights before it can produce a reusable classifier. Do not run a long training job until those gaps are addressed. The continuous transcription program expects to run from the `project` directory and still does not call the emotion classifier.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Validated with Python 3.11, TensorFlow 2.15.1, and librosa 0.10.2.post1 on Windows:

- All three real speech clips produced finite 40-coefficient MFCC features.
- The retained saved model loaded and produced a finite output of shape `(1, 8)` for a sample.
- Five automated tests passed, covering samples, stereo resampling, short/empty audio rejection, invalid recording duration, and protection against overwriting recordings.
- Microphone devices were enumerated; live microphone recording was not performed during validation.

No emotion accuracy claims have been verified.
