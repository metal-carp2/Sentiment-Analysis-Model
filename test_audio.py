"""Test a WAV or record a short clip without loading Whisper or downloading models."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'project/dependencies/multimodal_sentiment/model'


def analyze(path):
    import librosa
    import numpy as np
    import soundfile as sf
    samples, rate = sf.read(path, dtype='float32', always_2d=True)
    if len(samples) == 0 or not np.isfinite(samples).all():
        raise ValueError('Audio is empty or contains invalid samples.')
    if len(samples) / rate > 60:
        raise ValueError('Use a clip of at most 60 seconds.')
    y = librosa.resample(samples.mean(axis=1), orig_sr=rate, target_sr=16000)
    if len(y) < 2048:
        raise ValueError('Use at least 0.13 seconds of audio.')
    mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=40)
    return {
        'file': str(path), 'duration_seconds': round(len(y)/16000, 3),
        'sample_rate': 16000, 'original_channels': samples.shape[1],
        'rms': float(np.sqrt(np.mean(y**2))),
        'mfcc_shape': list(mfcc.shape), 'features_finite': bool(np.isfinite(mfcc).all()),
    }, y


def check_model(y):
    import librosa
    import numpy as np
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL)
    # Reproduce the retained compiler's mean-over-coefficients sequence.
    # Training-time scaler was not recovered: this is ONLY a forward-pass check.
    sequence = librosa.feature.mfcc(y=librosa.util.normalize(y), sr=16000, n_mfcc=40).mean(axis=0)
    output = model(sequence[np.newaxis, :, np.newaxis], training=False).numpy()
    if not np.isfinite(output).all():
        raise ValueError('Model produced invalid values.')
    return {
        'path': str(MODEL), 'input_shape': list(model.input_shape),
        'output_shape': list(output.shape),
        'forward_pass_finite': True,
        'optimizer_steps': int(model.optimizer.iterations.numpy()) if model.optimizer else None,
        'note': 'Load/forward-pass test only; no emotion prediction. Training scaler and verified label mapping are missing.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--wav', type=Path, help='Existing WAV file (up to 60 seconds)')
    source.add_argument('--record', type=float, metavar='SECONDS', help='Record 1-60 seconds from your microphone')
    parser.add_argument('--output', type=Path, default=ROOT/'recordings/test.wav')
    parser.add_argument('--check-model', action='store_true', help='Also load the saved model and check a forward pass')
    args = parser.parse_args()
    try:
        if args.record is not None:
            if not 1 <= args.record <= 60:
                raise ValueError('Recording duration must be between 1 and 60 seconds.')
            if args.output.exists():
                raise ValueError('Output already exists; choose a new --output path.')
            import sounddevice as sd
            import soundfile as sf
            print(f'Recording for {args.record:g} seconds. Speak now.', flush=True)
            audio = sd.rec(int(args.record*16000), samplerate=16000, channels=1, dtype='float32')
            sd.wait()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(args.output, audio, 16000, subtype='PCM_16', format='WAV')
            args.wav = args.output
        report, y = analyze(args.wav)
        if args.check_model:
            report['model'] = check_model(y)
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        parser.exit(1, f'Test failed: {exc}\n')


if __name__ == '__main__':
    main()
