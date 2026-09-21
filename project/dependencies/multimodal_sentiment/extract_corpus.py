"""Extract MFCC features for the RAVDESS/TESS/CREMA-D/SAVEE files indexed by Combined_Dataset.csv.

Columns 0:40 are the same 40 MFCC means the runtime uses, so a model trained on
that slice alone loads through the stock predictor. The remaining columns add
spread and movement over time, which mean pooling discards.
"""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
import librosa
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from speech_emotion.models import SAMPLE_RATE, features as runtime_features

LABELS = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')
BLOCKS = ('mfcc_mean', 'mfcc_std', 'delta_mean', 'delta2_mean')


def speaker_of(path):
    """Group key identifying the person speaking, so splits can keep them apart."""
    corpus = path.parent.name
    if corpus.startswith('Actor_'):          # RAVDESS, one directory per actor
        return 'ravdess', corpus
    stem = path.stem
    if corpus == 'AudioWAV':                 # CREMA-D, 1001_DFA_ANG_XX.wav
        return 'crema', stem.split('_')[0]
    if corpus == 'ALL':                      # SAVEE, DC_a01.wav
        return 'savee', stem.split('_')[0]
    # TESS, in per-speaker folders like OAF_angry. The folder is authoritative because
    # one file in the corpus is misnamed OA_bite_neutral.
    return 'tess', corpus.split('_')[0]


def extract(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(y) < 2048 or not np.isfinite(y).all():
        return None
    mfcc = librosa.feature.mfcc(y=librosa.util.normalize(y), sr=SAMPLE_RATE, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    vector = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), delta.mean(axis=1),
                             librosa.feature.delta(mfcc, order=2).mean(axis=1)]).astype(np.float32)
    return vector if np.isfinite(vector).all() else None


def check_matches_runtime(paths):
    """The first 40 columns must equal what the live predictor computes, or a model
    trained on them would see different inputs at inference time."""
    for path in paths:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        if not np.allclose(extract(path)[:40], runtime_features(y, SAMPLE_RATE), atol=1e-4):
            raise SystemExit(f'Feature mismatch against speech_emotion.models.features for {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=ROOT/'Combined_Dataset.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'work/corpus-features.npz')
    parser.add_argument('--jobs', type=int, default=-1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new path.')

    rows = [r for r in csv.DictReader(args.index.open(encoding='utf-8')) if r['Emotions'] in LABELS]
    usable = [(Path(r['Path']), r['Emotions']) for r in rows if Path(r['Path']).is_file()]
    print(f'{len(rows)} rows carry a wanted label; {len(usable)} of those exist on disk.')

    check_matches_runtime([path for path, _ in usable[:5]])
    print('First 40 columns match speech_emotion.models.features.')

    vectors = Parallel(n_jobs=args.jobs, verbose=5)(delayed(extract)(path) for path, _ in usable)
    kept = [(vector, label, path) for vector, (path, label) in zip(vectors, usable) if vector is not None]
    print(f'{len(usable) - len(kept)} files were unreadable or too short.')

    x = np.stack([vector for vector, _, _ in kept])
    y = np.array([LABELS.index(label) for _, label, _ in kept], dtype=np.int64)
    corpora, speakers = zip(*(speaker_of(path) for _, _, path in kept))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x=x, y=y, labels=np.array(LABELS), blocks=np.array(BLOCKS),
                        corpus=np.array(corpora), speaker=np.array([f'{c}:{s}' for c, s in zip(corpora, speakers)]))
    print(f'Wrote {x.shape} to {args.output}')
    for corpus in sorted(set(corpora)):
        rows_here = [c for c in corpora if c == corpus]
        voices = {s for c, s in zip(corpora, speakers) if c == corpus}
        print(f'  {corpus:<8} {len(rows_here):>6} clips  {len(voices):>3} speakers')


if __name__ == '__main__':
    main()
