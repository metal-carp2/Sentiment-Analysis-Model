"""Train on the four-corpus features with a speaker-disjoint split.

No speaker appears in more than one split, so the held-out score reflects unheard
voices. This is stricter than train.py, which can only group by transcript because
IEMOCAP speaker identities are missing, and the two numbers are not comparable.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project.dependencies.multimodal_sentiment.construct_model import build_model

COLUMNS = {'mean': 40, 'all': 160}


def split_speakers(corpus, speaker, seed):
    """Hold out whole speakers. Corpora with too few voices stay in training, since
    holding one out would put a single person's recordings across an entire split."""
    rng = np.random.default_rng(seed)
    train, validation, test = [], [], []
    for name in sorted(set(corpus)):
        voices = sorted({s for s, c in zip(speaker, corpus) if c == name})
        if len(voices) < 4:
            train += voices
            continue
        order = [voices[i] for i in rng.permutation(len(voices))]
        count = max(1, round(len(order)*.15))
        test += order[:count]
        validation += order[count:2*count]
        train += order[2*count:]
    return set(train), set(validation), set(test)


def evaluate(model, x, y, labels):
    predicted = np.argmax(model(x, training=False).numpy(), axis=1)
    counts = np.bincount(y, minlength=len(labels))
    return {'accuracy': accuracy_score(y, predicted),
            'balanced_accuracy': balanced_accuracy_score(y, predicted),
            'majority_baseline_accuracy': float(counts.max()/counts.sum()),
            'classification_report': classification_report(y, predicted, labels=list(range(len(labels))),
                                                           target_names=labels, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y, predicted, labels=list(range(len(labels)))).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'work/corpus-features.npz')
    parser.add_argument('--output', type=Path, required=True, help='New model directory; existing outputs are refused')
    parser.add_argument('--features', choices=COLUMNS, default='mean',
                        help='mean: the 40 MFCC means the runtime already computes. all: adds spread and movement.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; use a new directory.')

    data = np.load(args.data, allow_pickle=False)
    labels = [str(label) for label in data['labels']]
    x = data['x'][:, :COLUMNS[args.features]]
    y, corpus, speaker = data['y'], data['corpus'], data['speaker']

    tf.keras.utils.set_random_seed(args.seed)
    train_voices, validation_voices, test_voices = split_speakers(corpus, speaker, args.seed)
    masks = {name: np.array([s in voices for s in speaker])
             for name, voices in (('train', train_voices), ('validation', validation_voices), ('test', test_voices))}
    for name, mask in masks.items():
        print(f'{name:<11} {mask.sum():>6} clips  {len({s for s, m in zip(speaker, mask) if m}):>3} speakers')
    overlap = (train_voices & test_voices) | (train_voices & validation_voices) | (validation_voices & test_voices)
    if overlap:
        raise SystemExit(f'Speakers appear in more than one split: {sorted(overlap)}')

    train, validation, test = (masks[name] for name in ('train', 'validation', 'test'))
    counts = np.bincount(y[train], minlength=len(labels))
    model = build_model(x[train].mean(axis=0), x[train].var(axis=0), len(labels))
    model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x[train], y[train], validation_data=(x[validation], y[validation]), batch_size=32, epochs=120,
                        class_weight={i: float(np.sqrt(train.sum()/(len(labels)*max(count, 1)))) for i, count in enumerate(counts)},
                        verbose=2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)])

    report = evaluate(model, x[test], y[test], labels)
    report.update(features=args.features, feature_columns=COLUMNS[args.features], seed=args.seed,
                  epochs=len(history.history['loss']), optimizer_steps=int(model.optimizer.iterations.numpy()),
                  split_method='Speaker-disjoint. Corpora with fewer than four speakers stay in training.',
                  split_counts={name: int(mask.sum()) for name, mask in masks.items()},
                  test_composition={name: int(((corpus == name) & test).sum()) for name in sorted(set(corpus.tolist()))})

    model.save(args.output)
    metadata = {'labels': labels, 'sample_rate': 16000, 'n_mfcc': 40,
                'aggregation': 'mean_over_time' if args.features == 'mean' else f'mfcc_stats_{COLUMNS[args.features]}',
                'audio_normalization': 'peak', 'feature_scaling': 'embedded training-only Normalization layer',
                'status': 'experimental_trained',
                'training_data': 'RAVDESS, TESS, CREMA-D and SAVEE via Combined_Dataset.csv; calm and unlabelled rows excluded',
                'evaluation': report}
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')

    print(f"\naccuracy           {report['accuracy']:.1%}")
    print(f"majority baseline  {report['majority_baseline_accuracy']:.1%}")
    print(f"balanced accuracy  {report['balanced_accuracy']:.1%}")
    print(f"macro F1           {report['classification_report']['macro avg']['f1-score']:.1%}")
    print('\nper-class recall')
    for label in labels:
        print(f"  {label:<10} {report['classification_report'][label]['recall']:6.1%}"
              f"  (support {int(report['classification_report'][label]['support'])})")


if __name__ == '__main__':
    main()
