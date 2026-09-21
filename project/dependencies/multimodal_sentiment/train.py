"""Train the single model using the supplied v2.1 CSV. Run from the repository root."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project.dependencies.multimodal_sentiment.construct_model import build_model


def load_data(path):
    with open(path, encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['major_emotion'] != 'other']
    x = np.array([json.loads(base64.b64decode(r['mfcc'])) for r in rows], dtype=np.float32)
    labels = sorted(set(r['major_emotion'] for r in rows))
    y = np.array([labels.index(r['major_emotion']) for r in rows])
    groups = np.array([' '.join(r['text'].lower().split()) for r in rows])
    if x.shape != (len(rows),40) or not np.isfinite(x).all():
        raise ValueError('Expected 40 finite MFCC values per row.')
    return x,y,groups,labels


def split_data(x,y,groups):
    # Choose the first deterministic group split with every class represented.
    # No model scores are used to select the split.
    classes=set(y)
    for seed in range(42,142):
        trainval,test=next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=seed).split(x,y,groups))
        tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.1765,random_state=seed).split(x[trainval],y[trainval],groups[trainval]))
        train,val=trainval[tr],trainval[va]
        if all(set(y[idx])==classes for idx in (train,val,test)):
            return train,val,test,seed
    raise ValueError('Could not make group-disjoint splits covering all classes.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(__file__).parent/'datasets/v2.1.csv')
    parser.add_argument('--output',type=Path,required=True,help='New empty model directory; existing outputs are refused')
    args=parser.parse_args()
    if args.output.exists():parser.error('Output already exists; use a new directory.')
    tf.keras.utils.set_random_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    x,y,groups,labels=load_data(args.data)
    train,val,test,seed=split_data(x,y,groups)
    model=build_model(x[train].mean(axis=0),x[train].var(axis=0),len(labels))
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    counts=np.bincount(y[train],minlength=len(labels))
    weights={i:float(np.sqrt(len(train)/(len(labels)*count))) for i,count in enumerate(counts)}
    history=model.fit(x[train],y[train],validation_data=(x[val],y[val]),batch_size=32,epochs=120,
        class_weight=weights,verbose=2,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True)])
    predicted=np.argmax(model(x[test],training=False).numpy(),axis=1)
    report={'accuracy':accuracy_score(y[test],predicted),'balanced_accuracy':balanced_accuracy_score(y[test],predicted),
        'majority_baseline_accuracy':float(np.mean(y[test]==np.argmax(counts))),
        'classification_report':classification_report(y[test],predicted,labels=list(range(len(labels))),target_names=labels,output_dict=True,zero_division=0),
        'confusion_matrix':confusion_matrix(y[test],predicted,labels=list(range(len(labels)))).tolist(),
        'split_counts':{'train':len(train),'validation':len(val),'test':len(test)},'split_seed':seed,
        'split_method':'Disjoint normalized transcript groups. Speaker IDs are unavailable; this is NOT a speaker-independent evaluation.',
        'epochs':len(history.history['loss']), 'optimizer_steps':int(model.optimizer.iterations.numpy()),
        'data_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest()}
    model.save(args.output)
    metadata={'labels':labels,'sample_rate':16000,'n_mfcc':40,'aggregation':'mean_over_time',
        'audio_normalization':'peak','feature_scaling':'embedded training-only Normalization layer',
        'status':'experimental_trained','training_data':'supplied v2.1.csv; other excluded',
        'preprocessing_note':'Reconstructed from recovered extract_dataset2. Original feature-generation provenance is not fully documented.',
        'evaluation':report}
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
