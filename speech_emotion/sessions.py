"""Backend-independent session analysis and human-readable reporting."""
import importlib
import json
from collections import defaultdict
import numpy as np


def load_backend(model=None, backend=None):
    if backend:
        module, separator, factory = backend.partition(':')
        if not separator:
            raise ValueError('Backend must be module:factory.')
        predictor = getattr(importlib.import_module(module), factory)(model)
    else:
        from .models import Predictor, MODEL_DIR
        predictor = Predictor(model or MODEL_DIR)
    if not callable(getattr(predictor, 'predict', None)):
        raise ValueError('Backend must expose predict(audio, sample_rate).')
    labels = getattr(predictor, 'labels', None)
    if not labels or not all(isinstance(label,str) and label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError('Backend must provide unique nonempty string labels.')
    return predictor


def analyze(audio, rate, predictor, window=3., threshold=.001):
    y=np.asarray(audio,dtype=np.float32)
    if y.ndim==2:y=y.mean(axis=1)
    if y.ndim!=1 or not len(y) or not np.isfinite(y).all():raise ValueError('Invalid or empty audio.')
    if rate<=0 or not 0<len(y)/rate<=60.001:raise ValueError('Sessions must be at most 60 seconds.')
    if not 1<=window<=10 or not 0<=threshold<=1:raise ValueError('Invalid window or silence threshold.')
    width=int(window*rate)
    segments=[];durations=defaultdict(float);quiet=0.;short=0.
    for start in range(0,len(y),width):
        clip=y[start:start+width];duration=len(clip)/rate
        segment={'start':start/rate,'end':(start+len(clip))/rate,'rms':float(np.sqrt(np.mean(clip**2)))}
        if duration<.13:
            segment['status']='too_short';short+=duration
        elif segment['rms']<threshold:
            segment['status']='quiet';quiet+=duration
        else:
            result=predictor.predict(clip,rate)
            scores=result['scores']
            if set(scores)!=set(predictor.labels):raise ValueError('Prediction labels disagree with backend labels.')
            values=np.array([scores[label] for label in predictor.labels],dtype=float)
            if not np.isfinite(values).all() or (values<0).any() or (values>1).any() or not np.isclose(values.sum(),1,atol=1e-4):
                raise ValueError('Backend must return finite probability scores that sum to one.')
            label=predictor.labels[int(values.argmax())]
            segment.update(status='estimate',emotion=label,score=float(values.max()),scores=dict(zip(predictor.labels,map(float,values))))
            durations[label]+=duration
        segments.append(segment)
    return {'duration_seconds':len(y)/rate,'window_seconds':window,'quiet_seconds':quiet,'too_short_seconds':short,
        'estimated_seconds':sum(durations.values()),'emotion_seconds':dict(durations),'segments':segments,
        'model':getattr(predictor,'metadata',{}),
        'interpretation':'Experimental model estimates, not measured feelings. Shares describe predicted audio windows; scores are not calibrated confidence.'}


def render(report):
    lines=[f"Session: {report.get('name','audio')}",f"Duration: {report['duration_seconds']:.1f}s | Analyzed: {report['estimated_seconds']:.1f}s | Quiet: {report['quiet_seconds']:.1f}s | Too short: {report['too_short_seconds']:.1f}s",'', 'Estimated emotion distribution (analyzed time only):']
    total=report['estimated_seconds']
    if not total:lines.append('  No audio windows exceeded the analysis threshold.')
    for label,duration in sorted(report['emotion_seconds'].items(),key=lambda item:-item[1]):
        lines.append(f'  {label:<14} {duration:5.1f}s  {duration/total:6.1%}')
    lines+=['','Timeline:']
    for item in report['segments']:
        text=item.get('emotion',item['status'])
        score=f" (model score {item['score']:.2f})" if 'score' in item else ''
        lines.append(f"  {item['start']:5.1f}-{item['end']:5.1f}s  {text}{score}")
    lines+=['',report['interpretation']]
    evaluation=report.get('model',{}).get('evaluation',{})
    if 'accuracy' in evaluation:lines.append(f"Model held-out accuracy: {evaluation['accuracy']:.1%}. This is not a live-session accuracy measurement.")
    return '\n'.join(lines)
