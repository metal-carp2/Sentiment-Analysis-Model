"""Command line entry point. Microphone access occurs only in record."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import uuid


def record_audio(seconds, device):
    import numpy as np
    import sounddevice as sd
    chunks=[];warnings=[];remaining=int(seconds*16000)
    print('Recording. Speak now; Ctrl+C ends the session early.',flush=True)
    try:
        with sd.InputStream(samplerate=16000,channels=1,dtype='float32',device=device) as stream:
            while remaining:
                count=min(1600,remaining)
                chunk,overflow=stream.read(count)
                chunks.append(chunk.copy());remaining-=len(chunk)
                if overflow and not warnings:warnings.append('Input overflow: some audio may be missing.')
    except KeyboardInterrupt:
        print('\nRecording stopped.')
    if not chunks:raise ValueError('No audio captured.')
    return np.concatenate(chunks),16000,warnings


def safe_name(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',name):
        raise ValueError('Session name must be 1-64 letters, numbers, hyphens or underscores.')
    if name.upper() in {'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(10)],*[f'LPT{i}' for i in range(10)]}:
        raise ValueError('Reserved session name.')
    return name


def main(argv=None):
    parser=argparse.ArgumentParser(prog='emotion-session',description='Record up to a minute and view an experimental emotion timeline.')
    commands=parser.add_subparsers(dest='command')
    for action in ('record','analyze'):
        p=commands.add_parser(action)
        if action=='record':
            p.add_argument('--seconds',type=float,default=60)
            p.add_argument('--device',type=int)
        else:p.add_argument('wav',type=Path)
        p.add_argument('--name')
        p.add_argument('--sessions-dir',type=Path,default=Path('emotion-sessions'))
        p.add_argument('--model',type=Path,help='Replacement model bundle directory')
        p.add_argument('--backend',help='Explicit Python module:factory for an installed custom model backend')
        p.add_argument('--window',type=float,default=3)
        p.add_argument('--silence-threshold',type=float,default=.001)
    commands.add_parser('devices')
    p=commands.add_parser('list');p.add_argument('--sessions-dir',type=Path,default=Path('emotion-sessions'))
    p=commands.add_parser('report');p.add_argument('name');p.add_argument('--sessions-dir',type=Path,default=Path('emotion-sessions'))
    p=commands.add_parser('menu');p.add_argument('--sessions-dir',type=Path,default=Path('emotion-sessions'))
    args=parser.parse_args(argv)
    if args.command is None:
        if not sys.stdin.isatty():parser.error('Choose a command.')
        args.command='menu';args.sessions_dir=Path('emotion-sessions')
    if args.command=='menu':
        from .menu import run
        return run(args.sessions_dir)
    session=None
    created=False
    try:
        if args.command=='devices':
            import sounddevice as sd
            print(sd.query_devices());return
        if args.command=='list':
            found=False
            for path in sorted(args.sessions_dir.glob('*/report.json')):
                print(path.parent.name);found=True
            if not found:print('No completed sessions yet.')
            return
        if args.command=='report':
            from .sessions import render
            data=json.loads((args.sessions_dir/safe_name(args.name)/'report.json').read_text(encoding='utf-8'))
            print(render(data));return
        if args.command=='record' and not 1<=args.seconds<=60:raise ValueError('Choose a duration between 1 and 60 seconds.')
        if not 1<=args.window<=10 or not 0<=args.silence_threshold<=1:raise ValueError('Window must be 1-10 seconds and threshold 0-1.')
        name=safe_name(args.name or datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
        session=args.sessions_dir/name
        if session.exists():raise ValueError('Session already exists. Choose another --name.')
        import soundfile as sf
        if args.command=='analyze':
            info=sf.info(args.wav)
            if not 0<info.duration<=60:raise ValueError('WAV must contain more than zero and at most 60 seconds of audio.')
        from .sessions import load_backend,analyze,render
        print('Loading model...',flush=True)
        predictor=load_backend(args.model,args.backend)
        session.mkdir(parents=True,exist_ok=False)
        created=True
        if args.command=='record':audio,rate,warnings=record_audio(args.seconds,args.device)
        else:audio,rate=sf.read(args.wav,dtype='float32',always_2d=True);warnings=[]
        sf.write(session/'audio.wav',audio,rate,subtype='PCM_16')
        print('Analyzing session...',flush=True)
        report=analyze(audio,rate,predictor,args.window,args.silence_threshold)
        report.update(name=name,created_at=datetime.now(timezone.utc).isoformat(),warnings=warnings,
                      backend=args.backend or 'speech_emotion.models:Predictor',model_path=str(args.model) if args.model else 'bundled')
        text=render(report)
        if warnings:text+='\n'+'\n'.join(warnings)
        (session/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        (session/'report.txt').write_text(text+'\n',encoding='utf-8')
        print('\n'+text+'\n\nSaved: '+str(session.resolve()))
    except Exception as exc:
        message=f'Error: {exc}'
        if created and (session/'audio.wav').exists():message+=f'\nCaptured audio was preserved at {session / "audio.wav"}; retry with analyze.'
        if created and not any(session.iterdir()):session.rmdir()
        parser.exit(1,message+'\n')
