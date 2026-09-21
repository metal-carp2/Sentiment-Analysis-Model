"""Live emotion estimates from a microphone, or repeatable WAV replay."""
import argparse
from collections import deque
import json
import threading
import time
from pathlib import Path
import numpy as np
from emotion_model import Predictor, SAMPLE_RATE


def estimate(predictor, audio, threshold):
    rms=float(np.sqrt(np.mean(np.asarray(audio,dtype=np.float32)**2)))
    if rms < threshold:
        return {'status':'quiet','rms':rms}
    return {'status':'estimate','rms':rms,**predictor.predict(audio)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wav',type=Path,help='Replay a WAV instead of opening the microphone')
    parser.add_argument('--window',type=float,default=3.0,help='Audio window in seconds (1-10)')
    parser.add_argument('--interval',type=float,default=1.0,help='Seconds between updates (0.25-10)')
    parser.add_argument('--seconds',type=float,help='Stop microphone capture after this duration')
    parser.add_argument('--device',type=int,help='Input device index')
    parser.add_argument('--list-devices',action='store_true')
    parser.add_argument('--silence-threshold',type=float,default=.001,help='Minimum RMS for inference')
    args=parser.parse_args()
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices());return
    if not 1<=args.window<=10 or not .25<=args.interval<=10:
        parser.error('Use a window of 1-10 seconds and interval of 0.25-10 seconds.')
    if not 0<=args.silence_threshold<=1 or (args.seconds is not None and args.seconds<args.window):
        parser.error('Threshold must be 0-1; seconds must be at least window length.')
    predictor=Predictor()
    size=int(args.window*SAMPLE_RATE)
    # Warm up feature extraction and inference before opening the microphone.
    predictor.predict(np.sin(np.arange(size,dtype=np.float32)*.08)*.01)
    print('Experimental emotion estimates; scores are not calibrated confidence. Ctrl+C stops.',flush=True)
    if args.wav:
        import librosa
        y,_=librosa.load(args.wav,sr=SAMPLE_RATE,mono=True)
        if len(y)<2048:parser.error('WAV is too short.')
        step=int(args.interval*SAMPLE_RATE)
        endpoints=list(range(min(size,len(y)),len(y)+1,step))
        if endpoints[-1]!=len(y):endpoints.append(len(y))
        for end in endpoints:
            result=estimate(predictor,y[max(0,end-size):end],args.silence_threshold)
            print(json.dumps({'seconds':round(end/SAMPLE_RATE,2),**result}),flush=True)
        return
    import sounddevice as sd
    buffer=deque(maxlen=int(np.ceil(args.window/.1)))
    lock=threading.Lock()
    stream_errors=deque(maxlen=10)
    def callback(indata,frames,time_info,status):
        with lock:
            if status:stream_errors.append(str(status))
            buffer.append(indata[:,0].copy())
    started=time.monotonic()
    next_update=started+args.window
    print('Microphone listening. Audio stays in a bounded memory buffer; no file is saved.',flush=True)
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE,channels=1,dtype='float32',blocksize=1600,device=args.device,callback=callback):
            while True:
                now=time.monotonic()
                if now>=next_update:
                    with lock:
                        audio=np.concatenate(list(buffer)) if buffer else np.array([])
                        errors=list(stream_errors);stream_errors.clear()
                    if errors: print(json.dumps({'status':'audio_warning','messages':errors}),flush=True)
                    if len(audio)>=2048:
                        print(json.dumps({'seconds':round(now-started,1),**estimate(predictor,audio[-size:],args.silence_threshold)}),flush=True)
                    next_update=time.monotonic()+args.interval
                if args.seconds is not None and time.monotonic()-started>=args.seconds:break
                time.sleep(.05)
    except KeyboardInterrupt:
        print('\nStopped.')
    except sd.PortAudioError as exc:
        parser.exit(1,f'Microphone error: {exc}\nUse --list-devices and --device INDEX to choose an input.\n')


if __name__=='__main__':main()
