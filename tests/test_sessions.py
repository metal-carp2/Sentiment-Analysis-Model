import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from speech_emotion.sessions import analyze,render
from speech_emotion.cli import main,safe_name,record_audio

class Fake:
    labels=['happy','sad'];metadata={}
    def predict(self,audio,rate):return {'scores':{'happy':.7,'sad':.3}}

class SessionTests(unittest.TestCase):
    def test_duration_and_silence(self):
        y=np.concatenate([np.ones(48000)*.1,np.zeros(48000),np.ones(16000)*.1])
        report=analyze(y,16000,Fake())
        self.assertEqual(report['estimated_seconds'],4)
        self.assertEqual(report['quiet_seconds'],3)
        self.assertEqual(report['segments'][-1]['end'],7)
        self.assertIn('100.0%',render(report))
    def test_quiet_report(self):
        self.assertIn('No audio windows',render(analyze(np.zeros(16000),16000,Fake())))
    def test_invalid_scores(self):
        predictor=Fake();predictor.predict=lambda a,s:{'scores':{'happy':float('nan'),'sad':0}}
        with self.assertRaises(ValueError):analyze(np.ones(16000),16000,predictor)
    def test_duration_limit_and_names(self):
        with self.assertRaises(ValueError):analyze(np.ones(61*16000),16000,Fake())
        for name in ('../escape','CON','a/b',''):
            with self.assertRaises(ValueError):safe_name(name)
    def test_save_list_report_and_collision(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            args=['record','--seconds','1','--name','demo','--sessions-dir',temp]
            with patch('speech_emotion.sessions.load_backend',return_value=Fake()),patch('speech_emotion.cli.record_audio',return_value=(np.ones((16000,1))*.1,16000,[])):
                main(args)
                with self.assertRaises(SystemExit):main(args)
            report=json.loads((Path(temp)/'demo/report.json').read_text())
            self.assertEqual(report['duration_seconds'],1)
            self.assertTrue((Path(temp)/'demo/audio.wav').exists())
            main(['list','--sessions-dir',temp]);main(['report','demo','--sessions-dir',temp])
    def test_interrupt_preserves_partial_audio(self):
        class Stream:
            calls=0
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,n):
                self.calls+=1
                if self.calls>2:raise KeyboardInterrupt
                return np.ones((n,1),dtype=np.float32),False
        with patch('sounddevice.InputStream',return_value=Stream()),contextlib.redirect_stdout(io.StringIO()):
            audio,rate,warnings=record_audio(60,None)
        self.assertEqual(len(audio),3200)

if __name__=='__main__':unittest.main()
