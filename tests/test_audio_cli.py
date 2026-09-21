import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import test_audio


class AudioTests(unittest.TestCase):
    def test_real_samples(self):
        for path in (test_audio.ROOT/'samples').glob('*.wav'):
            with self.subTest(path=path.name):
                report, audio = test_audio.analyze(path)
                self.assertEqual(report['sample_rate'], 16000)
                self.assertEqual(report['mfcc_shape'][0], 40)
                self.assertTrue(report['features_finite'])
                self.assertGreater(report['rms'], 0)
                self.assertGreater(len(audio), 16000)

    def test_empty_and_short_audio_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'short.wav'
            for n in (0, 100):
                sf.write(path, np.zeros(n), 16000)
                with self.assertRaises(ValueError):
                    test_audio.analyze(path)

    def test_stereo_resampling(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'stereo.wav'
            t=np.arange(44100)/44100
            sf.write(path,np.column_stack([np.sin(2*np.pi*220*t)]*2),44100)
            report,audio=test_audio.analyze(path)
            self.assertEqual(len(audio),16000)
            self.assertEqual(report['original_channels'],2)

    def test_invalid_duration_does_not_record(self):
        with patch('sys.argv',['test_audio.py','--record','0']):
            with self.assertRaises(SystemExit) as error:
                test_audio.main()
            self.assertEqual(error.exception.code,1)

    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'recording.wav';path.write_bytes(b'keep')
            with patch('sys.argv',['test_audio.py','--record','1','--output',str(path)]):
                with self.assertRaises(SystemExit):test_audio.main()
            self.assertEqual(path.read_bytes(),b'keep')


if __name__=='__main__':
    unittest.main()
