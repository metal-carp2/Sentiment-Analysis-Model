import unittest
from unittest.mock import Mock, patch
import numpy as np
from emotion_model import Predictor, features
from live_emotion import estimate
from project.dependencies.multimodal_sentiment.train import load_data,split_data
from test_audio import ROOT


class LiveTests(unittest.TestCase):
    def test_silence_skips_model(self):
        predictor=Mock()
        self.assertEqual(estimate(predictor,np.zeros(48000),.001)['status'],'quiet')
        predictor.predict.assert_not_called()

    def test_microphone_loop_with_simulated_input(self):
        import live_emotion
        fake=Mock()
        fake.predict.return_value={'emotion':'neutral','score':.5,'scores':{'neutral':.5}}
        class Input:
            def __init__(self, **kwargs): self.callback=kwargs['callback']
            def __enter__(self):
                for _ in range(10): self.callback(np.ones((1600,1),dtype=np.float32)*.01,1600,None,None)
                return self
            def __exit__(self,*args): pass
        with patch('sys.argv',['live_emotion.py','--window','1','--seconds','1']), patch('live_emotion.Predictor',return_value=fake), patch('sounddevice.InputStream',Input), patch('builtins.print'):
            live_emotion.main()
        self.assertGreaterEqual(fake.predict.call_count,2)

    def test_feature_shape_and_invalid_input(self):
        y=np.sin(np.arange(48000)*.1).astype(np.float32)
        self.assertEqual(features(y).shape,(40,))
        self.assertTrue(np.isfinite(features(y)).all())
        with self.assertRaises(ValueError):features(np.array([np.nan]*4000))

    def test_group_split_has_no_overlap(self):
        x,y,groups,labels=load_data(ROOT/'project/dependencies/multimodal_sentiment/datasets/v2.1.csv')
        train,val,test,_=split_data(x,y,groups)
        for a,b in [(train,val),(train,test),(val,test)]:
            self.assertFalse(set(groups[a]) & set(groups[b]))
        self.assertEqual(len(train)+len(val)+len(test),len(x))

    def test_trained_model_and_embedded_scaling(self):
        p=Predictor()
        self.assertGreater(p.metadata['evaluation']['optimizer_steps'],0)
        self.assertEqual(len(p.labels),8)
        self.assertIsNotNone(p.model.get_layer('training_normalization'))
        result=p.predict(np.sin(np.arange(48000)*.08).astype(np.float32)*.01)
        self.assertIn(result['emotion'],p.labels)
        self.assertAlmostEqual(sum(result['scores'].values()),1,places=5)


if __name__=='__main__':unittest.main()
