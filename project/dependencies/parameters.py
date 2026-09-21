import json
import pyaudio

with open("./dependencies/params.json", 'r') as file:
    __params = json.load(file)

CHUNK_DURATION = __params["CHUNK_DURATION"]
SAMPLE_RATE = __params["SAMPLE_RATE"]
CHANNELS = __params["CHANNELS"]
REF = __params["REF"]
INACTIVITY_TIMEOUT = __params["INACTIVITY_TIMEOUT"]
FMIN = __params["FMIN"]
FMAX = __params["FMAX"]
N_MFCC = __params["N_MFCC"]
FRAME_LENGTH = int(SAMPLE_RATE * CHUNK_DURATION)
FORMAT = pyaudio.paInt16

