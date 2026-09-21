from pydoc import text
import pyaudio
import webrtcvad
import time
import wave
import threading
import dependencies.audiodata as aud
import dependencies.parameters as param
from dependencies.livedebug import show_live_debug_display
from dependencies.recording import analyze_pitch_volume, process_audio_chunk, transcribeAudio
# Initialize microphone, Whisper model, and VAD

vad = webrtcvad.Vad(0)  # WebRTC VAD: 0 = least sensitive, 3 = most sensitive
audio = pyaudio.PyAudio()  # Initialize audio recording and state tracking
stamps = aud.AudioDataSet()  # Keep track of data
stream = audio.open(
    format=param.FORMAT, 
    channels=param.CHANNELS,
    rate=param.SAMPLE_RATE, 
    input=True,
    frames_per_buffer=param.FRAME_LENGTH
)

# Initialization Variables
buffer = []
frames = []
is_speaking = False
voiceStartTime = time.time()
last_speech_time = start_time = time.time()  # Track the last time speech was detected and track when speech started
initial = True  #  official start time
next_fire_time = 0

def transcription_ranges_iterable(segment, when_voice_started):
    stamps.add_range(
        start=stamps.closest_point(when_voice_started, 'floor')[0],
        end=stamps.closest_point(segment['end'] + when_voice_started, 'ceil')[0],
        dialogue=segment['text']
    )
    print()
    print(segment['text'], when_voice_started)

def processing_callback(rna):
    threading.Thread(
        target=transcribeAudio, 
        args=(rna, transcription_ranges_iterable, voiceStartTime)
    ).start()

def update_start_time(time):
    global next_fire_time
    global start_time
    start_time = time
    next_fire_time = time
    print("Recording...")

try:
    while True:
        if time.time() >= next_fire_time:
            
            audio_chunk = stream.read(param.FRAME_LENGTH)
            frames.append(audio_chunk)

            if initial:
                timestamp, pitch, volume, mfcc = analyze_pitch_volume(audio_chunk, start_time, update_start_time)
                initial = False
            else:
                timestamp, pitch, volume, mfcc = analyze_pitch_volume(audio_chunk, start_time)

            stamps.add_frame(timestamp, pitch, volume, mfcc)

            is_speech = vad.is_speech(audio_chunk, param.SAMPLE_RATE)

            if (volume > param.REF+0.01):   # volume > threshold

                if is_speech and not is_speaking:  # voice started
                    is_speaking = True
                    voiceStartTime = time.time() - start_time
    
                if is_speech:  # ongoing; during speech
                    last_speech_time = time.time()

            elif (not is_speech and is_speaking and time.time() - last_speech_time > param.INACTIVITY_TIMEOUT):

                is_speaking = False
                process_audio_chunk(buffer, processing_callback)
                buffer = []

            if is_speaking:

                buffer.append(audio_chunk)

            next_fire_time += param.CHUNK_DURATION  # Increment for the next chunk

except KeyboardInterrupt:
    print("Exiting...")
finally:
    print("Saving audio to output.wav...")
    # Combine the remaining buffered audio data into a single stream
    wf = wave.open("output.mp3", 'wb')
    wf.setnchannels(param.CHANNELS)
    wf.setsampwidth(audio.get_sample_size(param.FORMAT))
    wf.setframerate(param.SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    # Clean up resources after use
    stream.stop_stream()
    stream.close()
    audio.terminate()
    show_live_debug_display(stamps)