import soundfile as sf
import os
import sys

print(f"Python version: {sys.version}")

try:
    print(f"Soundfile version: {sf.__version__}")
    print(f"Soundfile file: {sf.__file__}")
    print(f"Libname: {sf._libname}")
except Exception as e:
    print(f"Error inspecting soundfile: {e}")

print("-" * 20)
print("Testing file read...")

# Create a dummy wav file if possible, or just fail gracefully
dummy_filename = "test_audio.wav"
import numpy as np
sr = 24000
audio = np.random.uniform(-0.5, 0.5, sr)
try:
    sf.write(dummy_filename, audio, sr)
    print(f"Created dummy file {dummy_filename}")
    data, samplerate = sf.read(dummy_filename)
    print(f"Successfully read {dummy_filename}, shape={data.shape}, sr={samplerate}")
    os.remove(dummy_filename)
except Exception as e:
    print(f"FAILED to write/read audio: {e}")
