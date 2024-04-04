import wave
import numpy as np
import requests
from denoise_audio import denoise_audio
import librosa
import soundfile as sf
import time

def resample_to_16k(wav_file_path, output_file_path):

    # Load the original audio file
    audio, sample_rate = librosa.load(wav_file_path, sr=None)

    # Resample the audio to 16kHz
    audio_resampled = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

    # Save the resampled audio to a new WAV file
    sf.write(output_file_path, audio_resampled, 16000)

# For PyTorch
import torch
print(f"PyTorch version: {torch.__version__}")

# For the Hugging Face Transformers library
import transformers
print(f"Transformers version: {transformers.__version__}")


def wav_to_np_array():
    original_wav_file_path = "record.wav"   
    wav_path = "resampled.wav"
    # resample_to_16k(original_wav_file_path, wav_path)
    with wave.open(wav_path, 'rb') as wav_file:
        nchannels, sampwidth, framerate, nframes, comptype, compname = wav_file.getparams()
        frames = wav_file.readframes(nframes)
        
        if sampwidth == 1:
            dtype = np.uint8
        elif sampwidth == 2:
            dtype = np.int16
        else:
            raise ValueError("Only supports 8 and 16 bit audio formats.")
        
        audio_np = np.frombuffer(frames, dtype=dtype)
        
        if nchannels == 2:
            audio_np = np.reshape(audio_np, (nframes, nchannels))
        
    return audio_np, framerate

def post_numpy_array(numpy_array, framerate):
    # Convert the numpy array to a list for JSON serialization
    numpy_array = denoise_audio(numpy_array)

    audio_list = numpy_array.tolist()
    
    # Create the data payload
    payload = {
        "audio_samples": audio_list,
        "framerate": framerate
    }
    
    # Post the data to the specified endpoint
    response = requests.post("http://34.41.127.218", json=payload)
    
    return response

# Load your WAV file and convert it

startNumpify = time.time()
numpy_array, fr = wav_to_np_array()
endNumpify = time.time()
print(f"Time to convert to numpy array: {endNumpify - startNumpify}")

# Post the numpy array to the server
response = post_numpy_array(numpy_array, fr)

print(response.text)
