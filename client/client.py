import wave
import numpy as np
import requests
from denoise_audio import denoise_audio
import librosa
import soundfile as sf
import time
import io

# endpoint = "http://34.141.193.219:8080"
endpoint = "http://127.0.0.1:8080"
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


def get_ping_duration():
    url = f'{endpoint}/ping'
    start_time = time.time()  # Record the start time
    response = requests.get(url)
    end_time = time.time()  # Record the end time
    
    duration = end_time - start_time  # Calculate the duration
    
    if response.status_code == 200:
        return duration
    else:
        return f"Error: {response.status_code}"

def wav_to_np_array():
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
    startTime= time.time()
    audio_bytes = numpy_array.astype(np.float64).tobytes()
    
    # Create the data payload
    payload = audio_bytes
    
    endTime = time.time()

    print("payload created", payload[:10])

    print(f'time:{endTime - startTime}')
    # Post the data to the specified endpoint
    ping_start = time.time()
    duration = get_ping_duration()
    print("ping duration", duration)
    ping_end = time.time()
    print(f'ping time:{ping_end - ping_start}')
    startNumpify = time.time()
    files = {'audio': ('audio_data', io.BytesIO(audio_bytes), 'application/octet-stream')}
    response = requests.post(endpoint, files=files)
    endNumpify = time.time()
    
    return response, startNumpify, endNumpify

# Load your WAV file and convert it


def infer():
    print("calling infer")
    numpy_array, fr = wav_to_np_array()
    
    response, startNumpify, endNumpify = post_numpy_array(numpy_array, fr)
    
    resp_json = response.json()
    print(resp_json)



infer()
