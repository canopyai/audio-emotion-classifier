import wave
import numpy as np
import requests
from denoise_audio import denoise_audio
import librosa
import soundfile as sf
import time

# endpoint = "http://35.234.142.114:8080"
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
    url = "http://35.234.142.114:8080/ping"
    start_time = time.time()  # Record the start time
    response = requests.get(url)
    end_time = time.time()  # Record the end time
    
    duration = end_time - start_time  # Calculate the duration
    
    if response.status_code == 200:
        return f"Response Time: {duration} seconds"
    else:
        return f"Error: {response.status_code}"

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
    ping_start = time.time()
    get_ping_duration()
    ping_end = time.time()
    print(f'ping time:{ping_end - ping_start}')
    startNumpify = time.time()
    response = requests.post(endpoint, json=payload)
    endNumpify = time.time()
    
    return response, startNumpify, endNumpify

# Load your WAV file and convert it


def infer():
    
    numpy_array, fr = wav_to_np_array()
    
    response, startNumpify, endNumpify = post_numpy_array(numpy_array, fr)
    
    resp_json = response.json()
    time1 = resp_json['time1']
    time2 = resp_json['time2']
    time3 = resp_json['time3']
    time4 = resp_json['time4']
    time5 = resp_json['time5']
    print(f'times:{time2 - time1}, {time3 - time2}, {time4 - time3}, {time5 - time4}')


infer()
