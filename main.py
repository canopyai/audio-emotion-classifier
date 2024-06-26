from flask import Flask, request, jsonify
from transformers import pipeline
import time
import numpy as np
from flask_cors import CORS
import torch
from denoiser import pretrained
from denoiser.dsp import convert_audio


app = Flask(__name__)
CORS(app)


# Load the audio classification pipeline
classifier = pipeline("audio-classification", model="amuvarma/audio-emotion-classifier-1-4", device=0 if torch.cuda.is_available() else -1)
denoiser_model = pretrained.dns64().cuda()
# Access the model object from the pipeline
model = classifier.model
is_cuda_available = torch.cuda.is_available()
device = 'cuda' if is_cuda_available else 'cpu'
model = model.to(device)


@app.route('/', methods=['POST'])
def classify_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "Missing audio data"}), 400

    # Measure time to read the audio file from request
    initial_time = time.time()
    audio_file = request.files['audio']
    audio_bytes = audio_file.read()


    # Convert bytes to numpy array
    audio_samples = np.frombuffer(audio_bytes, dtype=np.float64)
    wav_tensor = torch.tensor(audio_samples).float().cuda()

    with torch.no_grad():
        denoised = denoiser_model(wav_tensor.unsqueeze(0))[0]

    denoised_numpy = denoised.squeeze().cpu().numpy()


    # Process the numpy array as needed (your classification logic here)
    result = classifier(denoised_numpy, top_k=10)
    final_time = time.time()

    # You can include additional logic here if needed

    return jsonify({
        'result': result, 
        'inference_time': final_time - initial_time,
        "is_cuda_available": is_cuda_available
    })
@app.route('/ping')
def pong():
    return 'pong'

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
