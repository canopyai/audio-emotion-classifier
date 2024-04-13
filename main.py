from flask import Flask, request, jsonify
from transformers import pipeline
import time
import numpy as np
from flask_cors import CORS
import torch


app = Flask(__name__)
CORS(app)


# Load the audio classification pipeline
classifier = pipeline("audio-classification", model="amuvarma/audio-emotion-classifier-1-4")


# Access the model object from the pipeline
model = classifier.model
is_cuda_available = torch.cuda.is_available()
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')


@app.route('/', methods=['POST'])
def classify_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "Missing audio data"}), 400

    # Measure time to read the audio file from request
    time1 = time.time()
    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    time2 = time.time()

    # Convert bytes to numpy array
    audio_samples = np.frombuffer(audio_bytes, dtype=np.float64)
    time3 = time.time()

    # Process the numpy array as needed (your classification logic here)
    result = classifier(audio_samples)
    time4 = time.time()

    # You can include additional logic here if needed

    return jsonify({
        'result': result, 
        'inference_time': time4 - time1,
        "is_cuda_available": is_cuda_available
    })
@app.route('/ping')
def pong():
    return 'pong'

if __name__ == '__main__':
    app.run(debug=True, port=8083, host='0.0.0.0')
