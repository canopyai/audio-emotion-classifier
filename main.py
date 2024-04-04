from flask import Flask, request
from transformers import pipeline
import time
import numpy as np
from flask_cors import CORS
import torch

app = Flask(__name__)
CORS(app)


# Load the audio classification pipeline
classifier = pipeline("audio-classification", model="amuvarma/audio-emotion-classifier-1-0")


# Access the model object from the pipeline
model = classifier.model
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

@app.route('/', methods=['POST'])
def classify_audio():
    # takes in numpy array of an audio file
    time1 = time.time()
    request_data = request.get_json()
    time2 = time.time()
    audio_samples = request_data['audio_samples']
    time3 = time.time()
    audio_samples = np.array(audio_samples, dtype=np.float64)
    time4 = time.time()
    result = classifier(audio_samples)
    time5 = time.time()

    return {'result': result, 'time1': time1, 'time2': time2, 'time3': time3, 'time4': time4, 'time5': time5}

@app.route('/ping')
def pong():
    return 'pong'

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
