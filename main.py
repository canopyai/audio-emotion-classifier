from flask import Flask, request
from transformers import pipeline
import time
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Load the audio classification pipeline
classifier = pipeline("audio-classification", model="amuvarma/audio-emotion-classifier-1-0")

# Access the model object from the pipeline
model = classifier.model

@app.route('/', methods=['POST'])
def classify_audio():
    # takes in numpy array of an audio file
    request_data = request.get_json()
    
    audio_samples = request_data['audio_samples']
    audio_samples = np.array(audio_samples, dtype=np.float64)
    startTime = time.time()
    result = classifier(audio_samples)
    endTime = time.time()
    inference_time = endTime - startTime

    return {'result': result, 'inference_time': inference_time}

@app.route('/ping')
def pong():
    return 'pong'

if __name__ == '__main__':
    app.run(debug=True, port=8080)
