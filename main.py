from flask import Flask
from transformers import pipeline
import time
from flask import request
import numpy

app = Flask(__name__)


# Load the audio classification pipeline
classifier = pipeline("audio-classification", model="amuvarma/audio-emotion-classifier-1-0")

# Access the model object from the pipeline
model = classifier.model

@app.route('/', methods=['POST'])
def classify_audio():
    # takes in numpy array of an audio file
    request_data = request.get_json()
    
    audio_samples = request_data['audio_samples']
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
