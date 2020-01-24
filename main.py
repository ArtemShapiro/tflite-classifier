from PIL import Image
from urllib.request import urlopen

from tflite_runtime.interpreter import Interpreter

import json
import numpy as np


def load_labels(path):
    """" Reads labels from file """
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}


# Model initialization
interpreter = Interpreter('./mobnet/mobilenet.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

_, height, width, _ = input_details[0]['shape']

# Labels initialization
labels = load_labels('./mobnet/labels.txt')

# CONSTANTS initialization
INPUT_MEAN = 127.5
INPUT_STD = 127.5


def classify_image(image, top_k: int=1):
    """Returns a sorted array of classification results."""
    input_data = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = np.squeeze(interpreter.get_tensor(output_details[0]['index']))

    ordered = np.argpartition(-output_data, top_k)
    return [(i, output_data[i]) for i in ordered[:top_k]]


def preprocess_input(image):
    """
    Returns normalized image according to backbone input preprocess. Type of preprocess:
    1) Resize
    2) Normalizatino
    """
    image = image.resize((height, width), Image.ANTIALIAS)
    # return np.float32(image)/255
    return (np.float32(image) - INPUT_MEAN) / INPUT_STD


def process_url(url: str = ''):
    """ Functions for reading image from URL. Returns results of classification """
    # read from url
    image = Image.open(urlopen(url))
    # Normalizate input
    input_image = preprocess_input(image)
    # Run calssification
    return classify_image(input_image)


def predict(request):
    """ Endpoint functions. Classifies iamge from the link """
    url = request.get_json()["url"]
    label_id, prob = process_url(url)[0]

    return json.dumps({
        "labels": labels[label_id],
        "probability": float(prob)
    })
