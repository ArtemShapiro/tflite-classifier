from urllib.request import urlopen
from PIL import Image

from tflite_runtime.interpreter import Interpreter

import json
import numpy as np


def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


# Model
interpreter = Interpreter('./mobnet/mobilenet.tflite')
interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']

# Labels
labels = load_labels('./mobnet/labels.txt')


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]


def process_url(url):
    # read from url
    image = Image.open(urlopen(url))
    # resize image
    image = image.resize((height, width), Image.ANTIALIAS)

    image = np.array(image) / 255

    # Run calssification
    return classify_image(interpreter, image)


def predict(request):
    url = request.get_json()["url"]
    label_id, prob = process_url(url)[0]

    return json.dumps({
        "labels": labels[label_id],
        "probability": float(prob)
    })
