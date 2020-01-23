from tflite_runtime.interpreter import Interpreter

import json
import numpy as np


def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


# Model
interpreter = Interpreter('./mobnet/mobilenet_v1_1.0_224_quant.tflite')

# Labels
labels = load_labels('./mobnet/labels_mobilenet_quant_v1_224.txt')


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


def process_image(img):
    # Model
    interpreter.allocate_tensors()

    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    print(height, width)

    # image = Image.open(stream).convert('RGB').resize((width, height, Image.ANTIALIAS)
    # results = classify_image(interpreter, image)


def predict(event, context):
    body = json.loads(event.get('body'))

    response = {
        "statusCode": 200,
        "body": json.dumps(body),
    }

    return response

    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    """
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
    """
