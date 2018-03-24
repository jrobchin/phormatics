import io
import base64

import cv2
import numpy as np
from PIL import Image

def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

def base64_to_array(encoded):
    imgdata = base64.b64decode(encoded)
    return toRGB(Image.open(io.BytesIO(imgdata)))

def array_to_base64(array):
    retval, buffer = cv2.imencode('.png', array)
    encoded_response = base64.b64encode(buffer)
    return encoded_response.decode('utf-8')
    