import io
import base64

import cv2
import numpy as np
from PIL import Image
import scipy

def base64_to_array(encoded):
    return scipy.misc.imread(io.BytesIO(base64.b64decode(encoded)), mode='RGB')

def array_to_base64(array):
    retval, buffer = cv2.imencode('.png', array)
    encoded_response = base64.b64encode(buffer)
    return encoded_response.decode('utf-8')
    