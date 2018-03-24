import base64

from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import imutils

from utils import convert
from vision import debug

app = Flask(__name__)
app.config['DEBUG'] = True
CORS(app)

@app.route("/critique", methods=['POST'])
def critique():
    if request.method == 'POST':
        # Read image from request and convert it to an array
        encoded_string = request.form['image'].split(',')[-1]
        image = convert.base64_to_array(encoded_string)

        # Processing and analysis
        

        # Encode image
        response_string = "data:image/png;base64,{}".format(convert.array_to_base64(image))

        # Send response with image
        return jsonify({'data': response_string})

if __name__ == '__main__':
    app.run(threaded=False)