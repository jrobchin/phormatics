import sys
sys.path.append(".")
import base64

import settings

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import cv2
import imutils

from utils import convert
from vision import analyze

from helpers.estimator import TfPoseEstimator
from helpers.networks import get_graph_path, model_wh

# Setup Flask app
app = Flask(__name__)
app.config['DEBUG'] = True
CORS(app)

# Load model
pose_estimator = TfPoseEstimator(graph_path=settings.GRAPH_PATH, target_size=settings.PROCESSING_DIMS)

@app.route("/critique", methods=['POST'])
def critique():
    """
    Endpoint for analyzing the video feed.
    input:
    {
        "image": base64EncodedString,
        "workout": string,
        "rotate": int => (degrees),
        "side": char => ('L' or 'R'),
    }

    return:
    {
        "data": {
            "points": [],
            "critiques": [],
            "deviation": float,
            "state": int,
            "setcount": int,
            "repcount": int
        },
    }
    """
    print("HERE")

    if request.method == 'POST':
        # Read image from request
        encoded_string = request.form['image'].split(',')[-1]
        workout = request.form['workout']
        rotate = request.form.get('rotate')
        side = request.form.get('side')

        # Convert image string to image array
        image = convert.base64_to_array(encoded_string)

        # ---- Processing and analysis ----
        # Extract body parts
        try:
            body_parts = analyze.extract_body_parts(pose_estimator, image, rotate)
        except Exception as e:
            pass

        # Analyze workout from body parts
        deviation, critiques = analyze.analyze_workout(body_parts, workout, side)

        # Encode image
        response_string = "data:image/png;base64,{}".format(convert.array_to_base64(draw))

        # Send response with image
        return jsonify({'data': response_string})

@app.route("/login")
def test():
    return render_template('login.html')

if __name__ == '__main__':
    app.run(
        threaded=False,
        ssl_context=('cert.pem', 'key.pem')
    )