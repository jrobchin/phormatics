import sys
sys.path.append(".")
import os
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
app.config['STATIC_FOLDER'] = os.path.join(settings.BASE_DIR, 'static')
CORS(app, resources={r"/critique": {"origins": "https://localhost:5000"}})

# Load model
pose_estimator = TfPoseEstimator(graph_path=settings.GRAPH_PATH, target_size=settings.PROCESSING_DIMS)

"""
Static Files Serving
"""

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory(app.config['STATIC_FOLDER'], path)

"""
Front-end serving
"""
@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/login")
def login():
    return render_template('login.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/trainer")
def trainer():
    return render_template('trainer.html')

@app.route("/friends")
def friends():
    return render_template('friends.html')

@app.route("/chat")
def chat():
    return render_template('chat.html')

"""
Computer Vision
"""

@app.route("/critique", methods=['POST'])
def critique():
    """
    Endpoint for analyzing the video feed.
    input:
    {
        "image": base64EncodedString,
        "workout": string,
        "state": int,
        "rotate": (optional) int => (degrees),
        "side": (optional) char => ('F', 'L' or 'R'),
        "repcount": int
    }

    return:
    {
        "data": {
            "points": [],
            "critiques": [],
            "deviation": float,
            "state": int,
            "repcount": int
        },
    }

    states:
    0 - rest
    1 - down
    2 - up
    """
    if request.method == 'POST':
        # Read image from request
        encoded_string = request.form['image'].split(',')[-1]
        workout = request.form['workout']
        prev_state = int(request.form.get('state'))
        side = request.form.get('side')
        rotate = int(request.form.get('rotate'))
        rep_count = int(request.form.get('repCount'))

        if rotate == '0':
            rotate = False

        # Convert image string to image array
        image = convert.base64_to_array(encoded_string)

        # ---- Processing and analysis ----
        # Extract body parts
        body_parts = analyze.extract_body_parts(pose_estimator, image, rotate)
        if body_parts == -1:
            return jsonify({
                "data": {
                    "points": -1,
                    "critiques": -1,
                    "deviation": -1,
                    "state": prev_state,
                    "repCount": rep_count
                },
                "status": -1
            })
            
        # Assemble points array
        points = {}
        for i in range(18):
            try:
                points[i] = (body_parts[i].x, body_parts[i].y)
            except KeyError:
                points[i] = -1

        # Analyze workout from body parts
        deviation, critique, state = analyze.analyze_workout(body_parts, workout, prev_state, side)
        if prev_state == 2 and state == 1:
            rep_count += 1

        """ 
        TODO: 
        logic for rep counting
        logic for set counting
        database for storing reps and sets
        analysis for all workouts
        """

        # Send response with image
        return jsonify({
            "data": {
                "points": points,
                "critique": critique,
                "deviation": deviation,
                "state": state,
                "repCount": rep_count
            },
            "status": -1
        })

if __name__ == '__main__':
    app.run(
        threaded=True,
        ssl_context=('cert.pem', 'key.pem')
    )