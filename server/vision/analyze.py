import os
import time
import math

import cv2
import numpy as np
import imutils

import settings
from helpers.estimator import TfPoseEstimator
from helpers.networks import get_graph_path, model_wh
from .helpers import *
from vision import workouts

def extract_body_parts(estimator, image, rotate=None):
    """
    Perform inference on image for body part locations.

    returns the best human subject object
    """
   
    if rotate:
        image = imutils.rotate(image, rotate)

    # get human pose positions
    humans = estimator.inference(image)
    if not len(humans):
        return -1

    # extract the best subject
    subject = best_subject(humans)

    if subject:
        body_parts = subject.body_parts
    else:
        return -1

    if rotate:
        for i in body_parts:
            try:
                bp = (body_parts[i].x, body_parts[i].y)
                rt = rotation((0,0), bp, math.radians(-rotate))

                body_parts[i].x = rt[0]
                body_parts[i].y = rt[1]
            except AttributeError:
                pass

    return body_parts


def analyze_workout(body_parts, workout, state, side=None):
    """
    Run the appropriate analyzer.

    return deviation and appropriate critiques
    """

    analyzer = getattr(workouts, workout)

    if side:
        deviation, critique, state = analyzer(body_parts, state, side)
    else:
        deviation, critique, state = analyzer(body_parts, state)

    return deviation, critique, state