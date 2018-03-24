import math
import numpy

def calculate_angle(v1, v0, v2):
    x1, x2 = v1[0] - v0[0], v2[0] - v0[0]
    y1, y2 = v1[1] - v0[1], v2[1] - v0[1]
    dot_product = x1 * x2 + y1 * y2
    norm_product = math.sqrt(((x1 * x1) + (y1 * y1)) * ((x2 * x2) + (y2 * y2)))
    
    if (norm_product == 0):
        return -1
    
    return numpy.arccos(dot_product / norm_product)