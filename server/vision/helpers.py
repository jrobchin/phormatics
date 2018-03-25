import math
import numpy

# def rotation(a, angle):
#     b = [[numpy.cos(angle),-numpy.sin(angle)],
#          [numpy.sin(angle), numpy.cos(angle)]]
#     c = numpy.matmul(a,b) + (1, 0)
#     return(c)

def rotation(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return 1-qx, -qy

def is_hip_higher(a, h, s):
	v = ((s[0]-a[0]),(s[1]-a[1]))
	return (h[1] > a[1] + ((v[1] * h[0]) / v[0]))

def calculate_angle(v1, v0, v2):
    """
    Calculate angle from v1 to v0 to v2.

    returns -1 if zero check fails
    returns an angle if calculation if valid
    """
    x1, x2 = v1[0] - v0[0], v2[0] - v0[0]
    y1, y2 = v1[1] - v0[1], v2[1] - v0[1]
    dot_product = x1 * x2 + y1 * y2
    norm_product = math.sqrt(((x1 * x1) + (y1 * y1)) * ((x2 * x2) + (y2 * y2)))
    
    if (norm_product == 0):
        return -1
    
    return numpy.arccos(dot_product / norm_product)

def percent_deviation(optimal_angle, angle_detected):
    return abs(optimal_angle - angle_detected)

def bp_coordinates(body_parts, idx):
    """
    Convenience method for getting (x, y) coordinates for
    a given body part id.

    returns tuple of (x, y) coordinates
    """

    return (body_parts[idx].x, body_parts[idx].y)

def bp_coordinates_average(body_parts, idx1, idx2):
    """
    Given two body part ids, calculate the mid-point.
    Useful for finding the average height of symmetric 
    body parts (i.e. shoulders).

    If only one body part has been located, return those coordinates.

    returns a tuple of (x, y) coordinates
    """

    if idx1 in body_parts.keys() and idx2 in body_parts.keys():
        return ((body_parts[idx1].x + body_parts[idx2].x)/2, (body_parts[idx1].y + body_parts[idx2].y)/2) 
    elif idx1 in body_parts.keys():
        return bp_coordinates(body_parts, idx1)
    elif idx2 in body_parts.keys():
        return bp_coordinates(body_parts, idx2)
    else: 
        return False

def best_subject(humans):
    """ 
    Determine which human has the largest torso using an 
    approximation of the shoulders and hips.

    Body Part ids:
        Right Shoulder: 2
        Left Shoulder: 5
        Right Hip: 8
        Left Hip: 11

    returns the most likely best subject for tracking
    """

    human = None # placeholder
    largest_torso = 0
    for h in humans:
        try:
            # get body part positions
            shoulder_pos = bp_coordinates_average(h.body_parts, 2, 5)
            hip_pos = bp_coordinates_average(h.body_parts, 8, 11)
            
            if shoulder_pos and hip_pos:
                # compute graphical distance betweek the shoulders and hips
                torso = (hip_pos[0] - shoulder_pos[0])**2 + (hip_pos[1] - shoulder_pos[1])**2
                torso = math.sqrt(torso)
                
                # check if it's a new best candidate
                if torso > largest_torso:
                    largest_torso = torso
                    human = h
        except KeyError as e:
            pass
        
    return human