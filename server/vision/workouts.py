from .helpers import *

def shoulderpress(body_parts, state, side):
    """
    Problems:
    -> Too deep on down
    -> Arms straight extended outwards
    """

    def elbow_height(body_parts):
        try:
            rshoulder_pos = bp_coordinates(body_parts, 2) 
            lshoulder_pos = bp_coordinates(body_parts, 5) 
            relbow_pos = bp_coordinates(body_parts, 3)
            lelbow_pos = bp_coordinates(body_parts, 6)
        except KeyError as e:
            return 0, 0

        return 240*(relbow_pos[1] - rshoulder_pos[1]), 240*(lelbow_pos[1] - lshoulder_pos[1])

    def horizontal_extension(body_parts):
        r, l = 0, 0
        try:
            rwrist_pos = bp_coordinates(body_parts, 4) 
            relbow_pos = bp_coordinates(body_parts, 3)
            r = 240*(relbow_pos[0] - rwrist_pos[0])
        except KeyError as e:
            r = 0
        try:
            lwrist_pos = bp_coordinates(body_parts, 7) 
            lelbow_pos = bp_coordinates(body_parts, 6)
            l = 240*(lelbow_pos[0] - lwrist_pos[0])
        except KeyError as e:
            l = 0

        return r, -l

    re_height, le_height = elbow_height(body_parts)
    rextension, lextension = horizontal_extension(body_parts)
    thresh = 15
    critique = ""
    if rextension > thresh and lextension > thresh:
        critique = "You're over extending both of your arms." \
                 + " Bring your wrist over your elbow."
    elif rextension > thresh:
        critique = "You're over extending your right arm." \
                 + " Bring your wrist over your elbow."
    elif lextension > thresh:
        critique = "You're over extending your left arm." \
                 + " Bring your wrist over your elbow."
    elif rextension < -thresh and lextension < -thresh:
        critique = "You're over bending both of your arms." \
                 + " Bring your wrist over your elbow."
    elif rextension < -thresh:
        critique = "You're over bending your right arm." \
                 + " Bring your wrist over your elbow."
    elif lextension < -thresh:
        critique = "You're over bending your right arm." \
                 + " Bring your wrist over your elbow."
    else:
        critique = "Nice form! Keep it up."

    return (rextension, lextension), critique, state


def plank(body_parts):
    """
    Problems:
    ->  Deviation in waist:
        Body Parts:
           L/R Shoulder
           L/R Hip
           L/R Ankle
        Percent Deviation:
           (OptimalAngle - AngleDetected)/OptimalAngle * 100
        Params:
            OptimalAngle = pi
            Threshold = 0.1
    """

    def optimal_height_hips(body_parts):
        shoulder_pos = bp_coordinates_average(body_parts, 2, 5) 
        ankle_pos = bp_coordinates_average(body_parts, 10, 13)
    
    def deviation_in_hips(body_parts, optimal_angle):
        """
        Calculate the deviation in the hips from an optimal angle.
        """
        shoulder_pos = bp_coordinates_average(body_parts, 2, 5)
        hip_pos = bp_coordinates_average(body_parts, 8, 11)    
        ankle_pos = bp_coordinates_average(body_parts, 10, 13)
        try:
            # calculate angle
            angle_detected = calculate_angle(shoulder_pos, hip_pos, ankle_pos)
        except TypeError as e:
            raise e
        # calculate percent deviation
        deviation = percent_deviation(optimal_angle, angle_detected)
        return deviation
    
    return deviation_in_hips(body_parts, math.pi)

def curls(body_parts, state, side):
    """
    side - left or right, depending on user
    
    Problems:
    ->  Horizontal deviation in humerous to upper body:
        Body parts:
            L//R Shoulder
            L//R Elbow
            L//R Hip
        Percent Deviation:
            (OptimalAngle - AngleDetected)/OptimalAngle * 100
        Params:
            OptimalAngle = 0
            Threshold = 0.1
    """

    def angle_of_elbow(body_parts, side):
        try:
            if side == 'L':
                shoulder_pos = bp_coordinates(body_parts, 5)
                elbow_pos = bp_coordinates(body_parts, 6)
                wrist_pos = bp_coordinates(body_parts, 7)
            elif side == 'R':
                shoulder_pos = bp_coordinates(body_parts, 2)
                elbow_pos = bp_coordinates(body_parts, 3)
                wrist_pos = bp_coordinates(body_parts, 4)
            else:
                return -1
        except KeyError as e:
            return -1

        angle_detected = calculate_angle(wrist_pos, elbow_pos, shoulder_pos)

        return angle_detected

    
    def deviation_of_elbow(body_parts, side, optimal_angle):
        """
        Calculate the angular deviation of the elbow from the optimal
        """
        try:
            if side == 'L':
                shoulder_pos = bp_coordinates(body_parts, 5)
                elbow_pos = bp_coordinates(body_parts, 6)
                hip_pos = bp_coordinates(body_parts, 11)
            elif side == 'R':
                shoulder_pos = bp_coordinates(body_parts, 2)
                elbow_pos = bp_coordinates(body_parts, 3)
                hip_pos = bp_coordinates(body_parts, 8)
            else:
                return -1
        except KeyError as e:
            return -1
        
        try:
            if shoulder_pos and hip_pos and elbow_pos:
                # calculate angle
                angle_detected = calculate_angle(hip_pos, shoulder_pos, elbow_pos)
            else:
                return -1
        except TypeError as e:
            raise e
        
        # calculate percent deviation
        deviation = percent_deviation(optimal_angle, angle_detected)
        return deviation

    # Determine critique
    critique = "Nice form!"
    deviation = deviation_of_elbow(body_parts, side, 0)
    if deviation > 0.25:
        try:
            if side == 'L':
                elbow_pos = bp_coordinates(body_parts, 6)
                hip_pos = bp_coordinates(body_parts, 11)

                if elbow_pos[0] < hip_pos[0]:
                    critique = "Move your elbow backward."
                else: 
                    critique = "Move your elbow forward."
            elif side == 'R':
                elbow_pos = bp_coordinates(body_parts, 3)
                hip_pos = bp_coordinates(body_parts, 8)

                if elbow_pos[0] < hip_pos[0]:
                    critique = "Move your elbow forward."
                else: 
                    critique = "Move your elbow backward."
            else:
                return -1
        except KeyError as e:
            return -1

    # Determine if state change
    elbow_angle = angle_of_elbow(body_parts, side)
    if state == 1 and elbow_angle > 2.7:
        state = 2
    elif state == 2 and elbow_angle < 0.7:
        state = 1


    return deviation, critique, state

def pushup(body_parts, state, side):
    """
    Problems:
    ->  Deviation in waist:
        Body Parts:
           L/R Shoulder
           L/R Hip
           L/R Ankle
        Percent Deviation:
           (OptimalAngle - AngleDetected)/OptimalAngle * 100
        Params:
            OptimalAngle = pi
            Threshold = 0.1
    """
    
    def deviation_in_hips(body_parts, optimal_angle):
        # average shoulders
        shoulder_pos = bp_coordinates_average(body_parts, 2, 5)
        # average hips
        hip_pos = bp_coordinates_average(body_parts, 8, 11)    
        # average ankles
        ankle_pos = bp_coordinates_average(body_parts, 10, 13)
        try:
            if shoulder_pos and hip_pos and ankle_pos:
                # calculate angle
                angle_detected = calculate_angle(shoulder_pos, hip_pos, ankle_pos)
            else:
                return -1
        except TypeError as e:
            raise e
        # calculate percent deviation
        deviation = percent_deviation(optimal_angle, angle_detected)
        return deviation

    return deviation_in_hips(body_parts, math.pi), 0, 0

def squats(body_parts, state, side):
    """
    Problems:
    - Squat depth
        Body Parts:
           L/R Ankle
           L/R Knee
           L/R Hip
        Percent Deviation:
           (OptimalAngle - AngleDetected)/OptimalAngle * 100
        Params:
            OptimalAngle = pi/2
            Threshold = TBD
    - Forward knee movement
        Body Parts:
            L/R Ankle
            L/R Knee
        Percent Deviation:
           (X_ANKLE - X_KNEE)/TibiaLength * 100
        Params:
            OptimalDeviation = 0
            Threshold = TBD
    - 'Divebombing'
        Body Parts:
            L/R Shoulder
            L/R Hip
        Percent Deviation:
           (X_SHOULDER - X_HIP)/TorsoLength * 100
        Params:
            OptimalDeviation = 0
            Threshold = TBD
    """
    def squat_depth_angle(body_parts):
        ankle = bp_coordinates_average(body_parts, 10, 13)
        knee = bp_coordinates_average(body_parts, 9, 12)
        hip = bp_coordinates_average(body_parts, 8, 11)
        try:
            if ankle and knee and hip:
                return calculate_angle(ankle, knee, hip)
            else:
                 return -1
        except TypeError as e:
            return -1

    def tibia_deviation(body_parts):
        ankle = bp_coordinates_average(body_parts, 10, 13)
        knee = bp_coordinates_average(body_parts, 9, 12)
        try:
            if ankle and knee:
                return 240*(ankle[0] - knee[0])**2
            else:
                return -1
        except:
            return -1
    
    squat_depth = squat_depth_angle(body_parts)
    if squat_depth < (math.pi/2):
        if state == 1:
            state = 2

    if squat_depth > (math.pi - 0.5):
        if state == 2:
            state = 1

    deviation = tibia_deviation(body_parts)
    if abs(deviation) > 1.45:
        critique = "Keep your knees above your toes!"
    else:
        critique = "Nice form! Keep it up."
        

    return squat_depth, critique, state