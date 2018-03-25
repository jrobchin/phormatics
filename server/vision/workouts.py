import helpers

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
        shoulder_pos = helpers.bp_coordinates_average(body_parts, 2, 5) 
        ankle_pos = helpers.bp_coordinates_average(body_parts, 10, 13)
    
    def deviation_in_hips(body_parts, optimal_angle):
        """
        Calculate the deviation in the hips from an optimal angle.
        """
        shoulder_pos = helpers.bp_coordinates_average(body_parts, 2, 5)
        hip_pos = helpers.bp_coordinates_average(body_parts, 8, 11)    
        ankle_pos = helpers.bp_coordinates_average(body_parts, 10, 13)
        try:
            # calculate angle
            angle_detected = helpers.calculate_angle(shoulder_pos, hip_pos, ankle_pos)
        except TypeError as e:
            raise e
        # calculate percent deviation
        deviation = helpers.percent_deviation(optimal_angle, angle_detected)
        return deviation
    
    return deviation_in_hips(body_parts, math.pi)

def curl(body_parts, side):
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
    
    def deviation_of_elbow(body_parts, side, optimal_angle):
        """
        Calculate the angular deviation of the elbow from the optimal
        """
        try:
            if side == 'L':
                shoulder_pos = helpers.bp_coordinates(body_parts, 5)
                elbow = helpers.bp_coordinates(body_parts, 6)
                hip_pos = helpers.bp_coordinates(body_parts, 11)
            elif side == 'R':
                shoulder_pos = helpers.bp_coordinates(body_parts, 2)
                elbow = helpers.bp_coordinates(body_parts, 3)
                hip_pos = helpers.bp_coordinates(body_parts, 8)
            else:
                return -1
        except KeyError as e:
            return -1
        
        try:
            if shoulder_pos and hip_pos and ankle_pos:
                # calculate angle
                angle_detected = helpers.calculate_angle(shoulder_pos, hip_pos, ankle_pos)
            else:
                return -1
        except TypeError as e:
            raise e
        
        # calculate percent deviation
        deviation = helpers.percent_deviation(optimal_angle, angle_detected)
        return deviation
        
    return deviation_of_elbow(body_parts, side, 0)

def pushup(body_parts):
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
        shoulder_pos = helpers.bp_coordinates_average(body_parts, 2, 5)
        # average hips
        hip_pos = helpers.bp_coordinates_average(body_parts, 8, 11)    
        # average ankles
        ankle_pos = helpers.bp_coordinates_average(body_parts, 10, 13)
        try:
            if shoulder_pos and hip_pos and ankle_pos:
                # calculate angle
                angle_detected = helpers.calculate_angle(shoulder_pos, hip_pos, ankle_pos)
            else:
                return -1
        except TypeError as e:
            raise e
        # calculate percent deviation
        deviation = helpers.percent_deviation(optimal_angle, angle_detected)
        return deviation

    return deviation_in_hips(body_parts, math.pi)

def squat(body_parts):
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
    def squat_depth_angle(body_parts, optimal_angle):
        ankle = helpers.bp_coordinates_average(body_parts, 10, 13)
        knee = helpers.bp_coordinates_average(body_parts, 9, 12)
        hip = helpers.bp_coordinates_average(body_parts, 8, 11)
        try:
            if ankle and knee and hip:
                return helpers.calculate_angle(ankle, knee, hip)
            else:
                 return -1
        except TypeError as e:
            return -1

    def tibia_deviation(body_parts):
        ankle = helpers.bp_coordinates_average(body_parts, 10, 13)
        knee = helpers.bp_coordinates_average(body_parts, 9, 12)
        try:
            if ankle and knee:
                return abs(ankle[0] - knee[0])
            else:
                return -1
        except:
            return -1
    
    return squat_depth_angle(body_parts, (math.pi)/2), tibia_deviation(body_parts)