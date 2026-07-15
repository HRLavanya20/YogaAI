import numpy as np

# These are the landmark index pairs for each joint
# Format: (point_a, point_b, point_c)
# point_b is the joint whose angle we calculate
JOINT_LANDMARKS = {
    "left_elbow":    (11, 13, 15),
    "right_elbow":   (12, 14, 16),
    "left_shoulder": (13, 11, 23),
    "right_shoulder":(14, 12, 24),
    "left_knee":     (23, 25, 27),
    "right_knee":    (24, 26, 28),
    "left_hip":      (11, 23, 25),
    "right_hip":     (12, 24, 26)
}


def calculate_angle(a, b, c):
    """
    Calculate angle at point b given three points a, b, c
    a, b, c are each [x, y] coordinates
    Returns angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    # Normalize to 0-180 range
    if angle > 180.0:
        angle = 360 - angle

    return round(angle, 2)


def get_all_angles(landmarks):
    """
    Takes 33 landmarks from MediaPipe
    Returns dictionary of all joint angles
    
    landmarks: list of mediapipe landmark objects
               each has .x, .y, .z (normalized 0-1)
    """
    angles = {}

    for joint_name, (a_idx, b_idx, c_idx) in JOINT_LANDMARKS.items():
        # Get x, y coordinates for each point
        # MediaPipe gives normalized values (0 to 1)
        point_a = [landmarks[a_idx].x, landmarks[a_idx].y]
        point_b = [landmarks[b_idx].x, landmarks[b_idx].y]
        point_c = [landmarks[c_idx].x, landmarks[c_idx].y]

        # Calculate angle
        angle = calculate_angle(point_a, point_b, point_c)
        angles[joint_name] = angle

    return angles


def test_angle_calculator():
    """
    Quick test to verify angle calculation works
    Tests a simple 90 degree angle
    """
    # Simple test: three points forming 90 degrees
    a = [0, 1]   # top
    b = [0, 0]   # center (joint)
    c = [1, 0]   # right

    angle = calculate_angle(a, b, c)
    print(f"Test angle (should be 90): {angle}")

    if abs(angle - 90) < 1:
        print("✅ Angle calculator working correctly!")
    else:
        print("❌ Something is wrong with angle calculator")


if __name__ == "__main__":
    test_angle_calculator()