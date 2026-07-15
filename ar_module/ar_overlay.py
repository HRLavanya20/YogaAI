import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 100, 0)
BLACK = (0, 0, 0)

# Joint connections for drawing skeleton
JOINT_PAIRS = [
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 12),            # Shoulders
    (11, 23), (12, 24),  # Torso
    (23, 24),            # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
]

# Which landmark index maps to which joint name
LANDMARK_TO_JOINT = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    15: "left_wrist",
    16: "right_wrist"
}

# Which joint name maps back to landmark index
JOINT_TO_LANDMARK = {v: k for k, v in LANDMARK_TO_JOINT.items()}


def get_pixel_coords(landmark, frame_width, frame_height):
    """Convert normalized landmark to pixel coordinates"""
    x = int(landmark.x * frame_width)
    y = int(landmark.y * frame_height)
    return (x, y)


def draw_user_skeleton(frame, landmarks, validation_result):
    """
    Draws user skeleton with green/red joints
    Green = correct joint
    Red = wrong joint
    """
    h, w = frame.shape[:2]

    # Draw connections
    for (idx_a, idx_b) in JOINT_PAIRS:
        point_a = get_pixel_coords(landmarks[idx_a], w, h)
        point_b = get_pixel_coords(landmarks[idx_b], w, h)
        cv2.line(frame, point_a, point_b, WHITE, 2)

    # Draw joints with color based on validation
    for landmark_idx, joint_name in LANDMARK_TO_JOINT.items():
        point = get_pixel_coords(landmarks[landmark_idx], w, h)

        # Determine color
        if validation_result and joint_name in validation_result['joints']:
            status = validation_result['joints'][joint_name]['status']
            color = GREEN if status == "correct" else RED
        else:
            color = WHITE

        cv2.circle(frame, point, 8, color, -1)
        cv2.circle(frame, point, 8, BLACK, 1)

    return frame


def draw_ghost_skeleton(frame, landmarks, ideal_poses, pose_name):
    """
    Draws transparent ideal pose skeleton
    Shows where user SHOULD be
    """
    h, w = frame.shape[:2]

    # Create overlay for transparency
    overlay = frame.copy()

    # We approximate ghost skeleton using current landmark
    # positions shifted based on ideal angles
    # For now draw ideal reference as blue dots
    # (Full ghost skeleton requires inverse kinematics)
    for (idx_a, idx_b) in JOINT_PAIRS:
        point_a = get_pixel_coords(landmarks[idx_a], w, h)
        point_b = get_pixel_coords(landmarks[idx_b], w, h)
        cv2.line(overlay, point_a, point_b, BLUE, 2)

    # Apply transparency
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    return frame


def draw_direction_arrows(frame, landmarks, corrections):
    """
    Draws arrows on wrong joints showing
    which direction to move
    """
    h, w = frame.shape[:2]

    DIRECTION_VECTORS = {
        "up":     (0, -40),
        "down":   (0, 40),
        "extend": (40, 0),
        "bend":   (-40, 0),
        "adjust": (0, -30)
    }

    for correction in corrections:
        joint_name = correction['joint']
        direction = correction['direction']

        if joint_name in JOINT_TO_LANDMARK:
            landmark_idx = JOINT_TO_LANDMARK[joint_name]
            start = get_pixel_coords(
                landmarks[landmark_idx], w, h
            )

            dx, dy = DIRECTION_VECTORS.get(
                direction, (0, -30)
            )
            end = (start[0] + dx, start[1] + dy)

            cv2.arrowedLine(
                frame, start, end,
                YELLOW, 2, tipLength=0.4
            )

    return frame


def draw_hud(frame, pose_display_name,
             pose_score, progress,
             corrections, frames_to_pass,
             frames_passed):
    """
    Draws Heads Up Display on frame
    Shows pose name, score, progress, corrections
    """
    h, w = frame.shape[:2]

    # Background box for HUD
    cv2.rectangle(frame, (0, 0), (w, 110), BLACK, -1)
    cv2.rectangle(frame, (0, 0), (w, 110),
                  (50, 50, 50), 2)

    # Pose name
    cv2.putText(
        frame,
        pose_display_name,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75, WHITE, 2
    )

    # Progress
    cv2.putText(
        frame,
        f"Progress: {progress}",
        (w - 160, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, YELLOW, 2
    )

    # Score bar background
    cv2.rectangle(frame, (10, 50), (300, 75),
                  (50, 50, 50), -1)

    # Score bar fill
    bar_width = int((pose_score / 100) * 290)
    bar_color = GREEN if pose_score >= 80 else RED
    cv2.rectangle(frame, (10, 50),
                  (10 + bar_width, 75), bar_color, -1)

    # Score text
    cv2.putText(
        frame,
        f"Score: {pose_score}%",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, WHITE, 2
    )

    # Progress bar for holding pose
    hold_progress = int((frames_passed / frames_to_pass) * 290)
    cv2.rectangle(frame, (10, 82), (300, 102),
                  (50, 50, 50), -1)
    cv2.rectangle(frame, (10, 82),
                  (10 + hold_progress, 102),
                  (255, 165, 0), -1)
    cv2.putText(
        frame,
        "Hold:",
        (10, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, WHITE, 1
    )

    # Corrections at bottom
    if corrections:
        for i, correction in enumerate(corrections[:2]):
            cv2.putText(
                frame,
                f"→ {correction['message']}",
                (10, h - 60 + (i * 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, RED, 2
            )
    else:
        cv2.putText(
            frame,
            "✓ Great Pose! Hold it!",
            (10, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, GREEN, 2
        )

    return frame


def draw_session_complete(frame):
    """Shows session complete screen"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), BLACK, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(
        frame,
        "Surya Namaskar Complete!",
        (w//2 - 250, h//2 - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2, GREEN, 3
    )
    cv2.putText(
        frame,
        "Press R to restart or Q to quit",
        (w//2 - 220, h//2 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, WHITE, 2
    )
    return frame