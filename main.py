import cv2
import mediapipe as mp

from ar_module.angle_calculator import get_all_angles
from ar_module.pose_validator import validate_pose
from ar_module.correction_engine import generate_corrections
from ar_module.pose_sequence import PoseSequenceManager
from ar_module.ar_overlay import (
    draw_user_skeleton,
    draw_ghost_skeleton,
    draw_direction_arrows,
    draw_hud,
    draw_session_complete
)
from ar_module.pose_recognizer import load_ideal_poses

# Initialize MediaPipe
mp_pose = mp.solutions.pose

# Load ideal poses
ideal_poses = load_ideal_poses()

# Initialize pose sequence manager
sequence_manager = PoseSequenceManager()

# Open webcam
cap = cv2.VideoCapture(0)

print("✅ Starting YogaAI AR System...")
print("Press Q to quit | Press R to restart")

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror view
        frame = cv2.flip(frame, 1)

        # Session complete screen
        if sequence_manager.session_complete:
            frame = draw_session_complete(frame)
            cv2.imshow('YogaAI', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                sequence_manager.reset()
            continue

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get current pose from sequence
            current_pose = sequence_manager.get_current_pose()

            # Calculate user angles
            user_angles = get_all_angles(landmarks)

            # Validate against current pose only
            validation = validate_pose(
                user_angles,
                ideal_poses,
                current_pose
            )

            # Generate corrections
            corrections_result = generate_corrections(validation)
            corrections = corrections_result['corrections']
            pose_score = validation['pose_score']

            # Update sequence manager
            sequence_manager.update(pose_score)

            # Draw AR overlays
            frame = draw_ghost_skeleton(
                frame, landmarks,
                ideal_poses, current_pose
            )
            frame = draw_user_skeleton(
                frame, landmarks, validation
            )
            frame = draw_direction_arrows(
                frame, landmarks, corrections
            )
            frame = draw_hud(
                frame,
                sequence_manager.get_current_display_name(),
                pose_score,
                sequence_manager.get_progress(),
                corrections,
                sequence_manager.FRAMES_TO_PASS,
                sequence_manager.frames_passed
            )

        # Show frame
        cv2.imshow('YogaAI', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            sequence_manager.reset()

cap.release()
cv2.destroyAllWindows()
print("✅ YogaAI stopped")