import cv2

from ar_module.angle_calculator import get_all_angles
from ar_module.pose_sequence import PoseSequenceManager
from ar_module.ar_overlay import (
    draw_user_skeleton,
    draw_ghost_skeleton,
    draw_direction_arrows,
    draw_hud,
    draw_session_complete,
    draw_ideal_pose_reference,
    load_ideal_pose_images,
)
from ar_module.pose_classification import PoseClassifier, load_ideal_poses
from ar_module.pose_comparison import compare_pose_prediction
from ar_module.pose_detection import PoseDetector

WINDOW_NAME = "YogaAI"


def window_was_closed(window_name=WINDOW_NAME):
    """Return True if the user closed the window with the X button."""
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


# Load ideal poses
ideal_poses = load_ideal_poses()
ideal_pose_images = load_ideal_pose_images()

# Initialize detector and classifier
detector = PoseDetector(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
classifier = PoseClassifier()

# Use a saved Yoga82 model if one exists, otherwise fall back to templates
classifier_model_path = "ar_module/data/yoga82_pose_classifier.json"
try:
    classifier.load(classifier_model_path)
except FileNotFoundError:
    classifier.fit_from_pose_templates(ideal_poses)

# Initialize pose sequence manager
sequence_manager = PoseSequenceManager()

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
)

print("Starting YogaAI AR System (fullscreen)...")
print("Press Q to quit | Press R to restart | Press F to toggle fullscreen")
print("Or click the window X to close")

fullscreen = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)

    # Session complete screen
    if sequence_manager.session_complete:
        frame = draw_session_complete(frame)
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if window_was_closed() or key == ord('q'):
            break
        if key == ord('r'):
            sequence_manager.reset()
        if key == ord('f'):
            fullscreen = not fullscreen
            mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)
        continue

    results = detector.process(frame)
    landmarks = detector.extract_landmarks(results)

    current_pose = sequence_manager.get_current_pose()
    pose_score = 0
    corrections = []
    validation = None

    if landmarks:
        user_angles = get_all_angles(landmarks)
        prediction = classifier.predict_from_angles(user_angles)
        comparison = compare_pose_prediction(
            prediction_result=prediction,
            target_pose=current_pose,
            user_angles=user_angles,
            ideal_poses=ideal_poses,
        )

        validation = comparison['validation']
        corrections_result = comparison['corrections_result']
        corrections = corrections_result['corrections']
        pose_score = comparison['validation_score']
        sequence_manager.update(comparison['pose_score'])

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
        sequence_manager.get_frames_to_pass(),
        sequence_manager.frames_passed,
        pass_threshold=sequence_manager.get_pass_threshold(),
    )

    frame = draw_ideal_pose_reference(
        frame, current_pose, ideal_pose_images
    )

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if window_was_closed() or key == ord('q'):
        break
    if key == ord('r'):
        sequence_manager.reset()
    if key == ord('f'):
        fullscreen = not fullscreen
        mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)

detector.close()

cap.release()
cv2.destroyAllWindows()
print("YogaAI stopped")
