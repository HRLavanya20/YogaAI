import cv2
import mediapipe as mp


class PoseDetector:
    """
    Wraps MediaPipe pose estimation behind a small detection API.
    """

    def __init__(
        self,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr):
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._pose.process(rgb_frame)

    @staticmethod
    def extract_landmarks(results):
        if results and results.pose_landmarks:
            return results.pose_landmarks.landmark
        return None

    def close(self):
        if self._pose:
            self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
