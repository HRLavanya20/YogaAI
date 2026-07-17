import json
import os
from pathlib import Path

import cv2
import numpy as np

from .angle_calculator import JOINT_LANDMARKS, get_all_angles
from .pose_detection import PoseDetector


def load_ideal_poses():
    """Load the built-in 12 Surya Namaskar template poses."""
    json_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "ideal_poses.json",
    )
    with open(json_path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _angles_to_vector(angle_map, feature_keys):
    return np.array([
        angle_map[key] / 180.0 for key in feature_keys
    ], dtype=np.float32)


class PoseClassifier:
    """
    Trainable pose classifier built on angle features.

    The default training flow is:
    1. Use PoseDetector to extract landmarks from Yoga82 images.
    2. Convert landmarks into joint-angle features.
    3. Build a centroid per pose label.

    This keeps the classifier separate from the detection module and
    allows the model to be replaced later with a more advanced learner.
    """

    def __init__(self):
        self.feature_keys = list(JOINT_LANDMARKS.keys())
        self.class_centroids = {}
        self.class_sample_counts = {}
        self.class_average_distances = {}

    @property
    def is_trained(self):
        return bool(self.class_centroids)

    def fit_from_pose_templates(self, ideal_poses):
        """Build a quick fallback model from the embedded pose templates."""
        self.class_centroids = {}
        self.class_sample_counts = {}
        self.class_average_distances = {}

        for pose_name, pose_data in ideal_poses.items():
            feature_vector = self._vector_from_pose_template(pose_data)
            self.class_centroids[pose_name] = feature_vector.tolist()
            self.class_sample_counts[pose_name] = 1
            self.class_average_distances[pose_name] = 0.0

        return self

    def fit_from_directory(self, dataset_root, label_map=None, detector=None):
        """Train the classifier from a Yoga82-style image dataset."""
        dataset_root = Path(dataset_root)
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

        detector = detector or PoseDetector()
        samples_by_label = {}

        for image_path, label in self._iter_dataset_images(dataset_root):
            if label_map and label in label_map:
                label = label_map[label]

            frame = cv2.imread(str(image_path))
            if frame is None:
                continue

            results = detector.process(frame)
            landmarks = detector.extract_landmarks(results)
            if landmarks is None:
                continue

            feature_vector = _angles_to_vector(
                get_all_angles(landmarks),
                self.feature_keys,
            )
            samples_by_label.setdefault(label, []).append(feature_vector)

        try:
            ideal_poses = load_ideal_poses()
        except Exception:
            ideal_poses = None

        self._build_model(samples_by_label, ideal_poses)
        return self

    def predict_from_angles(self, user_angles):
        """Return the closest pose label for the provided joint angles."""
        if not self.is_trained:
            raise RuntimeError("PoseClassifier is not trained")

        user_vector = _angles_to_vector(user_angles, self.feature_keys)
        best_label = None
        best_distance = float("inf")

        for label, centroid in self.class_centroids.items():
            centroid_vector = np.array(centroid, dtype=np.float32)
            distance = float(np.linalg.norm(user_vector - centroid_vector))
            if distance < best_distance:
                best_distance = distance
                best_label = label

        max_distance = float(np.sqrt(len(self.feature_keys)))
        confidence = max(0.0, (1.0 - best_distance / max_distance) * 100.0)

        return {
            "pose_name": best_label,
            "confidence": round(confidence, 1),
            "distance": round(best_distance, 4),
        }

    def predict_from_landmarks(self, landmarks):
        """Convenience wrapper around predict_from_angles."""
        return self.predict_from_angles(get_all_angles(landmarks))

    def save(self, file_path):
        """Persist the learned centroids as JSON."""
        payload = {
            "feature_keys": self.feature_keys,
            "class_centroids": self.class_centroids,
            "class_sample_counts": self.class_sample_counts,
            "class_average_distances": self.class_average_distances,
        }

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)

    def load(self, file_path):
        """Load centroids from a saved JSON model."""
        file_path = Path(file_path)
        with file_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)

        self.feature_keys = payload.get("feature_keys", self.feature_keys)
        self.class_centroids = payload.get("class_centroids", {})
        self.class_sample_counts = payload.get("class_sample_counts", {})
        self.class_average_distances = payload.get("class_average_distances", {})
        return self

    def _build_model(self, samples_by_label, ideal_poses=None):
        self.class_centroids = {}
        self.class_sample_counts = {}
        self.class_average_distances = {}

        # Merge with ideal_poses keys to guarantee all 12 classes exist in the model
        all_labels = set(samples_by_label.keys())
        if ideal_poses:
            all_labels.update(ideal_poses.keys())

        for label in all_labels:
            samples = samples_by_label.get(label, [])
            if not samples:
                if ideal_poses and label in ideal_poses:
                    # Fallback to the math template pose
                    pose_data = ideal_poses[label]
                    feature_vector = self._vector_from_pose_template(pose_data)
                    self.class_centroids[label] = feature_vector.tolist()
                    self.class_sample_counts[label] = 0  # 0 indicates fallback template
                    self.class_average_distances[label] = 0.0
                continue

            stacked_samples = np.vstack(samples)
            centroid = stacked_samples.mean(axis=0)
            distances = np.linalg.norm(stacked_samples - centroid, axis=1)

            self.class_centroids[label] = centroid.tolist()
            self.class_sample_counts[label] = len(samples)
            self.class_average_distances[label] = float(np.mean(distances))

    def _vector_from_pose_template(self, pose_data):
        template_angles = {}
        for joint_name in self.feature_keys:
            template_angles[joint_name] = pose_data.get(joint_name, 0)
        return _angles_to_vector(template_angles, self.feature_keys)

    @staticmethod
    def _iter_dataset_images(dataset_root):
        """
        Yield (image_path, label) pairs from either of these layouts:
        1. dataset_root/class_name/image.jpg
        2. dataset_root/split/class_name/image.jpg
        """
        image_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}

        for first_level in sorted(dataset_root.iterdir()):
            if not first_level.is_dir():
                continue

            child_dirs = [path for path in sorted(first_level.iterdir()) if path.is_dir()]
            if child_dirs:
                for class_folder in child_dirs:
                    label = class_folder.name
                    for image_path in sorted(class_folder.iterdir()):
                        if image_path.suffix.lower() in image_suffixes:
                            yield image_path, label
                continue

            label = first_level.name
            for image_path in sorted(first_level.iterdir()):
                if image_path.suffix.lower() in image_suffixes:
                    yield image_path, label

