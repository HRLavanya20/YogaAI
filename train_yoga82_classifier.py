"""
Train a reusable Yoga-82 pose classifier and save it as JSON.

Expected dataset layouts:
1. dataset_root/class_name/image.jpg
2. dataset_root/train/class_name/image.jpg
3. dataset_root/validation/class_name/image.jpg

Run example:
python train_yoga82_classifier.py --dataset-root D:\data\Yoga82 --output ar_module/data/yoga82_pose_classifier.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ar_module.pose_classification import PoseClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train Yoga-82 pose classifier")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root folder of the extracted Yoga-82 dataset",
    )
    parser.add_argument(
        "--output",
        default="ar_module/data/yoga82_pose_classifier.json",
        help="Where to save the trained classifier JSON",
    )
    parser.add_argument(
        "--label-map",
        default=None,
        help="Optional JSON file mapping dataset folder names to the 12 Surya Namaskar labels",
    )
    return parser.parse_args()


def load_label_map(label_map_path):
    if not label_map_path:
        return None

    path = Path(label_map_path)
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def main():
    args = parse_args()
    label_map = load_label_map(args.label_map)

    classifier = PoseClassifier()
    classifier.fit_from_directory(args.dataset_root, label_map=label_map)
    classifier.save(args.output)

    print(f"Saved trained classifier to {Path(args.output).resolve()}")
    print(f"Classes learned: {len(classifier.class_centroids)}")


if __name__ == "__main__":
    main()
