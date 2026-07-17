"""
Download and retain only the Yoga-82 poses needed for Surya Namaskar.

The Yoga-82 repository stores per-pose image URLs in TXT files. This script
reads those link files, downloads the images, and writes a clean local subset
that matches the pose labels used by YogaAI.

Example:
python prepare_yoga82_subset.py --links-root ar_module/data/yoga_dataset_links --output-root ar_module/data/yoga82_subset
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


TARGET_POSE_SOURCES = {
    "pose_1_pranamasana": [],
    "pose_2_hasta_uttanasana": [],
    "pose_3_hasta_padasana": ["Standing_Forward_Bend_pose_or_Uttanasana_.txt"],
    "pose_4_ashwa_sanchalanasana": ["Low_Lunge_pose_or_Anjaneyasana_.txt"],
    "pose_5_dandasana": ["Staff_Pose_or_Dandasana_.txt"],
    "pose_6_ashtanga_namaskara": ["Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_.txt"],
    "pose_7_bhujangasana": ["Cobra_Pose_or_Bhujangasana_.txt"],
    "pose_8_adho_mukha_svanasana": ["Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_.txt"],
    "pose_9_ashwa_sanchalanasana_2": ["Low_Lunge_pose_or_Anjaneyasana_.txt"],
    "pose_10_hasta_padasana_2": ["Standing_Forward_Bend_pose_or_Uttanasana_.txt"],
    "pose_11_hasta_uttanasana_2": [],
    "pose_12_pranamasana_2": [],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a Yoga-82 Surya Namaskar subset")
    parser.add_argument(
        "--links-root",
        default="ar_module/data/yoga_dataset_links",
        help="Folder containing the Yoga-82 pose link TXT files",
    )
    parser.add_argument(
        "--output-root",
        default="ar_module/data/yoga82_subset",
        help="Folder where the retained images will be downloaded",
    )
    parser.add_argument(
        "--pose",
        action="append",
        dest="poses",
        help="Optional pose label to include. Repeat to limit the subset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum images to download per pose label",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete the output folder before downloading",
    )
    return parser.parse_args()


def read_link_file(file_path):
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if "\t" in line:
            parts = line.split("\t", 1)
        else:
            parts = line.split(",", 1)

        if len(parts) != 2:
            continue

        relative_path = parts[0].strip()
        url = parts[1].strip()
        if relative_path and url:
            yield relative_path, url


def download_file(url, destination_path):
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=20) as response, destination_path.open("wb") as file_handle:
        shutil.copyfileobj(response, file_handle)


def infer_filename(relative_path, url):
    relative_name = Path(relative_path).name
    if relative_name:
        return relative_name

    parsed_url = urlparse(url)
    url_name = Path(parsed_url.path).name
    return url_name or "image.jpg"


def main():
    args = parse_args()

    links_root = Path(args.links_root)
    output_root = Path(args.output_root)

    if args.clear_output and output_root.exists():
        shutil.rmtree(output_root)

    if not links_root.exists():
        raise FileNotFoundError(f"Links folder not found: {links_root}")

    selected_poses = args.poses or list(TARGET_POSE_SOURCES.keys())
    downloaded = 0
    skipped_poses = []

    for target_pose in selected_poses:
        source_files = TARGET_POSE_SOURCES.get(target_pose, [])
        if not source_files:
            skipped_poses.append(target_pose)
            continue

        target_dir = output_root / target_pose
        existing_count = len(list(target_dir.glob("*"))) if target_dir.exists() else 0
        pose_downloaded = existing_count

        if args.limit is not None and pose_downloaded >= args.limit:
            continue

        for source_file_name in source_files:
            if args.limit is not None and pose_downloaded >= args.limit:
                break

            source_file = links_root / source_file_name
            if not source_file.exists():
                continue

            for relative_path, url in read_link_file(source_file):
                if args.limit is not None and pose_downloaded >= args.limit:
                    break

                filename = infer_filename(relative_path, url)
                output_name = f"{source_file.stem}_{filename}"
                destination = target_dir / output_name

                if destination.exists():
                    continue

                try:
                    download_file(url, destination)
                    downloaded += 1
                    pose_downloaded += 1
                except Exception as exc:
                    print(f"[skip] {url} -> {exc}")

    print(f"Downloaded {downloaded} images into {output_root.resolve()}")
    if skipped_poses:
        print("These poses have no source mapping yet:")
        for pose_name in skipped_poses:
            print(f"- {pose_name}")


if __name__ == "__main__":
    main()
