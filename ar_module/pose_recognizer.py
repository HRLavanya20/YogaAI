import json
import os


def load_ideal_poses():
    """
    Loads ideal pose angles from JSON file
    Returns dictionary of all 12 poses
    """
    json_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "ideal_poses.json"
    )
    with open(json_path, "r") as f:
        poses = json.load(f)
    return poses


def recognize_pose(user_angles, ideal_poses):
    """
    Compares user angles against all 12 ideal poses
    Returns the closest matching pose name and confidence

    user_angles: dict of joint angles from angle_calculator
    ideal_poses: dict loaded from ideal_poses.json
    """
    best_match = None
    best_score = float('inf')  # lower is better

    for pose_name, pose_data in ideal_poses.items():
        total_diff = 0
        joint_count = 0

        for joint_name, ideal_angle in pose_data.items():
            # Skip description field
            if joint_name == "description":
                continue

            # Only compare joints we have data for
            if joint_name in user_angles:
                diff = abs(user_angles[joint_name] - ideal_angle)
                total_diff += diff
                joint_count += 1

        if joint_count > 0:
            avg_diff = total_diff / joint_count

            if avg_diff < best_score:
                best_score = avg_diff
                best_match = pose_name

    # Calculate confidence (lower diff = higher confidence)
    # Max possible avg diff is 180 degrees
    confidence = max(0, round((1 - best_score / 180) * 100, 1))

    # Get pose description
    description = ideal_poses[best_match]["description"] \
        if best_match else "Unknown"

    return {
        "pose_name": best_match,
        "description": description,
        "confidence": confidence,
        "avg_diff": round(best_score, 2)
    }


def test_pose_recognizer():
    """
    Test with fake angles matching pose 1 (Pranamasana)
    """
    # Fake angles close to Pranamasana
    fake_angles = {
        "left_elbow": 168,
        "right_elbow": 172,
        "left_shoulder": 173,
        "right_shoulder": 176,
        "left_knee": 174,
        "right_knee": 175,
        "left_hip": 174,
        "right_hip": 176
    }

    ideal_poses = load_ideal_poses()
    result = recognize_pose(fake_angles, ideal_poses)

    print(f"Detected Pose: {result['description']}")
    print(f"Pose Name: {result['pose_name']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Avg Difference: {result['avg_diff']} degrees")

    if "pranamasana" in result['pose_name']:
        print("✅ Pose recognizer working correctly!")
    else:
        print("❌ Wrong pose detected — check JSON angles")


if __name__ == "__main__":
    test_pose_recognizer()