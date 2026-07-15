# How many degrees difference is acceptable
THRESHOLD = 15


def validate_pose(user_angles, ideal_poses, pose_name):
    """
    Compares user angles against ideal angles for detected pose
    Returns which joints are correct and which are wrong

    user_angles: dict from angle_calculator
    ideal_poses: dict from ideal_poses.json
    pose_name: string — which pose was detected
    """
    if pose_name not in ideal_poses:
        return None

    pose_data = ideal_poses[pose_name]
    validation_result = {}
    correct_count = 0
    total_count = 0

    for joint_name, ideal_angle in pose_data.items():
        # Skip description
        if joint_name == "description":
            continue

        if joint_name in user_angles:
            user_angle = user_angles[joint_name]
            diff = user_angle - ideal_angle
            abs_diff = abs(diff)

            if abs_diff <= THRESHOLD:
                status = "correct"
                correct_count += 1
            else:
                status = "wrong"

            validation_result[joint_name] = {
                "user_angle": user_angle,
                "ideal_angle": ideal_angle,
                "difference": round(diff, 2),
                "abs_difference": round(abs_diff, 2),
                "status": status
            }
            total_count += 1

    # Calculate pose score
    pose_score = round((correct_count / total_count) * 100) \
        if total_count > 0 else 0

    return {
        "joints": validation_result,
        "pose_score": pose_score,
        "correct_joints": correct_count,
        "total_joints": total_count
    }


def test_pose_validator():
    """
    Test validator with slightly off angles
    """
    import json
    import os

    json_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "ideal_poses.json"
    )
    with open(json_path, "r") as f:
        ideal_poses = json.load(f)

    # Slightly off angles — some correct some wrong
    fake_angles = {
        "left_elbow": 168,    # correct (diff = 2)
        "right_elbow": 145,   # wrong (diff = 25)
        "left_shoulder": 173, # correct (diff = 2)
        "right_shoulder": 140,# wrong (diff = 35)
        "left_knee": 174,     # correct (diff = 1)
        "right_knee": 175,    # correct (diff = 0)
        "left_hip": 174,      # correct (diff = 1)
        "right_hip": 176      # correct (diff = 1)
    }

    result = validate_pose(
        fake_angles,
        ideal_poses,
        "pose_1_pranamasana"
    )

    print(f"\nPose Score: {result['pose_score']}%")
    print(f"Correct Joints: {result['correct_joints']}"
          f"/{result['total_joints']}")
    print("\nJoint Details:")

    for joint, data in result['joints'].items():
        status_icon = "✅" if data['status'] == "correct" else "❌"
        print(f"{status_icon} {joint}: "
              f"User={data['user_angle']}° "
              f"Ideal={data['ideal_angle']}° "
              f"Diff={data['difference']}°")

    print("\n✅ Pose validator working correctly!")


if __name__ == "__main__":
    test_pose_validator()