# How many degrees difference is acceptable by default (strict)
THRESHOLD = 15

# Looser angle tolerance for poses 2–12
LOOSE_THRESHOLD = 40

LOOSE_POSES = {
    "pose_2_hasta_uttanasana",
    "pose_3_hasta_padasana",
    "pose_4_ashwa_sanchalanasana",
    "pose_5_dandasana",
    "pose_6_ashtanga_namaskara",
    "pose_7_bhujangasana",
    "pose_8_adho_mukha_svanasana",
    "pose_9_ashwa_sanchalanasana_2",
    "pose_10_hasta_padasana_2",
    "pose_11_hasta_uttanasana_2",
    "pose_12_pranamasana_2",
}

POSE_THRESHOLDS = {pose: LOOSE_THRESHOLD for pose in LOOSE_POSES}

POSE_REQUIRED_JOINTS = {
    "pose_2_hasta_uttanasana": {
        "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
    },
    "pose_3_hasta_padasana": {
        "left_hip", "right_hip", "left_knee", "right_knee",
    },
    "pose_4_ashwa_sanchalanasana": {
        "left_knee", "right_knee", "left_hip", "right_hip",
    },
    "pose_5_dandasana": {
        "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
        "left_knee", "right_knee",
    },
    "pose_6_ashtanga_namaskara": {
        "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
        "left_knee", "right_knee",
    },
    "pose_7_bhujangasana": {
        "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
    },
    "pose_8_adho_mukha_svanasana": {
        "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
        "left_hip", "right_hip",
    },
    "pose_9_ashwa_sanchalanasana_2": {
        "left_knee", "right_knee", "left_hip", "right_hip",
    },
    "pose_10_hasta_padasana_2": {
        "left_hip", "right_hip", "left_knee", "right_knee",
    },
    "pose_11_hasta_uttanasana_2": {
        "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
    },
    "pose_12_pranamasana_2": {
        "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
    },
}


def validate_pose(user_angles, ideal_poses, pose_name):
    """
    Compares user angles against ideal angles for detected pose
    Returns which joints are correct and which are wrong
    """
    if pose_name not in ideal_poses:
        return None

    pose_data = ideal_poses[pose_name]
    threshold = POSE_THRESHOLDS.get(pose_name, THRESHOLD)
    required_joints = POSE_REQUIRED_JOINTS.get(pose_name)

    validation_result = {}
    correct_count = 0
    total_count = 0

    for joint_name, ideal_angle in pose_data.items():
        if joint_name == "description":
            continue
        if required_joints is not None and joint_name not in required_joints:
            continue

        if joint_name in user_angles:
            user_angle = user_angles[joint_name]
            diff = user_angle - ideal_angle
            abs_diff = abs(diff)

            if abs_diff <= threshold:
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

    pose_score = round((correct_count / total_count) * 100) if total_count > 0 else 0

    return {
        "joints": validation_result,
        "pose_score": pose_score,
        "correct_joints": correct_count,
        "total_joints": total_count
    }
