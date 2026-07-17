from .correction_engine import generate_corrections
from .pose_validator import LOOSE_POSES, validate_pose

LOOSE_CLASSIFIER_POSES = set(LOOSE_POSES)
POSE_CONFIDENCE_THRESHOLDS = {pose: 35 for pose in LOOSE_POSES}


def compare_pose_prediction(
    prediction_result,
    target_pose,
    user_angles,
    ideal_poses,
    confidence_threshold=70,
):
    """
    Compare the classifier result with the current sequence target.
    """
    validation_result = None
    corrections_result = {"corrections": [], "pose_score": 0, "needs_correction": False}

    if target_pose and user_angles:
        validation_result = validate_pose(user_angles, ideal_poses, target_pose)
        if validation_result is not None:
            corrections_result = generate_corrections(validation_result)

    predicted_pose = prediction_result.get("pose_name") if prediction_result else None
    predicted_confidence = prediction_result.get("confidence", 0) if prediction_result else 0
    effective_confidence = POSE_CONFIDENCE_THRESHOLDS.get(
        target_pose, confidence_threshold
    )
    matches_target = (
        predicted_pose == target_pose
        and predicted_confidence >= effective_confidence
    )

    validation_score = validation_result["pose_score"] if validation_result else 0

    if target_pose in LOOSE_CLASSIFIER_POSES:
        gated_score = validation_score
    else:
        gated_score = validation_score if matches_target else 0

    return {
        "prediction": prediction_result,
        "target_pose": target_pose,
        "matches_target": matches_target,
        "validation": validation_result,
        "corrections_result": corrections_result,
        "validation_score": validation_score,
        "pose_score": gated_score,
    }
