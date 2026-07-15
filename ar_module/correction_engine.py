def generate_corrections(validation_result):
    """
    Takes validation result
    Returns list of correction messages and directions

    validation_result: dict from pose_validator
    """
    corrections = []

    for joint_name, data in validation_result['joints'].items():
        if data['status'] == "wrong":
            diff = data['difference']
            correction = get_correction_message(joint_name, diff)
            corrections.append({
                "joint": joint_name,
                "message": correction['message'],
                "direction": correction['direction'],
                "severity": get_severity(data['abs_difference'])
            })

    return {
        "corrections": corrections,
        "pose_score": validation_result['pose_score'],
        "needs_correction": len(corrections) > 0
    }


def get_correction_message(joint_name, diff):
    """
    Returns correction message and direction for a joint
    diff > 0 means user angle too large — needs to decrease
    diff < 0 means user angle too small — needs to increase
    """
    messages = {
        "left_elbow": {
            "increase": {
                "message": "Straighten your left elbow",
                "direction": "extend"
            },
            "decrease": {
                "message": "Bend your left elbow more",
                "direction": "bend"
            }
        },
        "right_elbow": {
            "increase": {
                "message": "Straighten your right elbow",
                "direction": "extend"
            },
            "decrease": {
                "message": "Bend your right elbow more",
                "direction": "bend"
            }
        },
        "left_shoulder": {
            "increase": {
                "message": "Raise your left arm higher",
                "direction": "up"
            },
            "decrease": {
                "message": "Lower your left arm",
                "direction": "down"
            }
        },
        "right_shoulder": {
            "increase": {
                "message": "Raise your right arm higher",
                "direction": "up"
            },
            "decrease": {
                "message": "Lower your right arm",
                "direction": "down"
            }
        },
        "left_knee": {
            "increase": {
                "message": "Straighten your left knee",
                "direction": "extend"
            },
            "decrease": {
                "message": "Bend your left knee more",
                "direction": "bend"
            }
        },
        "right_knee": {
            "increase": {
                "message": "Straighten your right knee",
                "direction": "extend"
            },
            "decrease": {
                "message": "Bend your right knee more",
                "direction": "bend"
            }
        },
        "left_hip": {
            "increase": {
                "message": "Straighten your left hip",
                "direction": "extend"
            },
            "decrease": {
                "message": "Lower your left hip",
                "direction": "down"
            }
        },
        "right_hip": {
            "increase": {
                "message": "Straighten your right hip",
                "direction": "extend"
            },
            "decrease": {
                "message": "Lower your right hip",
                "direction": "down"
            }
        }
    }

    if joint_name in messages:
        if diff > 0:
            return messages[joint_name]["decrease"]
        else:
            return messages[joint_name]["increase"]

    return {
        "message": f"Adjust your {joint_name.replace('_', ' ')}",
        "direction": "adjust"
    }


def get_severity(abs_diff):
    """
    Returns severity level based on how wrong the angle is
    """
    if abs_diff <= 30:
        return "mild"
    elif abs_diff <= 60:
        return "moderate"
    else:
        return "severe"


def test_correction_engine():
    """
    Test correction engine with fake validation result
    """
    fake_validation = {
        "pose_score": 75,
        "joints": {
            "left_elbow": {
                "status": "correct",
                "difference": -2,
                "abs_difference": 2
            },
            "right_elbow": {
                "status": "wrong",
                "difference": -25,
                "abs_difference": 25
            },
            "left_shoulder": {
                "status": "correct",
                "difference": -2,
                "abs_difference": 2
            },
            "right_shoulder": {
                "status": "wrong",
                "difference": -35,
                "abs_difference": 35
            }
        }
    }

    result = generate_corrections(fake_validation)

    print(f"Needs Correction: {result['needs_correction']}")
    print(f"Pose Score: {result['pose_score']}%")
    print(f"\nCorrections needed: {len(result['corrections'])}")

    for correction in result['corrections']:
        print(f"\n❌ Joint: {correction['joint']}")
        print(f"   Message: {correction['message']}")
        print(f"   Direction: {correction['direction']}")
        print(f"   Severity: {correction['severity']}")

    print("\n✅ Correction engine working correctly!")


if __name__ == "__main__":
    test_correction_engine()