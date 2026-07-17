"""
Rule based mudra detection
Based on finger states and distances
from MediaPipe 21 hand landmarks

Landmark indices:
0  = Wrist
1  = Thumb CMC
2  = Thumb MCP  
3  = Thumb IP
4  = Thumb TIP
5  = Index MCP
6  = Index PIP
7  = Index DIP
8  = Index TIP
9  = Middle MCP
10 = Middle PIP
11 = Middle DIP
12 = Middle TIP
13 = Ring MCP
14 = Ring PIP
15 = Ring DIP
16 = Ring TIP
17 = Pinky MCP
18 = Pinky PIP
19 = Pinky DIP
20 = Pinky TIP
"""

import numpy as np


def get_distance(lm1, lm2):
    """
    Calculate distance between two landmarks
    """
    return np.sqrt(
        (lm1.x - lm2.x) ** 2 +
        (lm1.y - lm2.y) ** 2
    )


def is_touching(lm1, lm2, threshold=0.05):
    """
    Check if two landmarks are touching
    threshold controls sensitivity
    """
    return get_distance(lm1, lm2) < threshold


def get_finger_states(landmarks):
    """
    Returns dict of finger states
    True = straight/extended
    False = bent/folded

    Uses tip vs pip comparison
    Lower y = higher on screen = extended
    """
    return {
        "thumb": landmarks[4].x < landmarks[3].x,
        "index": landmarks[8].y < landmarks[6].y,
        "middle": landmarks[12].y < landmarks[10].y,
        "ring": landmarks[16].y < landmarks[14].y,
        "pinky": landmarks[20].y < landmarks[18].y
    }


def detect_gyan_mudra(landmarks):
    """
    Gyan Mudra:
    Index fingertip touches thumb tip
    Middle, ring, pinky all straight
    """
    fingers = get_finger_states(landmarks)

    index_touches_thumb = is_touching(
        landmarks[8],   # index tip
        landmarks[4],   # thumb tip
        threshold=0.06
    )

    middle_straight = fingers["middle"]
    ring_straight = fingers["ring"]
    pinky_straight = fingers["pinky"]

    if (index_touches_thumb and
            middle_straight and
            ring_straight and
            pinky_straight):
        return True, "Index tip touches thumb tip ✓"
    return False, "Touch index fingertip to thumb tip"


def detect_hakini_mudra(landmarks_left,
                         landmarks_right):
    """
    Hakini Mudra — needs BOTH hands
    All 5 fingertips of both hands touch
    each other in dome shape
    """
    touches = [
        is_touching(
            landmarks_left[4],
            landmarks_right[4],
            threshold=0.08
        ),   # thumbs
        is_touching(
            landmarks_left[8],
            landmarks_right[8],
            threshold=0.08
        ),   # index
        is_touching(
            landmarks_left[12],
            landmarks_right[12],
            threshold=0.08
        ),  # middle
        is_touching(
            landmarks_left[16],
            landmarks_right[16],
            threshold=0.08
        ),   # ring
        is_touching(
            landmarks_left[20],
            landmarks_right[20],
            threshold=0.08
        ),  # pinky
    ]

    touches_count = sum(touches)

    if touches_count >= 4:
        return True, "All fingertips touching ✓"
    return False, \
        f"Bring all fingertips together ({touches_count}/5)"


def detect_vayu_mudra(landmarks):
    """
    Vayu Mudra:
    Index finger folds to BASE of thumb
    Thumb crosses OVER index finger
    Middle, ring, pinky straight
    """
    fingers = get_finger_states(landmarks)

    # Index tip should be near thumb MCP (base)
    index_folded_to_base = is_touching(
        landmarks[8],   # index tip
        landmarks[2],   # thumb MCP (base)
        threshold=0.07
    )

    # Index should NOT be straight (it's folded)
    index_bent = not fingers["index"]

    middle_straight = fingers["middle"]
    ring_straight = fingers["ring"]
    pinky_straight = fingers["pinky"]

    if (index_folded_to_base and
            index_bent and
            middle_straight and
            ring_straight and
            pinky_straight):
        return True, "Index folded to thumb base ✓"
    return False, \
        "Fold index finger to base of thumb"


def detect_apan_vayu_mudra(landmarks):
    """
    Apan Vayu Mudra — Very complex:
    Index finger folds DOWN (bent)
    Thumb touches BOTH middle AND ring fingertips
    Pinky stays straight
    """
    fingers = get_finger_states(landmarks)

    # Index must be folded/bent
    index_bent = not fingers["index"]

    # Thumb touches middle fingertip
    thumb_touches_middle = is_touching(
        landmarks[4],    # thumb tip
        landmarks[12],   # middle tip
        threshold=0.06
    )

    # Thumb touches ring fingertip
    thumb_touches_ring = is_touching(
        landmarks[4],    # thumb tip
        landmarks[16],   # ring tip
        threshold=0.06
    )

    # Pinky should be straight
    pinky_straight = fingers["pinky"]

    if (index_bent and
            thumb_touches_middle and
            thumb_touches_ring and
            pinky_straight):
        return True, "Apan Vayu formed correctly ✓"

    # Give specific feedback
    if not index_bent:
        return False, "Fold your index finger down"
    if not thumb_touches_middle:
        return False, \
            "Touch thumb to middle fingertip"
    if not thumb_touches_ring:
        return False, \
            "Touch thumb to ring fingertip too"
    if not pinky_straight:
        return False, "Keep pinky finger straight"

    return False, "Form Apan Vayu Mudra"


def detect_rudra_mudra(landmarks):
    """
    Rudra Mudra — Very complex:
    Thumb touches BOTH index AND ring fingertips
    Middle and pinky stay straight
    """
    fingers = get_finger_states(landmarks)

    # Thumb touches index fingertip
    thumb_touches_index = is_touching(
        landmarks[4],   # thumb tip
        landmarks[8],   # index tip
        threshold=0.06
    )

    # Thumb touches ring fingertip
    thumb_touches_ring = is_touching(
        landmarks[4],    # thumb tip
        landmarks[16],   # ring tip
        threshold=0.06
    )

    # Middle must be straight
    middle_straight = fingers["middle"]

    # Pinky must be straight
    pinky_straight = fingers["pinky"]

    if (thumb_touches_index and
            thumb_touches_ring and
            middle_straight and
            pinky_straight):
        return True, "Rudra Mudra formed correctly ✓"

    # Specific feedback
    if not thumb_touches_index:
        return False, \
            "Touch thumb to index fingertip"
    if not thumb_touches_ring:
        return False, \
            "Also touch thumb to ring fingertip"
    if not middle_straight:
        return False, "Keep middle finger straight"
    if not pinky_straight:
        return False, "Keep pinky finger straight"

    return False, "Form Rudra Mudra"


def detect_mudra(landmarks_list):
    """
    Main detection function
    landmarks_list: list of hand landmarks
                   can be 1 or 2 hands

    Returns detected mudra name,
    confidence and feedback message
    """
    if not landmarks_list:
        return {
            "mudra": "No Hand",
            "detected": False,
            "confidence": 0,
            "message": "Show your hand to camera",
            "color": (128, 128, 128)
        }

    # Single hand detections
    lm = landmarks_list[0].landmark

    # Try each single hand mudra
    gyan, gyan_msg = detect_gyan_mudra(lm)
    if gyan:
        return {
            "mudra": "Gyan Mudra",
            "detected": True,
            "confidence": 95,
            "message": gyan_msg,
            "color": (0, 255, 0),
            "benefit": "Concentration & Wisdom"
        }

    vayu, vayu_msg = detect_vayu_mudra(lm)
    if vayu:
        return {
            "mudra": "Vayu Mudra",
            "detected": True,
            "confidence": 95,
            "message": vayu_msg,
            "color": (0, 255, 0),
            "benefit": "Calm Mind & Focus"
        }

    apan, apan_msg = detect_apan_vayu_mudra(lm)
    if apan:
        return {
            "mudra": "Apan Vayu Mudra",
            "detected": True,
            "confidence": 95,
            "message": apan_msg,
            "color": (0, 255, 0),
            "benefit": "Mental Stability & Clarity"
        }

    rudra, rudra_msg = detect_rudra_mudra(lm)
    if rudra:
        return {
            "mudra": "Rudra Mudra",
            "detected": True,
            "confidence": 95,
            "message": rudra_msg,
            "color": (0, 255, 0),
            "benefit": "Brain Energy & Concentration"
        }

    # Two hand detection — Hakini
    if len(landmarks_list) >= 2:
        lm_left = landmarks_list[0].landmark
        lm_right = landmarks_list[1].landmark
        hakini, hakini_msg = detect_hakini_mudra(
            lm_left, lm_right
        )
        if hakini:
            return {
                "mudra": "Hakini Mudra",
                "detected": True,
                "confidence": 95,
                "message": hakini_msg,
                "color": (0, 255, 0),
                "benefit": "Brain Power & Memory"
            }

    return {
        "mudra": "Unknown",
        "detected": False,
        "confidence": 0,
        "message": "No mudra detected",
        "color": (0, 165, 255)
    }