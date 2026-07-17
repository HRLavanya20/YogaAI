import numpy as np


def extract_features(landmarks):
    """
    Extracts 21 features from hand landmarks
    For future CNN/ML training use
    """
    features = []

    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z])

    return features