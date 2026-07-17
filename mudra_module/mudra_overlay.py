import cv2

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)


def draw_mudra_hud(frame, mudra_result):
    """
    Draws mudra detection HUD on frame
    Shows mudra name, confidence,
    benefit and feedback
    """
    h, w = frame.shape[:2]

    # Background box at bottom right
    box_x = w - 380
    box_y = h - 160
    cv2.rectangle(
        frame,
        (box_x, box_y),
        (w - 10, h - 10),
        BLACK, -1
    )
    cv2.rectangle(
        frame,
        (box_x, box_y),
        (w - 10, h - 10),
        mudra_result['color'], 2
    )

    # Mudra title
    cv2.putText(
        frame,
        f"Mudra: {mudra_result['mudra']}",
        (box_x + 10, box_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, WHITE, 2
    )

    # Confidence
    if mudra_result['detected']:
        cv2.putText(
            frame,
            f"Confidence: {mudra_result['confidence']}%",
            (box_x + 10, box_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, GREEN, 2
        )

        # Benefit
        cv2.putText(
            frame,
            f"Benefit: {mudra_result.get('benefit', '')}",
            (box_x + 10, box_y + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, YELLOW, 1
        )
    else:
        # Show feedback message
        cv2.putText(
            frame,
            mudra_result['message'],
            (box_x + 10, box_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, ORANGE, 1
        )

    # Status indicator
    status = "DETECTED" \
        if mudra_result['detected'] else "NOT DETECTED"
    color = GREEN if mudra_result['detected'] else RED
    cv2.putText(
        frame,
        status,
        (box_x + 10, box_y + 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, color, 2
    )

    return frame


def draw_hand_landmarks(frame, results, mp_hands,
                         mp_drawing):
    """
    Draws hand landmarks on frame
    """
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
    return frame