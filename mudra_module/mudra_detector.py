import cv2
import mediapipe as mp
from mudra_rules import detect_mudra
from mudra_overlay import draw_mudra_hud, \
    draw_hand_landmarks

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def run_mudra_detection():
    """
    Standalone mudra detection module
    Runs independently from main yoga system
    Press Q to quit
    """
    cap = cv2.VideoCapture(0)

    print("=" * 50)
    print("YogaAI — Mudra Detection Module")
    print("=" * 50)
    print("Mudras to try:")
    print("1. Gyan Mudra — index tip to thumb tip")
    print("2. Vayu Mudra — fold index to thumb base")
    print("3. Apan Vayu — fold index, thumb to"
          " middle+ring")
    print("4. Rudra Mudra — thumb to index+ring")
    print("5. Hakini Mudra — both hands fingertips"
          " touch")
    print("Press Q to quit")
    print("=" * 50)

    with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=2
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for mirror view
            frame = cv2.flip(frame, 1)

            # Convert to RGB
            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )
            results = hands.process(rgb)

            # Draw hand landmarks
            frame = draw_hand_landmarks(
                frame, results,
                mp_hands, mp_drawing
            )

            # Detect mudra
            if results.multi_hand_landmarks:
                mudra_result = detect_mudra(
                    results.multi_hand_landmarks
                )
            else:
                mudra_result = {
                    "mudra": "No Hand",
                    "detected": False,
                    "confidence": 0,
                    "message": "Show hand to camera",
                    "color": (128, 128, 128)
                }

            # Draw HUD
            frame = draw_mudra_hud(
                frame, mudra_result
            )

            # Show mudra guide on top left
            cv2.putText(
                frame,
                "YogaAI Mudra Detection",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2
            )

            cv2.imshow(
                'YogaAI — Mudra Detection', frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Mudra detection stopped")


if __name__ == "__main__":
    run_mudra_detection()