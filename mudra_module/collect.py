import cv2
import mediapipe as mp
import os
import time


# ============================================================
# CONFIGURATION
# ============================================================

# Mudras to collect
MUDRAS = [
    "gyan_mudra",
    "vayu_mudra",
    "apan_vayu_mudra",
    "rudra_mudra",
    "hakini_mudra"
]

# Two-handed mudras — need countdown
TWO_HAND_MUDRAS = [
    "hakini_mudra"
]

# Number of images required per mudra
IMAGES_PER_MUDRA = 200

# Image size
IMG_SIZE = 224

# Countdown for two-hand mudras
COUNTDOWN_SECONDS = 5


# ============================================================
# SAVE PATH
# ============================================================

BASE_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "mudra_images"
)


# ============================================================
# MUDRA INSTRUCTIONS
# ============================================================

INSTRUCTIONS = {

    "gyan_mudra":
        "Touch INDEX tip to THUMB tip. Other 3 fingers straight.",

    "vayu_mudra":
        "Fold INDEX to BASE of thumb. Others straight.",

    "apan_vayu_mudra":
        "Fold INDEX down. THUMB touches MIDDLE + RING. PINKY straight.",

    "rudra_mudra":
        "THUMB touches INDEX tip + RING tip. MIDDLE + PINKY straight.",

    "hakini_mudra":
        "BOTH HANDS - All 5 fingertips touch in dome. Press SPACE then form mudra!"
}


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# DRAW USER INTERFACE
# ============================================================

def draw_ui(
    frame,
    mudra_name,
    img_count,
    collecting,
    countdown_val
):

    h, w = frame.shape[:2]

    # --------------------------------------------------------
    # Top black bar
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (0, 0),
        (w, 110),
        (0, 0, 0),
        -1
    )

    # --------------------------------------------------------
    # Mudra name
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Mudra: {mudra_name.replace('_', ' ').upper()}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Image count
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Images: {img_count}/{IMAGES_PER_MUDRA}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # --------------------------------------------------------
    # Progress bar
    # --------------------------------------------------------

    bar_width = int(
        (img_count / IMAGES_PER_MUDRA) * (w - 20)
    )

    cv2.rectangle(
        frame,
        (10, 82),
        (w - 10, 100),
        (50, 50, 50),
        -1
    )

    cv2.rectangle(
        frame,
        (10, 82),
        (10 + bar_width, 100),
        (0, 255, 0),
        -1
    )

    # --------------------------------------------------------
    # Instruction
    # --------------------------------------------------------

    instruction = INSTRUCTIONS[mudra_name]

    cv2.putText(
        frame,
        instruction,
        (10, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # --------------------------------------------------------
    # Countdown
    # --------------------------------------------------------

    if countdown_val > 0:

        cv2.putText(
            frame,
            str(countdown_val),
            (w // 2 - 30, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            (0, 0, 255),
            8
        )

        cv2.putText(
            frame,
            "FORM THE MUDRA NOW!",
            (w // 2 - 150, h // 2 + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # --------------------------------------------------------
    # Collecting status
    # --------------------------------------------------------

    elif collecting:

        cv2.putText(
            frame,
            "COLLECTING... Hold steady!",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # --------------------------------------------------------
    # Waiting status
    # --------------------------------------------------------

    else:

        cv2.putText(
            frame,
            "Form mudra then press SPACE",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2
        )

    return frame


# ============================================================
# COUNT EXISTING IMAGES
# ============================================================

def get_existing_images(save_path, mudra_name):

    """
    Counts existing JPG images for the current mudra.

    This allows the program to resume instead of starting
    from zero every time.
    """

    if not os.path.exists(save_path):
        return 0

    images = []

    for filename in os.listdir(save_path):

        if filename.lower().endswith(".jpg"):
            images.append(filename)

    return len(images)


# ============================================================
# MAIN COLLECTION FUNCTION
# ============================================================

def collect_images():

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("❌ ERROR: Could not open camera.")
        return

    # --------------------------------------------------------
    # Create base directory
    # --------------------------------------------------------

    os.makedirs(
        BASE_PATH,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Startup information
    # --------------------------------------------------------

    print("=" * 60)
    print("YogaAI — Mudra Image Collection")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Images per mudra: {IMAGES_PER_MUDRA}")
    print(
        f"Total images required: "
        f"{IMAGES_PER_MUDRA * len(MUDRAS)}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # MediaPipe Hands
    # --------------------------------------------------------

    with mp_hands.Hands(

        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=2

    ) as hands:

        # ====================================================
        # LOOP THROUGH MUDRAS
        # ====================================================

        for mudra_name in MUDRAS:

            # ------------------------------------------------
            # Create folder
            # ------------------------------------------------

            save_path = os.path.join(
                BASE_PATH,
                mudra_name
            )

            os.makedirs(
                save_path,
                exist_ok=True
            )

            # ------------------------------------------------
            # CHECK EXISTING IMAGES
            # ------------------------------------------------

            existing_images = get_existing_images(
                save_path,
                mudra_name
            )

            # ------------------------------------------------
            # ALREADY COMPLETE
            # ------------------------------------------------

            if existing_images >= IMAGES_PER_MUDRA:

                print()
                print("=" * 60)
                print(
                    f"✅ {mudra_name.upper()} "
                    f"already complete!"
                )
                print(
                    f"   Existing images: "
                    f"{existing_images}/{IMAGES_PER_MUDRA}"
                )
                print("   Skipping to next mudra...")
                print("=" * 60)

                continue

            # ------------------------------------------------
            # RESUME PARTIALLY COMPLETED MUDRA
            # ------------------------------------------------

            if existing_images > 0:

                img_count = existing_images

                print()
                print("=" * 60)
                print(
                    f"⚠️  RESUMING: "
                    f"{mudra_name.upper()}"
                )
                print(
                    f"   Existing images: "
                    f"{existing_images}/{IMAGES_PER_MUDRA}"
                )
                print(
                    f"   Remaining images: "
                    f"{IMAGES_PER_MUDRA - existing_images}"
                )
                print("=" * 60)

            # ------------------------------------------------
            # START NEW MUDRA
            # ------------------------------------------------

            else:

                img_count = 0

                print()
                print("=" * 60)
                print(
                    f"📌 Starting: "
                    f"{mudra_name.upper()}"
                )
                print("=" * 60)

            # ------------------------------------------------
            # Variables
            # ------------------------------------------------

            collecting = False

            countdown_val = 0

            countdown_start = None

            is_two_hand = (
                mudra_name in TWO_HAND_MUDRAS
            )

            # ------------------------------------------------
            # Instructions in terminal
            # ------------------------------------------------

            print(
                f"📋 How: "
                f"{INSTRUCTIONS[mudra_name]}"
            )

            if is_two_hand:

                print(
                    "⚠️ TWO HAND MUDRA"
                )

                print(
                    "Press SPACE, then form the mudra "
                    "during the countdown."
                )

            else:

                print(
                    "Press SPACE to start collecting."
                )

            print(
                "Press Q to quit."
            )

            # =================================================
            # COLLECTION LOOP
            # =================================================

            while img_count < IMAGES_PER_MUDRA:

                # ------------------------------------------------
                # Read camera
                # ------------------------------------------------

                ret, frame = cap.read()

                if not ret:

                    print(
                        "❌ Could not read frame from camera."
                    )

                    break

                # ------------------------------------------------
                # Mirror image
                # ------------------------------------------------

                frame = cv2.flip(
                    frame,
                    1
                )

                # ------------------------------------------------
                # CLEAN FRAME
                #
                # IMPORTANT:
                # This copy is taken BEFORE landmarks/UI
                # are drawn.
                # ------------------------------------------------

                clean = frame.copy()

                # ------------------------------------------------
                # Convert to RGB for MediaPipe
                # ------------------------------------------------

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # ------------------------------------------------
                # Detect hands
                # ------------------------------------------------

                results = hands.process(rgb)

                # ------------------------------------------------
                # Draw hand landmarks
                # ------------------------------------------------

                if results.multi_hand_landmarks:

                    for hand_lm in results.multi_hand_landmarks:

                        mp_drawing.draw_landmarks(
                            frame,
                            hand_lm,
                            mp_hands.HAND_CONNECTIONS
                        )

                # =================================================
                # HANDLE COUNTDOWN
                # =================================================

                if countdown_start is not None:

                    elapsed = (
                        time.time()
                        - countdown_start
                    )

                    remaining = (
                        COUNTDOWN_SECONDS
                        - int(elapsed)
                    )

                    if remaining > 0:

                        countdown_val = remaining

                    else:

                        # Countdown finished
                        countdown_val = 0

                        countdown_start = None

                        collecting = True

                        print(
                            "🟢 Collecting started!"
                        )

                # =================================================
                # DRAW UI
                # =================================================

                frame = draw_ui(

                    frame,
                    mudra_name,
                    img_count,
                    collecting,
                    countdown_val

                )

                # =================================================
                # SAVE IMAGE
                # =================================================

                if (
                    collecting
                    and
                    img_count < IMAGES_PER_MUDRA
                ):

                    # --------------------------------------------
                    # Resize clean frame
                    # --------------------------------------------

                    clean_frame = cv2.resize(
                        clean,
                        (IMG_SIZE, IMG_SIZE)
                    )

                    # --------------------------------------------
                    # Image filename
                    #
                    # Example:
                    # gyan_mudra_0000.jpg
                    # gyan_mudra_0001.jpg
                    # ...
                    # --------------------------------------------

                    img_path = os.path.join(

                        save_path,

                        f"{mudra_name}_"
                        f"{img_count:04d}.jpg"

                    )

                    # --------------------------------------------
                    # Save
                    # --------------------------------------------

                    success = cv2.imwrite(
                        img_path,
                        clean_frame
                    )

                    if success:

                        img_count += 1

                    else:

                        print(
                            f"❌ Failed to save: "
                            f"{img_path}"
                        )

                    # Small delay
                    time.sleep(0.05)

                # =================================================
                # CHECK COMPLETION
                # =================================================

                if img_count >= IMAGES_PER_MUDRA:

                    print()
                    print(
                        f"✅ {mudra_name.upper()} "
                        f"— {img_count} images saved!"
                    )

                    # --------------------------------------------
                    # Completion box
                    # --------------------------------------------

                    cv2.rectangle(

                        frame,

                        (50, 200),
                        (590, 320),

                        (0, 100, 0),

                        -1

                    )

                    cv2.putText(

                        frame,

                        f"DONE! "
                        f"{img_count} images saved",

                        (70, 255),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.9,

                        (255, 255, 255),

                        2

                    )

                    cv2.putText(

                        frame,

                        "Moving to next mudra...",

                        (100, 295),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.7,

                        (200, 255, 200),

                        1

                    )

                    # --------------------------------------------
                    # Show completion
                    # --------------------------------------------

                    cv2.imshow(
                        "YogaAI Mudra Collector",
                        frame
                    )

                    cv2.waitKey(2000)

                    break

                # =================================================
                # SHOW CAMERA
                # =================================================

                cv2.imshow(
                    "YogaAI Mudra Collector",
                    frame
                )

                # =================================================
                # KEYBOARD INPUT
                # =================================================

                key = cv2.waitKey(1) & 0xFF

                # ------------------------------------------------
                # SPACE
                # ------------------------------------------------

                if key == ord(" "):

                    if is_two_hand:

                        # ----------------------------------------
                        # Start countdown
                        # ----------------------------------------

                        countdown_start = time.time()

                        countdown_val = COUNTDOWN_SECONDS

                        collecting = False

                        print(
                            f"⏳ Countdown started!"
                        )

                        print(
                            f"Form mudra in "
                            f"{COUNTDOWN_SECONDS} seconds!"
                        )

                    else:

                        # ----------------------------------------
                        # Start collecting immediately
                        # ----------------------------------------

                        collecting = True

                        print(
                            f"🟢 Collecting "
                            f"{mudra_name}..."
                        )

                # ------------------------------------------------
                # Q — QUIT
                # ------------------------------------------------

                if key == ord("q"):

                    print()
                    print(
                        "🛑 Quit by user."
                    )

                    print(
                        f"Progress saved for "
                        f"{mudra_name}: "
                        f"{img_count}/{IMAGES_PER_MUDRA}"
                    )

                    cap.release()

                    cv2.destroyAllWindows()

                    return

            # =====================================================
            # END OF CURRENT MUDRA
            # =====================================================

            print(
                f"Finished processing "
                f"{mudra_name}."
            )

    # ============================================================
    # CLEANUP
    # ============================================================

    cap.release()

    cv2.destroyAllWindows()

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print()
    print("=" * 60)
    print("🎉 ALL MUDRA IMAGES COLLECTED!")
    print("=" * 60)

    print(
        f"Expected total images: "
        f"{IMAGES_PER_MUDRA * len(MUDRAS)}"
    )

    print(
        f"Saved to: {BASE_PATH}"
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Run preprocess_mudra.py"
    )

    print("=" * 60)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    collect_images()
