from pathlib import Path

import cv2
import numpy as np

# YogaAI palette (BGR)
INK = (22, 20, 18)
PANEL = (36, 32, 28)
PANEL_EDGE = (70, 62, 52)
CREAM = (236, 240, 245)
MUTED = (170, 175, 185)
SAGE = (110, 175, 95)
SAGE_DIM = (70, 120, 65)
AMBER = (45, 165, 230)
CORAL = (85, 95, 220)
GHOST = (210, 155, 70)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

JOINT_PAIRS = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

LANDMARK_TO_JOINT = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    15: "left_wrist",
    16: "right_wrist",
}

JOINT_TO_LANDMARK = {v: k for k, v in LANDMARK_TO_JOINT.items()}

IDEAL_STICK_FIGURES = {
    "pose_1_pranamasana": {
        11: (0.42, 0.28), 12: (0.58, 0.28),
        13: (0.40, 0.42), 14: (0.60, 0.42),
        15: (0.48, 0.50), 16: (0.52, 0.50),
        23: (0.44, 0.55), 24: (0.56, 0.55),
        25: (0.44, 0.72), 26: (0.56, 0.72),
        27: (0.44, 0.90), 28: (0.56, 0.90),
    },
    "pose_2_hasta_uttanasana": {
        11: (0.42, 0.32), 12: (0.58, 0.32),
        13: (0.38, 0.18), 14: (0.62, 0.18),
        15: (0.36, 0.06), 16: (0.64, 0.06),
        23: (0.44, 0.55), 24: (0.56, 0.55),
        25: (0.44, 0.72), 26: (0.56, 0.72),
        27: (0.44, 0.90), 28: (0.56, 0.90),
    },
    "pose_3_hasta_padasana": {
        11: (0.42, 0.45), 12: (0.58, 0.45),
        13: (0.40, 0.62), 14: (0.60, 0.62),
        15: (0.42, 0.82), 16: (0.58, 0.82),
        23: (0.44, 0.55), 24: (0.56, 0.55),
        25: (0.44, 0.72), 26: (0.56, 0.72),
        27: (0.44, 0.90), 28: (0.56, 0.90),
    },
    "pose_4_ashwa_sanchalanasana": {
        11: (0.40, 0.30), 12: (0.55, 0.30),
        13: (0.35, 0.45), 14: (0.60, 0.45),
        15: (0.32, 0.58), 16: (0.62, 0.58),
        23: (0.42, 0.52), 24: (0.55, 0.52),
        25: (0.55, 0.70), 26: (0.42, 0.78),
        27: (0.62, 0.88), 28: (0.30, 0.90),
    },
    "pose_5_dandasana": {
        11: (0.35, 0.42), 12: (0.50, 0.42),
        13: (0.28, 0.50), 14: (0.42, 0.50),
        15: (0.22, 0.58), 16: (0.36, 0.58),
        23: (0.42, 0.52), 24: (0.55, 0.52),
        25: (0.58, 0.55), 26: (0.70, 0.55),
        27: (0.75, 0.58), 28: (0.85, 0.58),
    },
    "pose_6_ashtanga_namaskara": {
        11: (0.38, 0.48), 12: (0.55, 0.48),
        13: (0.32, 0.58), 14: (0.60, 0.58),
        15: (0.28, 0.68), 16: (0.64, 0.68),
        23: (0.40, 0.62), 24: (0.55, 0.62),
        25: (0.42, 0.75), 26: (0.55, 0.75),
        27: (0.42, 0.88), 28: (0.55, 0.88),
    },
    "pose_7_bhujangasana": {
        11: (0.38, 0.45), 12: (0.55, 0.45),
        13: (0.32, 0.55), 14: (0.60, 0.55),
        15: (0.28, 0.65), 16: (0.64, 0.65),
        23: (0.40, 0.62), 24: (0.55, 0.62),
        25: (0.42, 0.72), 26: (0.55, 0.72),
        27: (0.42, 0.88), 28: (0.55, 0.88),
    },
    "pose_8_adho_mukha_svanasana": {
        11: (0.32, 0.55), 12: (0.42, 0.55),
        13: (0.28, 0.65), 14: (0.38, 0.65),
        15: (0.25, 0.75), 16: (0.35, 0.75),
        23: (0.55, 0.45), 24: (0.62, 0.45),
        25: (0.68, 0.58), 26: (0.75, 0.58),
        27: (0.78, 0.78), 28: (0.85, 0.78),
    },
}

POSE_REFERENCE_ALIASES = {
    "pose_9_ashwa_sanchalanasana_2": "pose_4_ashwa_sanchalanasana",
    "pose_10_hasta_padasana_2": "pose_3_hasta_padasana",
    "pose_11_hasta_uttanasana_2": "pose_2_hasta_uttanasana",
    "pose_12_pranamasana_2": "pose_1_pranamasana",
}


def get_pixel_coords(landmark, frame_width, frame_height):
    return (int(landmark.x * frame_width), int(landmark.y * frame_height))


def _blend_rect(frame, x1, y1, x2, y2, color, alpha=0.65):
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def _draw_round_rect(frame, x1, y1, x2, y2, color, thickness=1, radius=12):
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    cv2.line(frame, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(frame, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(frame, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(frame, (x2, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def _put_text(frame, text, org, scale=0.6, color=CREAM, thickness=1, shadow=True):
    if shadow:
        cv2.putText(
            frame, text, (org[0] + 1, org[1] + 1),
            cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thickness + 1, cv2.LINE_AA
        )
    cv2.putText(
        frame, text, org,
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


def _draw_bar(frame, x, y, width, height, ratio, fill_color, track=PANEL_EDGE):
    ratio = max(0.0, min(1.0, ratio))
    cv2.rectangle(frame, (x, y), (x + width, y + height), track, -1)
    fill_w = int(width * ratio)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + height), fill_color, -1)
    _draw_round_rect(frame, x, y, x + width, y + height, MUTED, 1, 4)


def _short_pose_title(pose_display_name):
    if "—" in pose_display_name:
        return pose_display_name.split("—", 1)[1].strip()
    return pose_display_name


def _parse_step(progress):
    try:
        current, total = progress.split("/")
        return int(current), int(total)
    except (ValueError, AttributeError):
        return 1, 12


def load_ideal_pose_images(dataset_root=None):
    if dataset_root is None:
        dataset_root = Path(__file__).resolve().parent / "data" / "yoga82_subset"
    else:
        dataset_root = Path(dataset_root)

    images = {}
    if not dataset_root.exists():
        return images

    for pose_dir in sorted(dataset_root.iterdir()):
        if not pose_dir.is_dir():
            continue
        for image_path in sorted(pose_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            frame = cv2.imread(str(image_path))
            if frame is not None:
                images[pose_dir.name] = frame
                break
    return images


def _resolve_pose_key(pose_name):
    return POSE_REFERENCE_ALIASES.get(pose_name, pose_name)


def _draw_stick_figure(canvas, layout, color=GHOST, thickness=2):
    h, w = canvas.shape[:2]
    points = {idx: (int(xy[0] * w), int(xy[1] * h)) for idx, xy in layout.items()}
    for idx_a, idx_b in JOINT_PAIRS:
        if idx_a in points and idx_b in points:
            cv2.line(canvas, points[idx_a], points[idx_b], color, thickness, cv2.LINE_AA)
    for point in points.values():
        cv2.circle(canvas, point, 5, color, -1, cv2.LINE_AA)
    return canvas


def build_ideal_pose_panel(pose_name, ideal_pose_images, panel_size=(280, 360)):
    panel_w, panel_h = panel_size
    panel = np.full((panel_h, panel_w, 3), PANEL, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w, 44), INK, -1)
    cv2.rectangle(panel, (0, 42), (panel_w, 44), AMBER, -1)
    _put_text(panel, "IDEAL POSE", (14, 28), 0.65, AMBER, 2)

    pose_key = _resolve_pose_key(pose_name)
    content_top = 54
    content_h = panel_h - content_top - 14
    content = panel[content_top:content_top + content_h, 10:panel_w - 10]
    ch, cw = content.shape[:2]
    content[:] = (28, 26, 24)

    ref_image = ideal_pose_images.get(pose_name)
    if ref_image is None:
        ref_image = ideal_pose_images.get(pose_key)
    if ref_image is not None:
        ih, iw = ref_image.shape[:2]
        scale = min(cw / iw, ch / ih) * 0.92
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(ref_image, (nw, nh))
        x0 = (cw - nw) // 2
        y0 = (ch - nh) // 2
        content[y0:y0 + nh, x0:x0 + nw] = resized
    else:
        layout = IDEAL_STICK_FIGURES.get(pose_key) or IDEAL_STICK_FIGURES["pose_1_pranamasana"]
        _draw_stick_figure(content, layout, color=GHOST, thickness=3)
        _put_text(content, "Form guide", (12, ch - 14), 0.5, MUTED, 1)

    _draw_round_rect(panel, 0, 0, panel_w - 1, panel_h - 1, PANEL_EDGE, 2, 14)
    return panel


def draw_ideal_pose_reference(frame, pose_name, ideal_pose_images):
    if not pose_name:
        return frame

    h, w = frame.shape[:2]
    panel_w = max(240, int(w * 0.20))
    panel_h = max(300, int(h * 0.48))
    panel = build_ideal_pose_panel(
        pose_name, ideal_pose_images, panel_size=(panel_w, panel_h)
    )

    x0 = w - panel_w - 20
    y0 = 150
    y1 = min(h - 90, y0 + panel_h)
    x1 = min(w - 20, x0 + panel_w)
    ph, pw = y1 - y0, x1 - x0
    if ph <= 0 or pw <= 0:
        return frame

    shadow = frame.copy()
    cv2.rectangle(shadow, (x0 + 6, y0 + 8), (x1 + 6, y1 + 8), BLACK, -1)
    cv2.addWeighted(shadow, 0.35, frame, 0.65, 0, frame)

    roi = frame[y0:y1, x0:x1]
    blended = cv2.addWeighted(panel[:ph, :pw], 0.92, roi, 0.08, 0)
    frame[y0:y1, x0:x1] = blended
    return frame


def draw_user_skeleton(frame, landmarks, validation_result):
    h, w = frame.shape[:2]
    for idx_a, idx_b in JOINT_PAIRS:
        point_a = get_pixel_coords(landmarks[idx_a], w, h)
        point_b = get_pixel_coords(landmarks[idx_b], w, h)
        cv2.line(frame, point_a, point_b, CREAM, 3, cv2.LINE_AA)
        cv2.line(frame, point_a, point_b, (60, 60, 60), 1, cv2.LINE_AA)

    for landmark_idx, joint_name in LANDMARK_TO_JOINT.items():
        point = get_pixel_coords(landmarks[landmark_idx], w, h)
        if validation_result and joint_name in validation_result["joints"]:
            status = validation_result["joints"][joint_name]["status"]
            color = SAGE if status == "correct" else CORAL
        else:
            color = CREAM
        cv2.circle(frame, point, 9, color, -1, cv2.LINE_AA)
        cv2.circle(frame, point, 9, INK, 1, cv2.LINE_AA)
    return frame


def draw_ghost_skeleton(frame, landmarks, ideal_poses, pose_name):
    if not pose_name or not landmarks:
        return frame

    h, w = frame.shape[:2]
    pose_key = _resolve_pose_key(pose_name)
    layout = IDEAL_STICK_FIGURES.get(pose_key)
    if not layout:
        return frame

    left_hip = get_pixel_coords(landmarks[23], w, h)
    right_hip = get_pixel_coords(landmarks[24], w, h)
    left_shoulder = get_pixel_coords(landmarks[11], w, h)
    right_shoulder = get_pixel_coords(landmarks[12], w, h)

    user_hip = (
        (left_hip[0] + right_hip[0]) // 2,
        (left_hip[1] + right_hip[1]) // 2,
    )
    user_shoulder_width = max(40, abs(right_shoulder[0] - left_shoulder[0]))
    layout_hip = (
        (layout[23][0] + layout[24][0]) / 2,
        (layout[23][1] + layout[24][1]) / 2,
    )
    layout_shoulder_width = abs(layout[12][0] - layout[11][0])
    scale = user_shoulder_width / max(layout_shoulder_width, 0.05)

    overlay = frame.copy()
    points = {}
    for idx, (nx, ny) in layout.items():
        px = int(user_hip[0] + (nx - layout_hip[0]) * scale)
        py = int(user_hip[1] + (ny - layout_hip[1]) * scale)
        points[idx] = (px, py)

    for idx_a, idx_b in JOINT_PAIRS:
        if idx_a in points and idx_b in points:
            cv2.line(overlay, points[idx_a], points[idx_b], GHOST, 3, cv2.LINE_AA)
    for point in points.values():
        cv2.circle(overlay, point, 6, GHOST, -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
    return frame


def draw_direction_arrows(frame, landmarks, corrections):
    h, w = frame.shape[:2]
    direction_vectors = {
        "up": (0, -42),
        "down": (0, 42),
        "extend": (42, 0),
        "bend": (-42, 0),
        "adjust": (0, -32),
    }
    for correction in corrections:
        joint_name = correction["joint"]
        direction = correction["direction"]
        if joint_name not in JOINT_TO_LANDMARK:
            continue
        landmark_idx = JOINT_TO_LANDMARK[joint_name]
        start = get_pixel_coords(landmarks[landmark_idx], w, h)
        dx, dy = direction_vectors.get(direction, (0, -32))
        end = (start[0] + dx, start[1] + dy)
        cv2.arrowedLine(frame, start, end, AMBER, 3, tipLength=0.35, line_type=cv2.LINE_AA)
    return frame


def draw_hud(
    frame,
    pose_display_name,
    pose_score,
    progress,
    corrections,
    frames_to_pass,
    frames_passed,
    pass_threshold=80,
):
    h, w = frame.shape[:2]
    step, total = _parse_step(progress)
    title = _short_pose_title(pose_display_name)

    _blend_rect(frame, 0, 0, w, 112, INK, 0.72)
    cv2.line(frame, (0, 112), (w, 112), AMBER, 2)

    _put_text(frame, "YogaAI", (24, 38), 0.95, AMBER, 2)
    _put_text(frame, title[:42], (24, 78), 0.72, CREAM, 2)
    _put_text(frame, f"Step {step} of {total}", (24, 102), 0.48, MUTED, 1)

    meter_w = min(340, max(180, w // 3))
    meter_x = (w - meter_w) // 2
    score_color = SAGE if pose_score >= pass_threshold else (
        AMBER if pose_score >= pass_threshold * 0.6 else CORAL
    )
    _put_text(frame, f"MATCH  {pose_score}%", (meter_x, 34), 0.48, MUTED, 1)
    _draw_bar(frame, meter_x, 42, meter_w, 14, pose_score / 100.0, score_color)

    hold_ratio = frames_passed / max(frames_to_pass, 1)
    hold_label = "HOLD" if hold_ratio < 1 else "LOCKED"
    hold_color = AMBER if hold_ratio < 1 else SAGE
    _put_text(frame, hold_label, (meter_x, 78), 0.45, hold_color, 1)
    _draw_bar(frame, meter_x, 84, meter_w, 10, hold_ratio, hold_color)

    dots_x = w - 28 - (total * 16)
    for i in range(total):
        cx = dots_x + i * 16
        cy = 48
        if i + 1 < step:
            cv2.circle(frame, (cx, cy), 5, SAGE, -1, cv2.LINE_AA)
        elif i + 1 == step:
            cv2.circle(frame, (cx, cy), 7, AMBER, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 7, CREAM, 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, cy), 4, PANEL_EDGE, -1, cv2.LINE_AA)

    toast_w = min(w - 40, 640)
    toast_h = 68
    tx1, ty1 = 20, h - toast_h - 18
    tx2, ty2 = 20 + toast_w, h - 18
    _blend_rect(frame, tx1, ty1, tx2, ty2, INK, 0.78)
    _draw_round_rect(frame, tx1, ty1, tx2, ty2, PANEL_EDGE, 1, 10)

    if corrections:
        _put_text(frame, "ADJUST", (tx1 + 18, ty1 + 26), 0.5, CORAL, 1)
        _put_text(frame, corrections[0]["message"], (tx1 + 18, ty1 + 50), 0.58, CREAM, 1)
    else:
        tip = (
            "Step into frame to begin"
            if pose_score == 0 and frames_passed == 0
            else "Hold steady — you're aligned with the ideal pose"
        )
        label = "READY" if pose_score == 0 else "NICE FORM"
        _put_text(frame, label, (tx1 + 18, ty1 + 26), 0.5, SAGE, 1)
        _put_text(frame, tip, (tx1 + 18, ty1 + 50), 0.55, CREAM, 1)

    return frame


def draw_session_complete(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), INK, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    card_w, card_h = min(560, w - 80), 220
    x1 = (w - card_w) // 2
    y1 = (h - card_h) // 2
    x2, y2 = x1 + card_w, y1 + card_h

    _blend_rect(frame, x1, y1, x2, y2, PANEL, 0.92)
    _draw_round_rect(frame, x1, y1, x2, y2, AMBER, 2, 16)
    cv2.line(frame, (x1 + 24, y1 + 56), (x2 - 24, y1 + 56), PANEL_EDGE, 1)

    _put_text(frame, "YogaAI", (x1 + 36, y1 + 40), 0.7, AMBER, 2)
    _put_text(frame, "Surya Namaskar complete", (x1 + 36, y1 + 100), 0.85, CREAM, 2)
    _put_text(
        frame,
        "Press R to restart   ·   Q or X to quit",
        (x1 + 36, y1 + 150),
        0.55,
        MUTED,
        1,
    )
    return frame
