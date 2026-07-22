class PoseSequenceManager:
    """
    Manages which pose the user should be doing
    Tracks progress through all 12 Surya Namaskar poses
    """

    POSE_SEQUENCE = [
        "pose_1_pranamasana",
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
        "pose_12_pranamasana_2"
    ]

    POSE_DISPLAY_NAMES = {
        "pose_1_pranamasana": "Pose 1 — Pranamasana (Prayer Pose)",
        "pose_2_hasta_uttanasana": "Pose 2 — Hasta Uttanasana (Raised Arms)",
        "pose_3_hasta_padasana": "Pose 3 — Hasta Padasana (Forward Bend)",
        "pose_4_ashwa_sanchalanasana": "Pose 4 — Ashwa Sanchalanasana (Equestrian)",
        "pose_5_dandasana": "Pose 5 — Dandasana (Stick Pose)",
        "pose_6_ashtanga_namaskara": "Pose 6 — Ashtanga Namaskara (Eight Limbs)",
        "pose_7_bhujangasana": "Pose 7 — Bhujangasana (Cobra Pose)",
        "pose_8_adho_mukha_svanasana": "Pose 8 — Adho Mukha Svanasana (Downward Dog)",
        "pose_9_ashwa_sanchalanasana_2": "Pose 9 — Ashwa Sanchalanasana (Equestrian 2)",
        "pose_10_hasta_padasana_2": "Pose 10 — Hasta Padasana (Forward Bend 2)",
        "pose_11_hasta_uttanasana_2": "Pose 11 — Hasta Uttanasana (Raised Arms 2)",
        "pose_12_pranamasana_2": "Pose 12 — Pranamasana (Prayer Pose 2)"
    }

    PASS_THRESHOLD = 80
    FRAMES_TO_PASS = 30

    _LOOSE_HOLD_POSES = [
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
    ]
    POSE_PASS_THRESHOLDS = {pose: 60 for pose in _LOOSE_HOLD_POSES}
    POSE_FRAMES_TO_PASS = {pose: 15 for pose in _LOOSE_HOLD_POSES}

    def __init__(self):
        self.current_index = 0
        self.frames_passed = 0
        self.session_complete = False

    def get_current_pose(self):
        if self.session_complete:
            return None
        return self.POSE_SEQUENCE[self.current_index]

    def get_current_display_name(self):
        pose = self.get_current_pose()
        if pose:
            return self.POSE_DISPLAY_NAMES[pose]
        return "Session Complete!"

    def get_progress(self):
        return f"{self.current_index + 1}/12"

    def get_pass_threshold(self):
        pose = self.get_current_pose()
        return self.POSE_PASS_THRESHOLDS.get(pose, self.PASS_THRESHOLD)

    def get_frames_to_pass(self):
        pose = self.get_current_pose()
        return self.POSE_FRAMES_TO_PASS.get(pose, self.FRAMES_TO_PASS)

    def update(self, pose_score):
        if self.session_complete:
            return

        pass_threshold = self.get_pass_threshold()
        frames_to_pass = self.get_frames_to_pass()

        if pose_score >= pass_threshold:
            self.frames_passed += 1
        else:
            self.frames_passed = 0

        if self.frames_passed >= frames_to_pass:
            self.next_pose()

    def next_pose(self):
        self.frames_passed = 0
        self.current_index += 1

        if self.current_index >= len(self.POSE_SEQUENCE):
            self.session_complete = True
            self.current_index = len(self.POSE_SEQUENCE) - 1
            print("Surya Namaskar Complete!")

    def reset(self):
        self.current_index = 0
        self.frames_passed = 0
        self.session_complete = False
