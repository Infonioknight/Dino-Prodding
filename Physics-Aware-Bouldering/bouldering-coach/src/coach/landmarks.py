"""MediaPipe Pose landmark indices and the segment definitions shared by
ingest.py (calibration), pose.py (bone-length rejection) and physics.py
(de Leva segment masses). Keeping this in one place means the three stages
can't silently disagree on what "upper arm" means.
"""

NAMES = {
    0: "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}
IDX = {v: k for k, v in NAMES.items()}

# Bilateral calibration segments: (name, point_a, point_b). Median length in
# pixels over the calibration window becomes the ground truth for M1's
# limb-length table, reused by M3's bone-length rejection test.
CALIBRATION_SEGMENTS = [
    ("upper_arm_l", "left_shoulder", "left_elbow"),
    ("upper_arm_r", "right_shoulder", "right_elbow"),
    ("lower_arm_l", "left_elbow", "left_wrist"),
    ("lower_arm_r", "right_elbow", "right_wrist"),
    ("upper_leg_l", "left_hip", "left_knee"),
    ("upper_leg_r", "right_hip", "right_knee"),
    ("lower_leg_l", "left_knee", "left_ankle"),
    ("lower_leg_r", "right_knee", "right_ankle"),
    ("shoulder_width", "left_shoulder", "right_shoulder"),
]
# Torso height uses shoulder/hip midpoints rather than a single landmark
# pair, so it's computed separately in ingest.py.

# Hand/foot landmarks used for contact detection in pose.py.
CONTACT_LANDMARKS = {
    "left_hand": "left_wrist",
    "right_hand": "right_wrist",
    "left_foot": "left_foot_index",
    "right_foot": "right_foot_index",
}

VISIBILITY_THRESHOLD = 0.5
