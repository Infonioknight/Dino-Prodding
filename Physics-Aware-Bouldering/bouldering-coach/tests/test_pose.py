import numpy as np

from coach.landmarks import IDX
from coach.pose import bone_length_rejection, visibility_gate


def test_visibility_gate_drops_low_confidence_landmarks():
    lm = np.zeros((3, 33, 4))
    lm[..., 3] = 1.0
    lm[1, IDX["left_wrist"], 3] = 0.1  # below threshold
    out = visibility_gate(lm)
    assert np.isnan(out[1, IDX["left_wrist"], 0])
    assert not np.isnan(out[0, IDX["left_wrist"], 0])


def test_bone_length_rejection_flags_outlier_segment():
    lm = np.zeros((2, 33, 4))
    lm[..., 3] = 1.0
    lm[:, IDX["left_shoulder"], :2] = [0, 0]
    lm[:, IDX["left_elbow"], :2] = [0, 30]  # matches calibration in frame 0
    lm[1, IDX["left_elbow"], :2] = [0, 100]  # jumps far beyond tolerance in frame 1

    calibration = {"upper_arm_l": 30.0}
    out = bone_length_rejection(lm, calibration)

    assert not np.isnan(out[0, IDX["left_elbow"], 0])
    assert np.isnan(out[1, IDX["left_elbow"], 0])
