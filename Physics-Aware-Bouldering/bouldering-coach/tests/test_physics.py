import numpy as np

from coach.physics import com_to_hull_signed_distance, compute_com, segment_moves
from coach.landmarks import IDX


def _synthetic_landmarks(n_frames: int = 5) -> np.ndarray:
    """A standing figure, roughly centered at (100, 200), all landmarks
    visible. Enough to exercise compute_com's mass-weighted sum."""
    lm = np.zeros((n_frames, 33, 4))
    lm[..., 3] = 1.0  # fully visible

    layout = {
        "nose": (100, 50),
        "left_shoulder": (85, 100), "right_shoulder": (115, 100),
        "left_elbow": (75, 140), "right_elbow": (125, 140),
        "left_wrist": (70, 180), "right_wrist": (130, 180),
        "left_hip": (90, 200), "right_hip": (110, 200),
        "left_knee": (88, 260), "right_knee": (112, 260),
        "left_ankle": (86, 320), "right_ankle": (114, 320),
        "left_heel": (84, 330), "right_heel": (116, 330),
        "left_foot_index": (80, 335), "right_foot_index": (120, 335),
    }
    for name, (x, y) in layout.items():
        lm[:, IDX[name], 0] = x
        lm[:, IDX[name], 1] = y
    return lm


def test_compute_com_within_body_bounds():
    lm = _synthetic_landmarks()
    com = compute_com(lm)
    assert not np.isnan(com).any()
    # CoM of a standing figure should sit between shoulders and hips-ish,
    # well within the horizontal extent of the body.
    assert 70 < com[0, 0] < 130
    assert 90 < com[0, 1] < 260


def test_com_to_hull_signed_distance_inside_is_negative():
    triangle = np.array([[0, 0], [10, 0], [5, 10]])
    center = np.array([5.0, 3.0])
    d = com_to_hull_signed_distance(center, triangle)
    assert d < 0


def test_com_to_hull_signed_distance_outside_is_positive():
    triangle = np.array([[0, 0], [10, 0], [5, 10]])
    far_point = np.array([100.0, 100.0])
    d = com_to_hull_signed_distance(far_point, triangle)
    assert d > 0


def test_segment_moves_merges_close_boundaries():
    contacts = [
        {"limb": "left_hand", "hold_id": 1, "t_start": 1.0, "t_end": 2.0},
        {"limb": "right_hand", "hold_id": 2, "t_start": 1.2, "t_end": 2.5},  # <0.5s after prior start
        {"limb": "left_foot", "hold_id": 3, "t_start": 5.0, "t_end": 6.0},
    ]
    moves = segment_moves(contacts, fps=12.0, n_frames=120)
    boundary_starts = sorted({round(m.t_start, 3) for m in moves})
    assert 1.2 not in boundary_starts  # merged into the 1.0 boundary
