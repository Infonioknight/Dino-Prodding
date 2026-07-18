"""M4: physics features from cleaned 2D landmarks + contacts.

Everything here is honestly computable from a monocular 2D skeleton. No
force estimation (statically indeterminate from video — would be fake
precision). All Python logic is physics; judgment belongs to coach_llm.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, distance
from scipy.spatial.distance import cdist

from coach.landmarks import IDX

MOVE_MERGE_GAP_S = 0.5

# de Leva (1996) segment mass fractions (% body mass) and CoM position as a
# fraction of segment length from the proximal landmark. ~15 rows: 6
# bilateral segments (arm/forearm/hand, thigh/shank/foot) + trunk + head.
# MediaPipe has no neck/ear landmarks we trust, so head is approximated as
# a small point mass at the nose and trunk spans shoulder-mid to hip-mid.
SEGMENT_TABLE = [
    # name, mass_fraction, proximal, distal, com_fraction_from_proximal
    ("upper_arm_l", 0.0271, "left_shoulder", "left_elbow", 0.577),
    ("upper_arm_r", 0.0271, "right_shoulder", "right_elbow", 0.577),
    ("forearm_l", 0.0162, "left_elbow", "left_wrist", 0.457),
    ("forearm_r", 0.0162, "right_elbow", "right_wrist", 0.457),
    ("hand_l", 0.0061, "left_wrist", "left_wrist", 0.5),  # no finger landmark; treat as point mass at wrist
    ("hand_r", 0.0061, "right_wrist", "right_wrist", 0.5),
    ("thigh_l", 0.1416, "left_hip", "left_knee", 0.409),
    ("thigh_r", 0.1416, "right_hip", "right_knee", 0.409),
    ("shank_l", 0.0433, "left_knee", "left_ankle", 0.446),
    ("shank_r", 0.0433, "right_knee", "right_ankle", 0.446),
    ("foot_l", 0.0137, "left_ankle", "left_foot_index", 0.446),
    ("foot_r", 0.0137, "right_ankle", "right_foot_index", 0.446),
    ("head", 0.0826, "nose", "nose", 0.5),  # point mass approximation
]
# Trunk handled separately below (needs shoulder-mid/hip-mid, not single landmarks).
TRUNK_MASS_FRACTION = 0.4346


def _mid(landmarks_px: np.ndarray, name_a: str, name_b: str) -> np.ndarray:
    return (landmarks_px[:, IDX[name_a], :2] + landmarks_px[:, IDX[name_b], :2]) / 2.0


def compute_com(landmarks_px: np.ndarray) -> np.ndarray:
    """Per-frame 2D center of mass in pixel coordinates, shape (T, 2)."""
    t = landmarks_px.shape[0]
    com = np.zeros((t, 2))
    total_mass = 0.0

    for name, mass, prox, dist_name, frac in SEGMENT_TABLE:
        p = landmarks_px[:, IDX[prox], :2]
        d = landmarks_px[:, IDX[dist_name], :2]
        seg_com = p + frac * (d - p)
        valid = ~np.isnan(seg_com).any(axis=1)
        com = np.where(valid[:, None], com + mass * np.nan_to_num(seg_com), com)
        total_mass += mass

    shoulder_mid = _mid(landmarks_px, "left_shoulder", "right_shoulder")
    hip_mid = _mid(landmarks_px, "left_hip", "right_hip")
    trunk_com = hip_mid + 0.5 * (shoulder_mid - hip_mid)
    valid = ~np.isnan(trunk_com).any(axis=1)
    com = np.where(valid[:, None], com + TRUNK_MASS_FRACTION * np.nan_to_num(trunk_com), com)
    total_mass += TRUNK_MASS_FRACTION

    com = com / total_mass
    any_nan_frame = np.isnan(landmarks_px[:, [IDX["left_shoulder"], IDX["right_hip"]], :2]).any(axis=(1, 2))
    com[any_nan_frame] = np.nan
    return com


def base_of_support(hold_centroids: np.ndarray) -> ConvexHull | None:
    """Convex hull of active contact points (hold centroids). None if <3
    non-collinear points (falls back to the points themselves for distance)."""
    if len(hold_centroids) < 3:
        return None
    try:
        return ConvexHull(hold_centroids)
    except Exception:
        return None


def com_to_hull_signed_distance(com_point: np.ndarray, hold_centroids: np.ndarray) -> float:
    """Signed distance from CoM to the base-of-support hull boundary.
    Negative = inside the hull (balanced), positive = outside."""
    if len(hold_centroids) == 0 or np.isnan(com_point).any():
        return float("nan")
    if len(hold_centroids) < 3:
        # Not enough points for a hull — distance to nearest support point,
        # always reported as positive (no "inside" concept with <3 points).
        return float(np.min(cdist([com_point], hold_centroids)))

    hull = base_of_support(hold_centroids)
    if hull is None:
        return float(np.min(cdist([com_point], hold_centroids)))

    path_points = hold_centroids[hull.vertices]
    inside = _point_in_polygon(com_point, path_points)
    edge_dist = _min_dist_to_polygon_edges(com_point, path_points)
    return -edge_dist if inside else edge_dist


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _min_dist_to_polygon_edges(point: np.ndarray, polygon: np.ndarray) -> float:
    n = len(polygon)
    best = np.inf
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        seg = b - a
        seg_len2 = seg @ seg
        if seg_len2 < 1e-9:
            d = np.linalg.norm(point - a)
        else:
            tparam = np.clip(((point - a) @ seg) / seg_len2, 0, 1)
            proj = a + tparam * seg
            d = np.linalg.norm(point - proj)
        best = min(best, d)
    return float(best)


@dataclass
class Kinematics:
    velocity_norm: np.ndarray  # per-frame speed, normalized by shoulder width
    acceleration_norm: np.ndarray
    jerk_norm: np.ndarray


def compute_kinematics(com_px: np.ndarray, fps: float, shoulder_width_px: float) -> Kinematics:
    com_norm = com_px / shoulder_width_px
    velocity = np.gradient(com_norm, 1.0 / fps, axis=0)
    velocity_mag = np.linalg.norm(velocity, axis=1)
    acceleration = np.gradient(velocity, 1.0 / fps, axis=0)
    acceleration_mag = np.linalg.norm(acceleration, axis=1)
    jerk = np.gradient(acceleration, 1.0 / fps, axis=0)
    jerk_mag = np.linalg.norm(jerk, axis=1)
    return Kinematics(velocity_mag, acceleration_mag, jerk_mag)


def torso_wall_angle(
    landmarks_px: np.ndarray,
    calibration_torso_px: float,
    calibration_shoulder_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (angle_deg, confidence). Foreshortening on the
    hip-shoulder quad: apparent torso / apparent shoulder width, normalized
    by the same ratio at calibration (frontal stance) -> arccos gives the
    wall-lean angle. Confidence = agreement with MediaPipe's z delta."""
    shoulder_mid = _mid(landmarks_px, "left_shoulder", "right_shoulder")
    hip_mid = _mid(landmarks_px, "left_hip", "right_hip")
    apparent_torso = np.linalg.norm(shoulder_mid - hip_mid, axis=1)

    shoulder_l = landmarks_px[:, IDX["left_shoulder"], :2]
    shoulder_r = landmarks_px[:, IDX["right_shoulder"], :2]
    apparent_shoulder = np.linalg.norm(shoulder_l - shoulder_r, axis=1)

    calib_ratio = calibration_torso_px / calibration_shoulder_px
    apparent_ratio = apparent_torso / apparent_shoulder
    ratio = apparent_ratio / calib_ratio
    ratio_clamped = np.clip(ratio, 0.0, 1.0)
    angle_deg = np.degrees(np.arccos(ratio_clamped))

    # z-based cross-check: MediaPipe's relative z gives an independent lean
    # estimate; confidence is how well the two agree (1 = perfect agreement).
    z_shoulder = np.mean(landmarks_px[:, [IDX["left_shoulder"], IDX["right_shoulder"]], 2], axis=1)
    z_hip = np.mean(landmarks_px[:, [IDX["left_hip"], IDX["right_hip"]], 2], axis=1)
    z_delta = np.abs(z_shoulder - z_hip)
    with np.errstate(invalid="ignore", divide="ignore"):
        angle_z = np.degrees(np.arctan2(z_delta, apparent_torso))
    disagreement = np.abs(angle_deg - angle_z) / 90.0
    confidence = np.clip(1.0 - disagreement, 0.0, 1.0)
    confidence = np.where(np.isnan(angle_deg) | np.isnan(angle_z), 0.0, confidence)

    return angle_deg, confidence


@dataclass
class MoveWindow:
    move_id: int
    frame_start: int
    frame_end: int
    t_start: float
    t_end: float


def segment_moves(contacts: list[dict], fps: float, n_frames: int) -> list[MoveWindow]:
    """A move spans one contact-change event to the next; changes <0.5s
    apart are merged. Boundaries come from contact start times."""
    if not contacts:
        return [MoveWindow(0, 0, n_frames - 1, 0.0, (n_frames - 1) / fps)]

    boundary_times = sorted({c["t_start"] for c in contacts})
    merged = [boundary_times[0]]
    for tm in boundary_times[1:]:
        if tm - merged[-1] < MOVE_MERGE_GAP_S:
            continue
        merged.append(tm)

    if merged[0] > 0:
        merged.insert(0, 0.0)
    end_time = (n_frames - 1) / fps
    if merged[-1] < end_time:
        merged.append(end_time)

    moves = []
    for i in range(len(merged) - 1):
        t_start, t_end = merged[i], merged[i + 1]
        moves.append(MoveWindow(
            move_id=i,
            frame_start=int(round(t_start * fps)),
            frame_end=min(int(round(t_end * fps)), n_frames - 1),
            t_start=t_start,
            t_end=t_end,
        ))
    return moves
