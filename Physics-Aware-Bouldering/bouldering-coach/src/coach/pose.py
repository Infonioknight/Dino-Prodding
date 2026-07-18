"""M3: pose cleaning + contact detection.

Cleaning pipeline (in order), applied to the raw (T, 33, 4) landmark array
from ingest.run_pose:
1. Visibility gate — drop landmarks below VISIBILITY_THRESHOLD
2. Bone-length rejection — any frame where a calibrated segment deviates
   from its calibration length by >15% is a detection error
3. Interpolate short gaps (<0.3s); leave longer gaps as NaN
4. Savitzky-Golay smoothing (window ~0.5s)

Contact detection: hand/foot landmark within a shoulder-width-scaled
distance of any hold mask, sustained for >=N consecutive frames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from coach.landmarks import CALIBRATION_SEGMENTS, CONTACT_LANDMARKS, IDX, VISIBILITY_THRESHOLD

BONE_LENGTH_TOLERANCE = 0.15
MAX_GAP_INTERP_S = 0.3
SMOOTH_WINDOW_S = 0.5
CONTACT_DISTANCE_SHOULDER_WIDTHS = 0.5  # threshold scales with shoulder width
CONTACT_MIN_FRAMES = 3  # ~N consecutive frames at PROCESSING_FPS


@dataclass
class ContactEvent:
    limb: str
    hold_id: int
    t_start: float
    t_end: float
    frame_start: int
    frame_end: int

    def to_dict(self) -> dict:
        return {
            "limb": self.limb,
            "hold_id": self.hold_id,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
        }


def visibility_gate(landmarks_px: np.ndarray) -> np.ndarray:
    out = landmarks_px.copy()
    invisible = out[..., 3] < VISIBILITY_THRESHOLD
    out[invisible, 0:3] = np.nan
    return out


def bone_length_rejection(landmarks_px: np.ndarray, calibration_px: dict[str, float]) -> np.ndarray:
    """Any frame/segment pair exceeding its calibrated length by more than
    BONE_LENGTH_TOLERANCE marks both endpoint landmarks invalid for that frame."""
    out = landmarks_px.copy()
    t = out.shape[0]
    for name, a, b in CALIBRATION_SEGMENTS:
        if name not in calibration_px:
            continue
        ai, bi = IDX[a], IDX[b]
        ref = calibration_px[name]
        d = np.linalg.norm(out[:, ai, :2] - out[:, bi, :2], axis=1)
        bad = np.abs(d - ref) > BONE_LENGTH_TOLERANCE * ref
        bad = bad & ~np.isnan(d)
        out[bad, ai, 0:3] = np.nan
        out[bad, bi, 0:3] = np.nan
    return out


def interpolate_short_gaps(landmarks_px: np.ndarray, fps: float) -> np.ndarray:
    out = landmarks_px.copy()
    max_gap = max(int(round(MAX_GAP_INTERP_S * fps)), 1)
    t = out.shape[0]

    for lm in range(out.shape[1]):
        for dim in range(3):  # x, y, z — not visibility
            series = out[:, lm, dim]
            nan_mask = np.isnan(series)
            if not nan_mask.any() or nan_mask.all():
                continue

            idx = np.arange(t)
            gap_start = None
            for i in range(t):
                if nan_mask[i] and gap_start is None:
                    gap_start = i
                elif not nan_mask[i] and gap_start is not None:
                    gap_len = i - gap_start
                    if gap_len <= max_gap and gap_start > 0:
                        series[gap_start:i] = np.interp(
                            idx[gap_start:i], [gap_start - 1, i], [series[gap_start - 1], series[i]]
                        )
                    gap_start = None
            out[:, lm, dim] = series
    return out


def smooth(landmarks_px: np.ndarray, fps: float) -> np.ndarray:
    out = landmarks_px.copy()
    window = int(round(SMOOTH_WINDOW_S * fps))
    if window % 2 == 0:
        window += 1
    window = max(window, 5)
    polyorder = min(3, window - 1)

    for lm in range(out.shape[1]):
        for dim in range(3):
            series = out[:, lm, dim]
            valid = ~np.isnan(series)
            n_valid = valid.sum()
            if n_valid < window:
                continue
            # Savitzky-Golay requires no NaNs in-window; smooth only the
            # fully-valid contiguous run, leave gaps as NaN (per spec).
            run_start = None
            for i in range(len(series) + 1):
                at_end = i == len(series)
                if not at_end and valid[i] and run_start is None:
                    run_start = i
                elif (at_end or not valid[i]) and run_start is not None:
                    run_len = i - run_start
                    if run_len >= window:
                        series[run_start:i] = savgol_filter(series[run_start:i], window, polyorder)
                    run_start = None
            out[:, lm, dim] = series
    return out


def clean_landmarks(landmarks_px: np.ndarray, calibration_px: dict[str, float], fps: float) -> np.ndarray:
    out = visibility_gate(landmarks_px)
    out = bone_length_rejection(out, calibration_px)
    out = interpolate_short_gaps(out, fps)
    out = smooth(out, fps)
    assert not np.any(np.isinf(out)), "cleaning pipeline produced Inf — fix upstream, do not silently mask"
    return out


def detect_contacts(
    landmarks_px: np.ndarray,
    label_map: np.ndarray,
    shoulder_width_px: float,
    fps: float,
) -> list[ContactEvent]:
    """For each processing frame, find the nearest hold to each contact
    landmark (within threshold); collapse runs of >=CONTACT_MIN_FRAMES on
    the same hold into a single ContactEvent."""
    threshold = CONTACT_DISTANCE_SHOULDER_WIDTHS * shoulder_width_px
    t = landmarks_px.shape[0]

    # Precompute hold centroids and per-pixel nearest-hold via distance transform
    # per hold would be expensive; instead, for each frame, check distance from
    # the landmark to the nearest foreground pixel of each hold's bounding area.
    hold_ids = [i for i in np.unique(label_map) if i != 0]
    hold_pixel_coords = {}
    for hid in hold_ids:
        ys, xs = np.where(label_map == hid)
        hold_pixel_coords[hid] = np.stack([xs, ys], axis=1).astype(np.float64)

    events: list[ContactEvent] = []
    for limb, landmark_name in CONTACT_LANDMARKS.items():
        li = IDX[landmark_name]
        current_hold = None
        run_start = None

        for f in range(t):
            x, y = landmarks_px[f, li, 0], landmarks_px[f, li, 1]
            touched_hold = None
            if not np.isnan(x):
                best_dist = threshold
                for hid, coords in hold_pixel_coords.items():
                    d = np.min(np.linalg.norm(coords - np.array([x, y]), axis=1))
                    if d < best_dist:
                        best_dist = d
                        touched_hold = hid

            if touched_hold == current_hold and touched_hold is not None:
                continue
            # hold changed (or run ended)
            if current_hold is not None and run_start is not None:
                run_len = f - run_start
                if run_len >= CONTACT_MIN_FRAMES:
                    events.append(ContactEvent(
                        limb=limb, hold_id=int(current_hold),
                        t_start=run_start / fps, t_end=(f - 1) / fps,
                        frame_start=run_start, frame_end=f - 1,
                    ))
            current_hold = touched_hold
            run_start = f if touched_hold is not None else None

        if current_hold is not None and run_start is not None:
            run_len = t - run_start
            if run_len >= CONTACT_MIN_FRAMES:
                events.append(ContactEvent(
                    limb=limb, hold_id=int(current_hold),
                    t_start=run_start / fps, t_end=(t - 1) / fps,
                    frame_start=run_start, frame_end=t - 1,
                ))

    events.sort(key=lambda e: e.t_start)
    return events


def run(
    landmarks_px: np.ndarray,
    calibration: dict,
    label_map: np.ndarray,
    out_dir: str | Path,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = calibration["processing_fps"]
    seg_lengths = calibration["segment_lengths_px"]
    shoulder_width_px = seg_lengths["shoulder_width"]

    cleaned = clean_landmarks(landmarks_px, seg_lengths, fps)
    np.save(out_dir / "landmarks_clean.npy", cleaned)

    contacts = detect_contacts(cleaned, label_map, shoulder_width_px, fps)
    result = {"contacts": [c.to_dict() for c in contacts]}
    with open(out_dir / "contacts.json", "w") as f:
        json.dump(result, f, indent=2)

    return result
