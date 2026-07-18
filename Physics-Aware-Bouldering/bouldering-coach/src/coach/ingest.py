"""M1: frame extraction + calibration.

Pipeline: decode video -> downsample to a processing fps -> run MediaPipe
Pose on every processing frame -> find the calibration window (climber
standing still, both feet near the bottom of frame) -> compute median
per-segment pixel lengths over that window.

Artifact: out/<run>/calibration.json + out/<run>/calibration_frame.png
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from coach.landmarks import CALIBRATION_SEGMENTS, IDX, NAMES, VISIBILITY_THRESHOLD

PROCESSING_FPS = 12.0
CALIBRATION_WINDOW_S = 1.0  # sliding window used to find the calibration stance
FEET_BOTTOM_FRACTION = 0.85  # feet must be in the bottom 15% of the frame
MAX_VELOCITY_NORM = 0.01  # normalized hip-landmark velocity considered "still"

# mediapipe>=0.10.something dropped the legacy `mp.solutions.pose` API on
# some platform wheels in favor of the Tasks API, which needs a model
# bundle on disk rather than bundling weights in the pip package.
POSE_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "pose_landmarker_full.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"


@dataclass
class Frame:
    index: int
    timestamp_s: float
    image_bgr: np.ndarray


def extract_frames(video_path: str | Path, target_fps: float = PROCESSING_FPS) -> tuple[list[Frame], float, tuple[int, int]]:
    """Decode video, return frames downsampled to target_fps plus the
    original fps and (width, height) — both needed later for rendering."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(int(round(src_fps / target_fps)), 1)

    frames: list[Frame] = []
    idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(Frame(index=kept, timestamp_s=idx / src_fps, image_bgr=frame))
            kept += 1
        idx += 1
    cap.release()

    if not frames:
        raise ValueError(f"no frames decoded from {video_path}")
    return frames, src_fps, (width, height)


def _ensure_pose_model() -> Path:
    if not POSE_MODEL_PATH.exists():
        import urllib.request
        POSE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading pose landmarker model to {POSE_MODEL_PATH} ...")
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    return POSE_MODEL_PATH


def run_pose(frames: list[Frame]) -> np.ndarray:
    """Run MediaPipe Pose (Tasks API) sequentially over frames. Returns
    landmarks array of shape (T, 33, 4): x_px, y_px, z (relative depth),
    visibility."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    model_path = _ensure_pose_model()
    landmarks = np.full((len(frames), 33, 4), np.nan, dtype=np.float64)

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for f in frames:
            h, w = f.image_bgr.shape[:2]
            rgb = cv2.cvtColor(f.image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(f.timestamp_s * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            if not result.pose_landmarks:
                continue
            pose_landmarks = result.pose_landmarks[0]  # first detected person
            for i, lm in enumerate(pose_landmarks):
                landmarks[f.index, i] = (lm.x * w, lm.y * h, lm.z * w, lm.visibility)

    return landmarks


def find_calibration_window(landmarks_px: np.ndarray, frame_shape: tuple[int, int], fps: float) -> tuple[int, int] | None:
    """Locate a window where the climber stands still with both feet near
    the bottom of frame. Returns (start_idx, end_idx) inclusive, or None."""
    _, height = frame_shape
    window_len = max(int(round(CALIBRATION_WINDOW_S * fps)), 1)
    t = landmarks_px.shape[0]
    if t < window_len:
        return None

    hip_l, hip_r = IDX["left_hip"], IDX["right_hip"]
    heel_l, heel_r = IDX["left_heel"], IDX["right_heel"]

    # MediaPipe still emits an x/y guess even when it can't actually see a
    # landmark (visibility low) — gate on visibility first so a hallucinated
    # low-confidence position near the frame bottom can't masquerade as a
    # standing stance.
    stance_visible = np.all(landmarks_px[:, [hip_l, hip_r, heel_l, heel_r], 3] >= VISIBILITY_THRESHOLD, axis=1)

    hip_mid = np.nanmean(landmarks_px[:, [hip_l, hip_r], :2], axis=1)  # (T, 2)
    feet_y = np.nanmean(landmarks_px[:, [heel_l, heel_r], 1], axis=1)  # (T,)

    velocity = np.linalg.norm(np.diff(hip_mid, axis=0), axis=1) / max(height, 1)
    velocity = np.concatenate([[np.inf], velocity])  # align length, first frame has no prior

    for start in range(0, t - window_len + 1):
        end = start + window_len
        if not np.all(stance_visible[start:end]):
            continue
        feet_ok = np.all(feet_y[start:end] > FEET_BOTTOM_FRACTION * height)
        vel_ok = np.nanmean(velocity[start + 1:end]) < MAX_VELOCITY_NORM
        if feet_ok and vel_ok:
            return start, end - 1

    return None


def best_effort_window(landmarks_px: np.ndarray, fps: float) -> tuple[int, int]:
    """Fallback when find_calibration_window can't find a true standing
    stance (e.g. footage that doesn't follow the capture protocol): pick the
    window with the highest mean landmark visibility. Calibration lengths
    from this window are approximate — callers must flag it as such."""
    window_len = max(int(round(CALIBRATION_WINDOW_S * fps)), 1)
    t = landmarks_px.shape[0]
    window_len = min(window_len, t)

    visibility = landmarks_px[:, :, 3]
    mean_vis = np.nanmean(visibility, axis=1)  # (T,)
    best_start, best_score = 0, -1.0
    for start in range(0, t - window_len + 1):
        score = np.nanmean(mean_vis[start:start + window_len])
        if score > best_score:
            best_start, best_score = start, score
    return best_start, best_start + window_len - 1


def compute_calibration(landmarks_px: np.ndarray, window: tuple[int, int], mirror_missing_bilateral: bool = False) -> dict[str, float]:
    """Median per-segment pixel length over the calibration window, with
    visibility gating. Also computes torso height from shoulder/hip
    midpoints (shoulder width doubles as the perspective reference).

    mirror_missing_bilateral: for fallback calibration only (non-compliant
    footage where one side's limb is consistently occluded from the camera's
    angle) — borrow the mirrored side's median as an approximation rather
    than rejecting the clip outright. The real M1 acceptance test relies on
    a frontal calibration stance where this never triggers."""
    start, end = window
    lengths: dict[str, list[float]] = {name: [] for name, _, _ in CALIBRATION_SEGMENTS}
    lengths["torso_height"] = []

    for f in range(start, end + 1):
        for name, a, b in CALIBRATION_SEGMENTS:
            ai, bi = IDX[a], IDX[b]
            # NaN comparisons are always False, so "< threshold" alone would
            # let undetected (NaN-visibility) landmarks slip through the gate.
            if not (landmarks_px[f, ai, 3] >= VISIBILITY_THRESHOLD and landmarks_px[f, bi, 3] >= VISIBILITY_THRESHOLD):
                continue
            d = np.linalg.norm(landmarks_px[f, ai, :2] - landmarks_px[f, bi, :2])
            lengths[name].append(float(d))

        sh_l, sh_r = IDX["left_shoulder"], IDX["right_shoulder"]
        hip_l, hip_r = IDX["left_hip"], IDX["right_hip"]
        vis = landmarks_px[f, [sh_l, sh_r, hip_l, hip_r], 3]
        if np.all(vis >= VISIBILITY_THRESHOLD):
            shoulder_mid = landmarks_px[f, [sh_l, sh_r], :2].mean(axis=0)
            hip_mid = landmarks_px[f, [hip_l, hip_r], :2].mean(axis=0)
            lengths["torso_height"].append(float(np.linalg.norm(shoulder_mid - hip_mid)))

    if mirror_missing_bilateral:
        for name, values in lengths.items():
            if values or not name.endswith(("_l", "_r")):
                continue
            mirror = name[:-2] + ("_r" if name.endswith("_l") else "_l")
            if lengths.get(mirror):
                print(f"WARNING: '{name}' never visible enough to measure — borrowing '{mirror}' as an approximation")
                values.extend(lengths[mirror])

    calibration = {}
    for name, values in lengths.items():
        if not values:
            raise ValueError(f"no visible frames for segment '{name}' in calibration window — reject this clip")
        calibration[name] = float(np.median(values))
    return calibration


def save_calibration_frame(frame: Frame, landmarks_px: np.ndarray, frame_idx: int, out_path: Path) -> None:
    img = frame.image_bgr.copy()
    for i in range(33):
        x, y, _, vis = landmarks_px[frame_idx, i]
        if vis < VISIBILITY_THRESHOLD or np.isnan(x):
            continue
        cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
        if i in NAMES:
            cv2.putText(img, NAMES[i], (int(x) + 5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.imwrite(str(out_path), img)


def ingest(
    video_path: str | Path,
    out_dir: str | Path,
    allow_fallback_calibration: bool = False,
    force_fallback_calibration: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, src_fps, (width, height) = extract_frames(video_path)
    landmarks_px = run_pose(frames)

    window = None if force_fallback_calibration else find_calibration_window(landmarks_px, (width, height), PROCESSING_FPS)
    is_fallback = False
    if window is None:
        if not (allow_fallback_calibration or force_fallback_calibration):
            raise ValueError(
                "no calibration window found — capture protocol requires the climber "
                "to stand still on flat ground, facing camera, for ~2s at the start "
                "(pass allow_fallback_calibration=True to smoke-test non-compliant footage)"
            )
        print(
            "WARNING: no standing calibration stance found — falling back to a "
            "whole-video median per segment (using best_effort_window's highest-"
            "visibility window just for the annotated calibration frame). Segment "
            "lengths (and everything normalized by them) are approximate: a single "
            "1s window rarely has every limb visible on non-compliant footage, so "
            "the fallback pools visible frames across the entire clip instead. This "
            "is for pipeline smoke-testing only; M1's real acceptance test requires "
            "footage that follows the capture protocol."
        )
        window = (0, landmarks_px.shape[0] - 1)
        is_fallback = True

    calibration = compute_calibration(landmarks_px, window, mirror_missing_bilateral=is_fallback)

    if is_fallback:
        display_start, display_end = best_effort_window(landmarks_px, PROCESSING_FPS)
    else:
        display_start, display_end = window
    mid_frame = (display_start + display_end) // 2
    save_calibration_frame(frames[mid_frame], landmarks_px, mid_frame, out_dir / "calibration_frame.png")

    result = {
        "video_path": str(video_path),
        "src_fps": src_fps,
        "processing_fps": PROCESSING_FPS,
        "frame_size": {"width": width, "height": height},
        "calibration_window": {"start_frame": window[0], "end_frame": window[1]},
        "calibration_is_fallback": is_fallback,
        "segment_lengths_px": calibration,
    }
    with open(out_dir / "calibration.json", "w") as f:
        json.dump(result, f, indent=2)

    np.save(out_dir / "landmarks_px.npy", landmarks_px)
    return result
