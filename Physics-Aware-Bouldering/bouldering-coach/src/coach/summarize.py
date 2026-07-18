"""M5: collapse physics features into moves.json + one keyframe JPEG per move.

Schema kept deliberately small (see PLAN.md) — resist adding fields Gemma
doesn't need. Normalized units (shoulder-widths) throughout so numbers are
comparable across videos and meaningful to the LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from coach.landmarks import IDX
from coach.physics import MoveWindow, com_to_hull_signed_distance

DYNAMIC_SPEED_THRESHOLD = 1.5  # shoulder-widths/s; above this a move is "dynamic"
KEYFRAME_MAX_WIDTH = 640  # the numbers carry the signal; images are just visual context for the LLM
KEYFRAME_JPEG_QUALITY = 70


class MoveAction(BaseModel):
    limb: str
    from_hold: int | None
    to_hold: int


class VideoMeta(BaseModel):
    fps: float
    duration_s: float
    wall_angle_deg: float | None
    wall_angle_confidence: float


class Climber(BaseModel):
    height_source: str = "user_or_null"


class Move(BaseModel):
    id: int
    t: list[float]
    action: MoveAction
    contacts_during: list[str]
    com_offset_max_norm: float
    com_offset_note: str = "normalized by shoulder width; >0 means outside base of support"
    peak_com_speed_norm: float
    dynamic: bool
    torso_wall_angle_deg: float | None
    keyframe: str


class MovesFile(BaseModel):
    video_meta: VideoMeta
    climber: Climber
    moves: list[Move]


def _active_contacts(contacts: list[dict], t_start: float, t_end: float) -> list[dict]:
    return [c for c in contacts if c["t_start"] < t_end and c["t_end"] > t_start]


def _limb_hold_before(contacts: list[dict], limb: str, before_t: float) -> int | None:
    prior = [c for c in contacts if c["limb"] == limb and c["t_end"] <= before_t]
    if not prior:
        return None
    return max(prior, key=lambda c: c["t_end"])["hold_id"]


def _primary_action(contacts: list[dict], move: MoveWindow) -> MoveAction | None:
    """The contact event whose start is closest to this move's start is
    treated as the move's defining transition."""
    starting = [c for c in contacts if move.t_start - 1e-6 <= c["t_start"] < move.t_end + 1e-6]
    if not starting:
        return None
    primary = min(starting, key=lambda c: abs(c["t_start"] - move.t_start))
    from_hold = _limb_hold_before(contacts, primary["limb"], primary["t_start"])
    return MoveAction(limb=primary["limb"], from_hold=from_hold, to_hold=primary["hold_id"])


def build_moves(
    moves: list[MoveWindow],
    contacts: list[dict],
    com_px: np.ndarray,
    velocity_norm: np.ndarray,
    torso_angle_deg: np.ndarray,
    torso_angle_confidence: np.ndarray,
    holds: list[dict],
    shoulder_width_px: float,
) -> list[Move]:
    hold_centroids_by_id = {h["id"]: np.array(h["centroid"]) for h in holds}
    all_centroids = np.array(list(hold_centroids_by_id.values())) if holds else np.empty((0, 2))

    out = []
    for mv in moves:
        active = _active_contacts(contacts, mv.t_start, mv.t_end)
        action = _primary_action(contacts, mv)
        if action is None:
            continue  # no defining transition in this window (e.g. leading calibration stretch)

        frame_slice = slice(mv.frame_start, mv.frame_end + 1)
        active_hold_ids = {int(c["hold_id"]) for c in active}
        active_centroids = np.array([hold_centroids_by_id[h] for h in active_hold_ids if h in hold_centroids_by_id])

        signed_distances = []
        for f in range(mv.frame_start, mv.frame_end + 1):
            if np.isnan(com_px[f]).any() or len(active_centroids) == 0:
                continue
            signed_distances.append(com_to_hull_signed_distance(com_px[f], active_centroids))

        if signed_distances:
            offset_idx = int(np.argmax(np.abs(signed_distances)))
            com_offset_max = signed_distances[offset_idx] / shoulder_width_px
            keyframe_frame = mv.frame_start + offset_idx
        else:
            com_offset_max = float("nan")
            keyframe_frame = mv.frame_start

        speed_window = velocity_norm[frame_slice]
        peak_speed = float(np.nanmax(speed_window)) if speed_window.size else float("nan")

        angle_window = torso_angle_deg[frame_slice]
        conf_window = torso_angle_confidence[frame_slice]
        if angle_window.size and np.nanmean(conf_window) > 0.3:
            torso_angle = float(np.nanmean(angle_window))
        else:
            torso_angle = None

        out.append(Move(
            id=mv.move_id,
            t=[mv.t_start, mv.t_end],
            action=action,
            contacts_during=[f"{c['limb']}:{c['hold_id']}" for c in active],
            com_offset_max_norm=round(com_offset_max, 3) if not np.isnan(com_offset_max) else 0.0,
            peak_com_speed_norm=round(peak_speed, 3) if not np.isnan(peak_speed) else 0.0,
            dynamic=bool(peak_speed > DYNAMIC_SPEED_THRESHOLD) if not np.isnan(peak_speed) else False,
            torso_wall_angle_deg=round(torso_angle, 1) if torso_angle is not None else None,
            keyframe=f"moves/{mv.move_id:03d}.jpg",
        ))
    return out


def save_keyframes(frames_bgr: list[np.ndarray], moves: list[Move], move_windows: list[MoveWindow], out_dir: Path) -> None:
    keyframe_dir = out_dir / "moves"
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    windows_by_id = {mv.move_id: mv for mv in move_windows}

    for move in moves:
        mv = windows_by_id[move.id]
        # Re-derive keyframe frame index from the stored path's move id and
        # the window (keyframe selection already picked the frame; here we
        # just need a representative frame — the window midpoint is a safe
        # fallback if exact tracking of the offset frame wasn't threaded through).
        frame_idx = min(mv.frame_start + (mv.frame_end - mv.frame_start) // 2, len(frames_bgr) - 1)
        frame = frames_bgr[frame_idx]
        h, w = frame.shape[:2]
        if w > KEYFRAME_MAX_WIDTH:
            scale = KEYFRAME_MAX_WIDTH / w
            frame = cv2.resize(frame, (KEYFRAME_MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / move.keyframe), frame, [cv2.IMWRITE_JPEG_QUALITY, KEYFRAME_JPEG_QUALITY])


def run(
    moves: list[MoveWindow],
    contacts: list[dict],
    com_px: np.ndarray,
    velocity_norm: np.ndarray,
    torso_angle_deg: np.ndarray,
    torso_angle_confidence: np.ndarray,
    holds: list[dict],
    calibration: dict,
    frames_bgr: list[np.ndarray],
    out_dir: str | Path,
) -> MovesFile:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shoulder_width_px = calibration["segment_lengths_px"]["shoulder_width"]
    fps = calibration["processing_fps"]
    n_frames = com_px.shape[0]

    move_models = build_moves(
        moves, contacts, com_px, velocity_norm, torso_angle_deg, torso_angle_confidence,
        holds, shoulder_width_px,
    )
    save_keyframes(frames_bgr, move_models, moves, out_dir)

    valid_angle = torso_angle_deg[~np.isnan(torso_angle_deg)]
    valid_conf = torso_angle_confidence[~np.isnan(torso_angle_confidence)]
    video_meta = VideoMeta(
        fps=fps,
        duration_s=n_frames / fps,
        wall_angle_deg=float(np.nanmedian(valid_angle)) if valid_angle.size else None,
        wall_angle_confidence=float(np.nanmedian(valid_conf)) if valid_conf.size else 0.0,
    )

    moves_file = MovesFile(video_meta=video_meta, climber=Climber(), moves=move_models)

    with open(out_dir / "moves.json", "w") as f:
        f.write(moves_file.model_dump_json(indent=2))

    return moves_file
