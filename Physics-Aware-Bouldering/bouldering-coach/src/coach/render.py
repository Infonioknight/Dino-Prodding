"""M7: burn skeleton + hold masks + CoM trace + contact flashes into the
original-resolution video. This is both the demo artifact and the primary
debugging tool."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from coach.landmarks import IDX, NAMES

SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_heel", "left_foot_index"),
    ("right_ankle", "right_heel"), ("right_heel", "right_foot_index"),
]

HOLD_COLOR = (60, 200, 255)
SKELETON_COLOR = (0, 255, 0)
COM_COLOR = (0, 0, 255)
CONTACT_FLASH_COLOR = (255, 0, 255)


def _active_contact_holds(contacts: list[dict], t: float) -> set[int]:
    return {c["hold_id"] for c in contacts if c["t_start"] <= t <= c["t_end"]}


def render_video(
    frames: list[np.ndarray],
    fps: float,
    landmarks_px: np.ndarray,
    com_px: np.ndarray,
    label_map: np.ndarray,
    holds: list[dict],
    contacts: list[dict],
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    hold_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    lh, lw = label_map.shape
    for hold in holds:
        mask = label_map == hold["id"]
        hold_overlay[:lh, :lw][mask] = HOLD_COLOR

    com_trail: list[tuple[int, int]] = []

    for i, frame in enumerate(frames):
        img = cv2.addWeighted(frame, 1.0, hold_overlay, 0.25, 0)

        t = i / fps
        active_holds = _active_contact_holds(contacts, t)
        for hold in holds:
            if hold["id"] in active_holds:
                cx, cy = hold["centroid"]
                cv2.circle(img, (int(cx), int(cy)), 14, CONTACT_FLASH_COLOR, 3)

        if i < landmarks_px.shape[0]:
            for a, b in SKELETON_EDGES:
                pa, pb = landmarks_px[i, IDX[a], :2], landmarks_px[i, IDX[b], :2]
                if np.isnan(pa).any() or np.isnan(pb).any():
                    continue
                cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)), SKELETON_COLOR, 2)
            for name, idx in IDX.items():
                p = landmarks_px[i, idx, :2]
                if not np.isnan(p).any():
                    cv2.circle(img, tuple(p.astype(int)), 3, SKELETON_COLOR, -1)

        if i < com_px.shape[0] and not np.isnan(com_px[i]).any():
            pt = tuple(com_px[i].astype(int))
            com_trail.append(pt)
            cv2.circle(img, pt, 6, COM_COLOR, -1)
            for j in range(1, len(com_trail)):
                cv2.line(img, com_trail[j - 1], com_trail[j], COM_COLOR, 1)

        writer.write(img)

    writer.release()
