"""M2: one-shot hold segmentation on a single clean frame (holds are static,
camera is fixed — segment once, never deal with occlusion).

Method: DINOv2 (ViT-S/14) patch features -> PCA foreground/background split
(the classic DINO trick: the first principal component of patch tokens
separates salient foreground objects from a low-texture background) ->
per-patch score upsampled to image resolution -> threshold -> connected
components -> filter by area -> one mask + centroid per hold.

Fallback (only if masks come out unusable): train a light seg head on frozen
DINOv2 features with Tomas Slama's Kaggle indoor-hold dataset. Not
implemented here — don't start there per PLAN.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PATCH_SIZE = 14
MIN_HOLD_AREA_PX = 150  # filters speckle noise after thresholding
FG_PERCENTILE = 60.0  # patches above this percentile of PC1 score are foreground


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_dinov2(model_name: str = "dinov2_vits14"):
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval()
    return model


def _patch_grid_size(h: int, w: int) -> tuple[int, int]:
    return h // PATCH_SIZE, w // PATCH_SIZE


def _preprocess(image_bgr: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = image_bgr.shape[:2]
    gh, gw = _patch_grid_size(h, w)
    h_crop, w_crop = gh * PATCH_SIZE, gw * PATCH_SIZE
    image_rgb = cv2.cvtColor(image_bgr[:h_crop, :w_crop], cv2.COLOR_BGR2RGB)

    img = Image.fromarray(image_rgb)
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, (gh, gw)


def _foreground_score(model, image_bgr: np.ndarray, device: str) -> tuple[np.ndarray, tuple[int, int]]:
    """Returns per-patch PC1 score (gh*gw,) and the patch grid shape."""
    tensor, (gh, gw) = _preprocess(image_bgr)
    tensor = tensor.to(device)
    model = model.to(device)

    with torch.no_grad():
        features = model.forward_features(tensor)
        patch_tokens = features["x_norm_patchtokens"][0]  # (gh*gw, C)

    tokens = patch_tokens.cpu().numpy()
    tokens = tokens - tokens.mean(axis=0, keepdims=True)
    # PC1 via SVD; sign is arbitrary so we orient it below.
    _, _, vt = np.linalg.svd(tokens, full_matrices=False)
    score = tokens @ vt[0]

    # Orient so foreground (holds/climber) is high: assume image borders are
    # mostly background wall, so the border-patch mean should be the low end.
    grid = score.reshape(gh, gw)
    border = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
    if np.mean(border) > np.median(score):
        score = -score

    return score, (gh, gw)


def segment_holds(image_bgr: np.ndarray, model=None, device: str | None = None) -> tuple[np.ndarray, list[dict]]:
    """Returns (label_map, holds) where label_map is an int32 image (0 =
    background, 1..N = hold id) and holds is a list of {id, centroid, area_px, bbox}."""
    device = device or _device()
    model = model or _load_dinov2()

    score, (gh, gw) = _foreground_score(model, image_bgr, device)
    threshold = np.percentile(score, FG_PERCENTILE)
    fg_grid = (score > threshold).reshape(gh, gw).astype(np.uint8)

    h_crop, w_crop = gh * PATCH_SIZE, gw * PATCH_SIZE
    fg_mask = cv2.resize(fg_grid, (w_crop, h_crop), interpolation=cv2.INTER_NEAREST)

    # Morphological cleanup: close small gaps within a hold, open to drop speckle.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)

    label_map = np.zeros((image_bgr.shape[0], image_bgr.shape[1]), dtype=np.int32)
    holds = []
    hold_id = 1
    for lbl in range(1, n_labels):  # 0 is background
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < MIN_HOLD_AREA_PX:
            continue
        mask = labels == lbl
        label_map[:h_crop, :w_crop][mask] = hold_id
        x, y, w, h, _ = stats[lbl]
        cx, cy = centroids[lbl]
        holds.append({
            "id": hold_id,
            "centroid": [float(cx), float(cy)],
            "area_px": area,
            "bbox": [int(x), int(y), int(w), int(h)],
        })
        hold_id += 1

    return label_map, holds


def save_overlay(image_bgr: np.ndarray, label_map: np.ndarray, holds: list[dict], out_path: Path) -> None:
    overlay = image_bgr.copy()
    rng = np.random.default_rng(0)
    colors = {h["id"]: tuple(int(c) for c in rng.integers(60, 255, size=3)) for h in holds}

    for hold in holds:
        mask = label_map == hold["id"]
        color = colors[hold["id"]]
        overlay[mask] = (overlay[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        cx, cy = hold["centroid"]
        cv2.putText(overlay, str(hold["id"]), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imwrite(str(out_path), overlay)


def run(frame_bgr: np.ndarray, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map, holds = segment_holds(frame_bgr)

    np.save(out_dir / "holds_label_map.npy", label_map)
    save_overlay(frame_bgr, label_map, holds, out_dir / "holds_overlay.png")

    result = {"num_holds": len(holds), "holds": holds}
    with open(out_dir / "holds.json", "w") as f:
        json.dump(result, f, indent=2)
    return result
