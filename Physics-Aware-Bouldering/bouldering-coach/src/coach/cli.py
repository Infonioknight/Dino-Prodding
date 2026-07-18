"""Single entrypoint: uv run coach analyze data/raw/climb.mp4 --out out/run1/"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv


def analyze(
    video_path: str,
    out_dir: str,
    skip_llm: bool = False,
    allow_fallback_calibration: bool = False,
    force_fallback_calibration: bool = False,
) -> None:
    from coach import coach_llm, holds as holds_mod, ingest, physics, pose, render, summarize

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] ingest + calibration: {video_path}")
    calibration = ingest.ingest(
        video_path, out,
        allow_fallback_calibration=allow_fallback_calibration,
        force_fallback_calibration=force_fallback_calibration,
    )

    print("[2/7] hold segmentation")
    import numpy as np
    frames, _, _ = ingest.extract_frames(video_path)
    first_frame = frames[0].image_bgr
    holds_result = holds_mod.run(first_frame, out)
    label_map = np.load(out / "holds_label_map.npy")

    print("[3/7] pose cleaning + contacts")
    landmarks_px = np.load(out / "landmarks_px.npy")
    pose_result = pose.run(landmarks_px, calibration, label_map, out)
    landmarks_clean = np.load(out / "landmarks_clean.npy")

    print("[4/7] physics features")
    com_px = physics.compute_com(landmarks_clean)
    seg = calibration["segment_lengths_px"]
    kinematics = physics.compute_kinematics(com_px, calibration["processing_fps"], seg["shoulder_width"])
    torso_angle_deg, torso_angle_confidence = physics.torso_wall_angle(
        landmarks_clean, seg["torso_height"], seg["shoulder_width"]
    )
    moves = physics.segment_moves(pose_result["contacts"], calibration["processing_fps"], landmarks_clean.shape[0])

    print("[5/7] move summaries + keyframes")
    frames_bgr = [f.image_bgr for f in frames]
    moves_file = summarize.run(
        moves, pose_result["contacts"], com_px, kinematics.velocity_norm,
        torso_angle_deg, torso_angle_confidence, holds_result["holds"], calibration,
        frames_bgr, out,
    )

    if skip_llm:
        print("[6/7] coaching: skipped (--skip-llm)")
    else:
        print("[6/7] coaching (Gemma 4)")
        coach_llm.run(out / "moves.json", out)

    print("[7/7] rendering annotated video")
    render.render_video(
        frames_bgr, calibration["processing_fps"], landmarks_clean, com_px,
        label_map, holds_result["holds"], pose_result["contacts"],
        out / "annotated.mp4",
    )

    print(f"done: {out}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="coach")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze_p = sub.add_parser("analyze", help="run the full pipeline on a video")
    analyze_p.add_argument("video", help="path to input video")
    analyze_p.add_argument("--out", required=True, help="output directory")
    analyze_p.add_argument("--skip-llm", action="store_true", help="skip the Gemma coaching call")
    analyze_p.add_argument(
        "--allow-fallback-calibration", action="store_true",
        help="smoke-test footage that doesn't follow the capture protocol (no standing stance) "
             "by falling back to the highest-visibility window; calibration is then approximate",
    )
    analyze_p.add_argument(
        "--force-fallback-calibration", action="store_true",
        help="skip the standing-stance detector entirely and go straight to fallback calibration "
             "(use when you already know the footage has no calibration stance)",
    )

    args = parser.parse_args()
    if args.command == "analyze":
        analyze(
            args.video, args.out, skip_llm=args.skip_llm,
            allow_fallback_calibration=args.allow_fallback_calibration,
            force_fallback_calibration=args.force_fallback_calibration,
        )


if __name__ == "__main__":
    main()
