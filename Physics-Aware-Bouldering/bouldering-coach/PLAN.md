# Bouldering Coach MVP — Build Plan (Claude Code)

Drop this file in the repo root as `PLAN.md`. Work through milestones in order; each has a "done when" check so Claude Code has a concrete target and you can verify without reading every line of code.

## MVP definition

**Input:** one climbing video (fixed camera, roughly perpendicular to wall) + the first ~2 seconds containing the climber standing on flat ground (calibration frames are inside the same video — no separate file).

**Output:**
1. Annotated video: skeleton + hold masks + CoM trace + contact highlights
2. `moves.json`: per-move physics summary
3. `coaching.md`: Gemma 4's technique feedback grounded in the JSON

**Explicitly cut from MVP:** grade prediction, force estimation, route color grouping, multi-attempt comparison, any UI. These are all v2. The MVP proves the pipeline: pixels → physics → grounded language.

## Why this decomposition

The LLM never does physics; the physics code never does judgment. Every milestone below is independently runnable and produces a visual or JSON artifact. This matters specifically for Claude Code: it can't watch a video, so every stage must emit inspectable outputs (overlay PNGs, plots, JSON) that either you eyeball or a script asserts on. Milestones that only produce in-memory state are undebuggable in this workflow.

## Repo layout

```
bouldering-coach/
├── PLAN.md                  # this file
├── CLAUDE.md                # see bottom of this doc
├── pyproject.toml           # uv-managed
├── data/
│   ├── raw/                 # input videos (gitignored)
│   └── fixtures/            # 1 short clip (~10s) used as the test fixture everywhere
├── src/coach/
│   ├── ingest.py            # M1: frames, undistort, calibration
│   ├── holds.py             # M2: one-shot hold segmentation
│   ├── pose.py              # M3: MediaPipe + smoothing + rejection
│   ├── physics.py           # M4: CoM, base of support, wall angle, move segmentation
│   ├── summarize.py         # M5: moves.json + keyframes
│   ├── coach_llm.py         # M6: Gemma 4 call
│   └── render.py            # M7: overlay video
├── out/                     # all artifacts, one subdir per run
└── tests/                   # pytest, runs on the fixture clip
```

Justification: one module per milestone means each Claude Code session touches one file plus tests. `out/<run_id>/` keeps every stage's artifact side by side for debugging.

## Stack

| Choice | Why |
|---|---|
| Python 3.11 + `uv` | Fast env management; Claude Code handles `uv add` cleanly |
| OpenCV | Frame extraction, undistortion, homography, rendering — no reason to use anything else |
| MediaPipe Pose (`mediapipe` pip) | You already know its landmark IDs from the anthropometry work; CPU real-time on Mac; gives foot/heel/toe points needed for contact detection |
| PyTorch + MPS, DINOv2 (torch hub, ViT-S/14 to start) | You've already validated attention-map hold segmentation; reuse your Boulder code. ViT-S first — upgrade to B only if masks are bad |
| scipy / numpy | Savitzky-Golay smoothing, convex hulls, finite differences |
| Hosted Gemma 4 via OpenAI-compatible client (`openai` pip); endpoint set by env: `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | 8GB M1 Air can't host Gemma 4 locally. Both the Gemini API and the HF Inference Providers router speak the OpenAI-compatible protocol, so the backend is config, not code. Dev on Gemini's free tier (AI Studio key, 15 RPM, $0, model `gemma-4-26b-a4b-it`); flip env vars to `https://router.huggingface.co/v1` + `google/gemma-4-31B-it` (HF token) for the HF-themed demo — pin a provider suffix (e.g. `:cerebras`) rather than `:auto` to avoid per-provider image-handling quirks mid-run |
| pytest + a 10s fixture clip | Every milestone gets a test against the same clip, so regressions surface immediately |

## Milestone 0 — Environment + capture protocol

**Build:** `uv init`, deps installed, `.env.example` committed with `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` (default: Gemini API endpoint + `gemma-4-26b-a4b-it`, key from Google AI Studio), verified with a one-line image+text smoke test through the OpenAI-compatible client. Record 2–3 clips at the gym following the protocol; pick the shortest/cleanest as `data/fixtures/`.

**Capture protocol (this is a spec, not a suggestion):** phone on a tripod or propped stable, landscape, as perpendicular to the wall as the gym allows, full route + floor in frame, climber stands facing camera on flat ground for ~2s before touching the wall, 0.5x/wide lens OFF (main lens only — less distortion).

**Justification:** every downstream simplification (one-shot hold seg, foreshortening calibration, homography-free tracking) depends on the fixed camera and the calibration stance. Constraining capture is the cheapest engineering you'll do on this project.

**Done when:** `uv run python -c "import mediapipe, torch; print(torch.backends.mps.is_available())"` prints True, the configured LLM endpoint returns a caption for a test image, fixture clip is committed (git-lfs or just small enough).

## Milestone 1 — Ingestion + calibration

**Build:** `ingest.py` — extract frames (downsample to ~10–15 fps for processing; keep original for rendering), detect the calibration window (climber standing, low landmark velocity, both feet near the bottom of frame), run MediaPipe on those frames, compute median limb lengths in pixels (upper/lower arm, upper/lower leg, shoulder width, hip–shoulder torso height) with landmark-visibility gating.

Skip lens undistortion for MVP — main lens on a modern phone is mild, and calibrating intrinsics adds a whole workflow. Add a `TODO(v2)` marker.

**Justification:** median-over-window kills MediaPipe jitter; these pixel lengths are the ground truth for both the bone-length rejection test (M3) and the foreshortening wall-angle estimate (M4). Shoulder width doubles as the perspective-normalization reference later.

**Artifact:** `out/<run>/calibration.json` + one annotated calibration frame PNG.
**Done when:** limb lengths across the 2–3 recorded clips of the same person agree within ~5%.

## Milestone 2 — One-shot hold segmentation

**Build:** `holds.py` — pick one clean frame before the climber enters (reuse the M1 stance detection: the frame just before first wall contact, or simply frame 0 if the climber starts out of frame). Run your DINOv2 attention-map segmentation on it (port from the Boulder project). Post-process: threshold → connected components → filter by area → per-hold ID + centroid + mask. Serialize masks to a single PNG label map + `holds.json`.

**Justification:** holds are static and the camera is fixed, so segmenting once eliminates occlusion handling entirely — the biggest scope cut in the whole design. DINOv2 over RF-DETR because you own working code and it avoids a fine-tuning loop. **Fallback (only if masks are unusable):** train a light seg head on frozen DINOv2 features with Tomáš Sláma's Kaggle indoor-hold dataset. Don't start there.

**Artifact:** overlay PNG with numbered holds.
**Done when:** ≥90% of the route's holds get a mask on the fixture clip (count by eye once — that's the acceptance test).

## Milestone 3 — Pose + cleaning + contacts

**Build:** `pose.py` — MediaPipe per frame → raw landmark array (T × 33 × 3 + visibility). Cleaning pipeline, in order:
1. Visibility gate (drop landmarks below threshold)
2. Bone-length rejection: any frame where a segment exceeds its calibrated length by >15% is a detection error → mark landmark invalid (free test courtesy of M1)
3. Interpolate short gaps (<0.3s), leave longer gaps as NaN
4. Savitzky-Golay smooth (window ~0.5s)

Contact detection: hand/foot landmark within distance threshold of any hold mask for ≥N consecutive frames → contact event `(limb, hold_id, t_start, t_end)`. Threshold scales with shoulder width so it adapts to camera distance.

**Justification:** MediaPipe degrades on compressed/horizontal climbing poses and can swap left/right — the bone-length check catches most of it deterministically, no ML needed. Contact events are the backbone of everything after this: moves, base of support, keyframes.

**Artifact:** skeleton-overlay video (or every-10th-frame PNG strip) + `contacts.json`.
**Done when:** watching the overlay, contact highlights match reality for ≥80% of hand/foot placements on the fixture clip. This is the milestone where you'll iterate most — budget for it.

## Milestone 4 — Physics features

**Build:** `physics.py` —
- **CoM per frame:** de Leva segment mass fractions over the 2D skeleton. Hardcode the table; it's ~15 rows.
- **Base of support:** convex hull of active contact points; signed distance of CoM to hull (negative = inside).
- **Kinematics:** CoM velocity/acceleration by finite differences on the smoothed trace; per-move peak velocity and a jerk-based smoothness score.
- **Torso-to-wall angle:** foreshortening on the hip–shoulder quad — apparent torso height / calibrated torso height, normalized by shoulder width (perspective reference), `θ ≈ arccos(ratio)`, clamped and reported with a confidence value (agreement with MediaPipe's z-coordinates). Low confidence → emit `null`; Gemma asks the user instead (M6).
- **Move segmentation:** a "move" spans one contact-change event to the next, merging changes <0.5s apart.

**Justification:** these are the metrics a coach actually references (balance, hips, static vs dynamic), and all are honestly computable from monocular 2D. Torso quad over individual limbs because four averaged landmarks beat one noisy forearm and torso angle is the coachable quantity. No force estimation — statically indeterminate from video; reporting "loading" would be fake precision.

**Artifact:** matplotlib panel per run: CoM trace over the wall image, CoM-to-hull distance over time, wall-angle over time.
**Done when:** the CoM trace visually sits inside the body throughout, and move boundaries in the plot line up with contact changes you can see in the video.

## Milestone 5 — Move summaries + keyframes

**Build:** `summarize.py` — emit `moves.json` and one JPEG keyframe per move boundary (frame of max CoM-to-hull distance within the move — the most informative instant).

Schema (keep it this small; resist adding fields Gemma doesn't need):

```json
{
  "video_meta": {"fps": 30, "duration_s": 22.4, "wall_angle_deg": 12.0, "wall_angle_confidence": 0.71},
  "climber": {"height_source": "user_or_null"},
  "moves": [
    {
      "id": 3,
      "t": [8.2, 10.1],
      "action": {"limb": "left_hand", "from_hold": 4, "to_hold": 7},
      "contacts_during": ["right_hand:5", "left_foot:2", "right_foot:3"],
      "com_offset_max_norm": 0.42,
      "com_offset_note": "normalized by shoulder width; >0 means outside base of support",
      "peak_com_speed_norm": 1.8,
      "dynamic": true,
      "torso_wall_angle_deg": 31,
      "keyframe": "moves/003.jpg"
    }
  ]
}
```

**Justification:** normalized units (shoulder-widths, not pixels) make numbers comparable across videos and meaningful to the LLM. One keyframe per move keeps Gemma's visual context small — use a low visual token budget for these; the numbers carry the signal, the image is context.

**Done when:** `moves.json` validates against a pydantic model and a human reading it can reconstruct the climb's story without the video.

## Milestone 6 — Gemma 4 coaching

**Build:** `coach_llm.py` — two-turn flow through the OpenAI-compatible client, endpoint/model/key read from `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` and nothing hardcoded. Images go in as base64 data-URL `image_url` parts (portable across Gemini and the HF router). Exponential-backoff retry on 429s — Gemini's free tier is 15 RPM and HF free credits are small. Keep keyframes JPEG-compressed and few — the numbers carry the signal. Two-turn flow:
1. **Clarify:** if `wall_angle_confidence < threshold` or route style is ambiguous, Gemma asks the user 1–2 questions (CLI `input()` for MVP).
2. **Coach:** system prompt = role + the schema's field definitions + hard rule: *every claim must cite a move id and a number from the JSON; if the data doesn't support a claim, don't make it.* User message = `moves.json` + keyframes. Output = `coaching.md` with per-move notes + 2–3 overall priorities.

**Justification:** the citation rule is what makes this non-vibes — it's the contract between the physics layer and the language layer, and it's testable: a regex over the output can assert every paragraph references a move id. Two-turn clarify-then-coach matches the original concept without needing a UI.

**Done when:** on the fixture clip, the report contains zero claims not traceable to a JSON field, and at least one note matches your own read of the climb.

## Milestone 7 — Rendered output + CLI

**Build:** `render.py` (skeleton + hold masks + CoM trace + contact flashes burned into the video, coaching notes as timed subtitles if cheap) and a single entrypoint:

```
uv run coach analyze data/raw/climb.mp4 --out out/run1/
```

**Justification:** the overlay video is the demo artifact and your primary debugging tool rolled into one; the single CLI is what makes the whole thing reproducible end to end.

**Done when:** one command on a fresh clip produces the annotated video, `moves.json`, and `coaching.md` with no manual steps.

## CLAUDE.md (put this in the repo)

```markdown
# Project: bouldering coach MVP
Read PLAN.md first. Work one milestone at a time; do not start the next
until the current "done when" passes.

Rules:
- Every stage writes an inspectable artifact (PNG/JSON/plot) to out/<run>/.
  You cannot watch videos — emit frame strips and plots and reason from those.
- All tests run against data/fixtures/ only. Never require a new recording.
- No LLM calls anywhere except src/coach/coach_llm.py.
- LLM endpoint config (LLM_BASE_URL, LLM_MODEL, LLM_API_KEY) comes from the
  environment; never hardcode endpoints, models, or keys, never commit keys.
  Tests mock the LLM call — never hit any API from pytest.
- Physics stays in physics.py; no heuristic "coaching" logic in Python.
- Distances are normalized by calibrated shoulder width, never raw pixels.
- Prefer boring: numpy/scipy over new deps; ask before adding a dependency.
- MPS quirks: fall back to CPU for ops MPS doesn't support; never silently
  produce NaNs — assert on them.
```

## Order-of-work notes

- M1–M3 are the risk; M4–M7 are mostly deterministic once contacts are right. If you want a demo fast, a legitimate checkpoint after M3 is "skeleton + holds + contacts overlay video" — that alone is shareable.
- Expected time sinks, in order: contact thresholds (M3), DINOv2 mask post-processing (M2), MediaPipe on weird body positions (M3). Nothing in M4+ should surprise you.
- v2 backlog, deliberately parked: route color grouping, multi-attempt trajectory comparison (cheap once holds are registered), 3D lifting (MotionBERT), lens undistortion, grade estimate as a soft range, force ranges via static equilibrium with stated assumptions.
