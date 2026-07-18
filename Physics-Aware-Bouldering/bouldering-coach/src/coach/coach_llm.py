"""M6: Gemma 4 coaching. The only module allowed to make LLM calls.

Two-turn flow through an OpenAI-compatible client, endpoint/model/key read
entirely from env — nothing hardcoded, so swapping Gemini <-> HF router is
a config change, not a code change:
1. Clarify — if wall_angle_confidence is low or route style is ambiguous,
   ask the user 1-2 questions (CLI input() for MVP).
2. Coach — system prompt states the schema + the citation rule (every claim
   must cite a move id and a JSON number); output is coaching.md.
"""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

CLARIFY_CONFIDENCE_THRESHOLD = 0.4
MAX_KEYFRAME_IMAGES = 5  # provider-imposed cap on images per request; the numbers carry the signal anyway

SYSTEM_PROMPT = """You are a bouldering technique coach. You are given moves.json,
a per-move physics summary extracted from video (fields: t (start/end seconds),
action.limb/from_hold/to_hold, contacts_during, com_offset_max_norm — CoM
distance from the base of support in shoulder-widths, positive means outside
the support polygon; peak_com_speed_norm — peak CoM speed in shoulder-widths
per second; dynamic — whether the move exceeded a speed threshold;
torso_wall_angle_deg — torso lean off the wall, null if the video's estimate
was unreliable), plus one keyframe image per move.

Hard rule: every claim you make must cite a move id and a specific number
from the JSON. If the data does not support a claim, do not make it. Do not
guess at anything not present in the JSON or images.

Write per-move notes plus 2-3 overall priorities, in Markdown."""


_MOVE_REF_RE = re.compile(r"\bmove\s*#?\s*(\d+)\b", re.IGNORECASE)


def validate_citations(report_md: str, valid_move_ids: set[int]) -> list[str]:
    """Enforces the citation rule from SYSTEM_PROMPT: every body paragraph
    must reference a real move id. Returns the list of paragraphs that
    don't (empty list = report passes)."""
    violations = []
    for para in report_md.split("\n\n"):
        stripped = para.strip()
        if not stripped or stripped.startswith("#"):
            continue
        refs = {int(m) for m in _MOVE_REF_RE.findall(stripped)}
        if not refs or not (refs & valid_move_ids):
            violations.append(stripped)
    return violations


def _client() -> OpenAI:
    base_url = os.environ["LLM_BASE_URL"]
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("HF_TOKEN")
    if not api_key:
        raise RuntimeError("set LLM_API_KEY (or HF_TOKEN when LLM_BASE_URL is the HF router)")
    return OpenAI(base_url=base_url, api_key=api_key)


def _image_data_url(path: Path) -> str:
    ext = path.suffix.lstrip(".") or "jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{ext};base64,{data}"


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
)
def _chat(client: OpenAI, model: str, messages: list[dict]) -> str:
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content or ""


def clarify(moves_file: dict, ask_user=input) -> dict[str, str]:
    """Ask the user 1-2 questions when the data can't answer them itself.
    Returns a dict of question -> answer to fold into the coaching prompt."""
    answers = {}
    wall_conf = moves_file.get("video_meta", {}).get("wall_angle_confidence", 1.0)
    if wall_conf < CLARIFY_CONFIDENCE_THRESHOLD:
        answer = ask_user(
            "Wall-angle estimate is low-confidence — is this route on a slab, "
            "vertical wall, or overhang? "
        )
        answers["wall_style"] = answer
    return answers


def coach(moves_json_path: str | Path, keyframes_dir: str | Path, clarifications: dict[str, str] | None = None) -> str:
    """Runs the coach turn and returns the markdown report (does not write
    it — callers decide where coaching.md lands)."""
    import json

    moves_json_path = Path(moves_json_path)
    keyframes_dir = Path(keyframes_dir)
    moves_file = json.loads(moves_json_path.read_text())

    model = os.environ["LLM_MODEL"]
    client = _client()

    content: list[dict] = [{"type": "text", "text": json.dumps(moves_file, indent=2)}]
    if clarifications:
        clar_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in clarifications.items())
        content.append({"type": "text", "text": f"User clarifications:\n{clar_text}"})

    # The endpoint caps images per request well below one-per-move on longer
    # climbs; the JSON already carries every number for every move, so only
    # attach images for the most informative moves (largest balance loss).
    moves = moves_file.get("moves", [])
    notable = sorted(moves, key=lambda m: abs(m.get("com_offset_max_norm", 0.0)), reverse=True)[:MAX_KEYFRAME_IMAGES]
    notable_ids = {m["id"] for m in notable}
    for move in moves:
        if move["id"] not in notable_ids:
            continue
        img_path = keyframes_dir / move["keyframe"]
        if img_path.exists():
            content.append({"type": "text", "text": f"Keyframe image for move {move['id']}:"})
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(img_path)}})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    return _chat(client, model, messages)


def run(moves_json_path: str | Path, out_dir: str | Path, ask_user=input) -> Path:
    import json

    out_dir = Path(out_dir)
    moves_file = json.loads(Path(moves_json_path).read_text())

    clarifications = clarify(moves_file, ask_user=ask_user)
    report_md = coach(moves_json_path, out_dir, clarifications)

    out_path = out_dir / "coaching.md"
    out_path.write_text(report_md)
    return out_path
