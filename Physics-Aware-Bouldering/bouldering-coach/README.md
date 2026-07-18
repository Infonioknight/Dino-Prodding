# bouldering-coach

Pixels -> physics -> grounded coaching language. See [PLAN.md](PLAN.md) for
the full milestone breakdown and [CLAUDE.md](CLAUDE.md) for working rules.

## Setup

```
uv sync
cp .env.example .env   # fill in LLM_API_KEY or HF_TOKEN
```

Verify the environment:

```
uv run python -c "import mediapipe, torch; print(torch.backends.mps.is_available())"
uv run pytest -q
```

Record a fixture clip following the capture protocol in PLAN.md (Milestone
0) and drop it in `data/fixtures/`.

## Run

```
uv run coach analyze data/raw/climb.mp4 --out out/run1/
```

Produces `out/run1/annotated.mp4`, `moves.json`, and `coaching.md`.
