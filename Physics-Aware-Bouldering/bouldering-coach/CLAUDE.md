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
