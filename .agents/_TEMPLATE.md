# .agents changelog — copy this template per prompt. ≤30 lines. One prompt = one file.

## File naming

`.agents/YYYY-MM-DD_<short-slug>.md` — e.g. `.agents/2026-09-30_ingest-script.md`.
Never edit another prompt's file.

## Template (copy below the line)

---

# <short title> — YYYY-MM-DD

- **Prompt:** (one-line user request)
- **Phase:** P1 / P2 / P3 / P4 / P5 / P6 / infra
- **Changed:** (files + one line each on what/why)
- **Specs touched:** (e.g. chunk 5.0 s / tail <2.0 s / thr 0.20-0.65-0.50 / seed 42 / 128×157 — or "none")
- **Gold-set contact:** none (required line — if any, STOP and flag it)
- **Tests:** (command + result, e.g. `python -m pytest -q` → 12 passed)
- **For next agent:** (resumption pointer: what is done, what is open)
