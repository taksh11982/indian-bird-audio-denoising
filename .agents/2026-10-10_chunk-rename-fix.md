# Fix stdlib shadowing: chunk.py -> chunking.py — 2026-10-10

- **Prompt:** Colab ImportError (aifc did `from chunk import Chunk`, got ours)
- **Phase:** P4-Gavali bugfix (no model/threshold/data changes)
- **Changed:** `src/chunk.py` → `src/chunking.py` via `git mv` (history kept); `agents.md:52` layout line updated to match
- **Changed:** nothing else — no Gavali code touched; runbook prose needed no edit
- **Specs touched:** none (5.0 s, thr 0.20-0.65-0.50, seed 42 unchanged)
- **Gold-set contact:** none
- **Tests:** stdlib-collision scan over `src/*.py` → NONE; `src/chunk.py` gone confirmed; `py_compile` OK; `gavali_features --check` 3/3 (157,39) OK (local 3.13 lacks aifc/chunk, which is why smoke passed here and failed on Colab)
- **For next agent:** user runs `git pull` on Colab, reruns Cell 5 unchanged
