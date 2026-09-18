# Root cause: 747-entry recommendation list in the frontend

**One sentence:** `frontend-export/export_gameweek.py` originally reused each
specialist's full `recommendations`/`vetoes` list — sized for the
Manager/solver's aggregation, which needs the whole affordable candidate pool
(`orchestrator/callers.py`'s large `top_k`) — directly as the debate
transcript's display data, instead of building a separately curated top-N
summary for that field.

## Which of the two candidate explanations this actually was

Not a missing top-N cutoff bolted onto an otherwise-correct export step — the
export step never had a cutoff to begin with, because it was pointed at the
wrong source volume: it dumped the same `recommendations`/`vetoes` arrays the
Manager consumes internally straight into the transcript JSON, unfiltered.
The fix (`export_gameweek.py:65`, `TRANSCRIPT_DISPLAY_LIMIT = 5`) slices each
agent's `recommendations`/`vetoes` to 5 entries at the export boundary, after
confirming the Manager's own aggregation still runs on the full unsliced data
from `state["first_round"]` — only the display copy is truncated.

## Why this framing matters for future export code

Any new field added to the frontend export that's sourced from internal agent
or solver state needs its own explicit "how much of this is actually for
display" decision — internal sizing (`top_k`, full candidate pools, full
squad-selection traces) defaults to being too large for a UI, and nothing
upstream of the export step will cap it. The export step is the only place
in the pipeline responsible for that judgment; assuming an upstream size
limit already exists is what caused this bug.
