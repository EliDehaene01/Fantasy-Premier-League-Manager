# Human-in-the-loop: recommend and confirm

ARCHITECTURE.md 10's full description; this is the operational walkthrough -
what a human actually does, gameweek to gameweek.

## The flow

1. The `deadline-checker` CronJob (k8s/deadline-checker-cronjob.yaml) triggers
   the `ingestion` Job, which pulls fresh FPL data ahead of the deadline.
2. Once ingestion completes, something calls `POST /run` on the orchestrator
   Deployment (orchestrator/service.py) with the gameweek and the freshly-
   ingested player pool. The graph runs the six-agent debate, the Manager
   aggregates and solves, and the mode branch pauses at `interrupt()` -
   ARCHITECTURE.md 6c's live-mode behavior, already tested
   (orchestrator/test_graph.py::test_live_mode_pauses_for_approval_then_resumes).
3. The `frontend-export` step (not yet automatically invoked by the weekly
   pipeline script) would read that paused state and publish it as static JSON, the same
   shape `frontend-export/export_gameweek.py` already produces and the
   frontend already renders (verified in a real browser - see the frontend
   commit). No push notification is sent; the human checks the site.
4. **The human reviews the proposal on the site**: proposed transfers,
   captain/vice-captain, any chip call, and the full six-agent debate
   transcript (including the reaction round) - everything needed to
   understand *why*, not just *what*.
5. **The human applies the change manually in the real FPL app** -
   fantasy.premierleague.com or the official app. Nothing in this system
   writes to the live FPL account automatically (CLAUDE.md hard constraint;
   ARCHITECTURE.md 12 documents a possible future auto-apply design,
   deliberately not built).
6. Whether or not the human ends up matching the recommendation exactly,
   the decision is logged by resuming the graph's interrupt (see below) -
   this closes out the checkpointed state so the next gameweek starts
   clean, and gives an honest record of what was actually decided.

## Recording the decision (the "(Stretch)" approve/reject action)

The mechanism already exists and is tested
(orchestrator/service.py::resume, orchestrator/test_service.py) - what's
below is the lightweight way to actually call it. Deliberately NOT a
button on the frontend: CLAUDE.md's hard constraint is that the frontend
never calls any backend, directly or indirectly, so approval has to travel
through a separate channel.

```bash
curl -X POST http://<orchestrator-host>:8000/resume \
  -H "Content-Type: application/json" \
  -d '{"gameweek": 12, "decision": "approve"}'
```

Anything other than the literal string `"approve"` is logged as rejected -
ARCHITECTURE.md 6c: "Rejection/timeout: logged as declined, no auto-retry
or negotiation loop." There is no partial-approval or "reconsider with
feedback" flow (see ARCHITECTURE.md 12's future-extensions note); it's a
one-shot approve/reject exactly as scoped.

`scripts/approve_gameweek.py` wraps the same call as a short CLI, for
whoever finds a raw curl command less convenient.
