"""TODO.md Phase 2's last item: run a full-season backtest and write up
results. Not a test - a script, run once to produce the actual numbers for
the portfolio writeup (ARCHITECTURE.md 9 step 3's "key artifact"). Prints a
plain-text report; nothing here is asserted or checked, it's data
collection.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .benchmarks import never_transfer_total
from .engine import DEFAULT_START_GW, run_season


def main(season: str = "2025-26", start_gw: int = DEFAULT_START_GW, end_gw: int = 38) -> None:
    t0 = time.time()
    print(f"=== Baseline run: {season} GW{start_gw}-{end_gw} ===")
    baseline = run_season(season, start_gw, end_gw)
    print(f"baseline total points: {baseline.total_points:.1f} ({time.time() - t0:.0f}s)")

    t1 = time.time()
    never_transfer = never_transfer_total(season, start_gw, end_gw)
    print(f"never-transfer baseline: {never_transfer:.1f} ({time.time() - t1:.0f}s)")

    ablation_totals: dict[str, float] = {}
    for agent in ("fixtures", "contrarian", "template", "chips", "news"):
        t2 = time.time()
        report = run_season(season, start_gw, end_gw, ablate_agent=agent)
        ablation_totals[agent] = report.total_points
        delta = baseline.total_points - report.total_points
        print(f"ablate {agent}: {report.total_points:.1f} (delta vs baseline: {delta:+.1f}) ({time.time() - t2:.0f}s)")

    summary = {
        "season": season, "start_gw": start_gw, "end_gw": end_gw,
        "baseline_total_points": baseline.total_points,
        "never_transfer_total_points": never_transfer,
        "average_manager_total_points": None,
        "average_manager_unavailable_reason": (
            "not available for a completed season - see backtest/benchmarks.py"
        ),
        "ablation_totals": ablation_totals,
        "ablation_deltas": {k: baseline.total_points - v for k, v in ablation_totals.items()},
        "net_points_by_gw": baseline.net_points_by_gw,
        "final_bank": baseline.final_state.bank,
        "elapsed_seconds": time.time() - t0,
    }
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"Total elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
