# Backtest results: provenance and benchmark verification

Written in response to a specific request to verify, with evidence rather
than a restated claim, (1) which run the committed season-total number
actually is, and (2) whether the FPL average-manager benchmark
(ARCHITECTURE.md 9's second benchmark, alongside "never transfer") was
actually built or just asserted unavailable.

## 1. Is 2182 the corrected number, or the original confounded run?

**It is the corrected number.** Evidence, not just a claim:

- `git log --oneline --all -- backtest/results.json` returns exactly ONE
  commit: `9548f6b`. The file has never been committed more than once -
  there is no earlier, confounded version in history to have mixed up.
- That commit's message is literally titled *"fix: backtest ablation runs
  were confounded by shared predictions_log state"* and its body states
  the season was re-run with the fix applied before the file was written.
  The file was created BY that commit, as the direct output of the
  corrected code, not carried over from before it.
- The numbers in the committed file match the CORRECTED run's actual
  console output at the time (`baseline_total_points: 2182`,
  `ablate news: delta 0.0`), not the original confounded run's
  (`baseline: 2138`, `ablate news: delta -44.0` - see the same commit's
  message for the full before/after). News's ablation delta landing at
  *exactly* 0.0 is itself strong internal evidence of correctness: News
  contributes nothing to the debate in either configuration (ablated or
  not - see `data/backtest_seed.py`), so a confounded run that measures
  something other than "this agent's real contribution" would not
  reliably produce an exact zero by chance the way the fix's root-cause
  explanation predicts it should.

**2182 total points (GW2-38, 2025-26) is the number to use.** 2138 (the
original, confounded figure) never appears in any committed file and
should not be cited anywhere.

## 2. The FPL average-manager benchmark: not skipped, verified unavailable

Re-checked from scratch rather than trusting the earlier conclusion:

**Live API, checked directly** (`GET /api/bootstrap-static/`, today):
the `events` array currently being served is for **2026-27**, not
2025-26 - `events[0].deadline_time` is `2026-08-21T17:30:00Z`, and only
2026-27's own first four gameweeks are finished with real
`average_entry_score` values (50, 81, 51, 69). The live API has already
rolled over past 2025-26; its `average_entry_score` figures for that
season are gone, not merely unfetched.

**vaastav archive, checked at the repo root, not just the season
folder**: `data/` contains one directory per season plus exactly four
top-level files - `cleaned_merged_seasons.csv`,
`cleaned_merged_seasons_team_aggregated.csv`, `master_team_list.csv`,
`world_cup_2026.csv`. None of these, and nothing inside
`data/2025-26/` (`cleaned_players.csv`, `fixtures.csv`, `gws/`,
`player_idlist.csv`, `players/`, `players_raw.csv`, `teams.csv`), is a
per-gameweek or season-total average-entry-score file. The merged-seasons
files are player-level stat aggregates, not overall-manager averages.

**Conclusion**: this benchmark was not built because it cannot be built
from either data source this project uses - not an oversight, and not
worth a scraped one-off HTML page for a single historical number outside
any API this project otherwise depends on. `backtest/benchmarks.py`
already reflects this (`average_manager_total` returns `None` with the
reason attached); this document is the verification trail behind that
choice, checked again today rather than assumed still true.

## Numbers for the writeup

| | Total points, GW2-38, 2025-26 |
|---|---|
| This system | **2182** |
| Never-transfer baseline | 1398 |
| FPL average manager | not available (verified above) |

See `backtest/results.json` for the full per-gameweek and per-agent-ablation
breakdown.
