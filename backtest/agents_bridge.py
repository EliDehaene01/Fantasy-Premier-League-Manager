"""In-process callers wiring each specialist's REAL scoring logic to the
seeded backtest schema, matching ``orchestrator.graph.build_graph``'s
injectable ``agent_caller``/``manage_caller``/``react_caller`` signatures -
no HTTP, no live LLM (ARCHITECTURE.md 9: "guardrail calls stubbed... no
real external text in backtest mode"; this project's own established
convention, used throughout every agent's test suite, extends that to
reasoning/narration/reaction text too, disabled via each service's
``*_DISABLE_LLM`` env var - see this module's ``disable_all_llm_calls``).

Stats runs its actual trained model (``StatsModel``) against the
already-lagged feature frame. Fixtures/Contrarian/Template run their real
``scoring.rank_players`` against the seeded historical DB. Chips runs its
real ``scoring.evaluate_chip_timing``. News contributes nothing - see
``data/backtest_seed.py``'s module docstring for why that's a named
limitation, not a stub standing in for real logic.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from agents.chips import scoring as chips_scoring
from agents.contrarian import scoring as contrarian_scoring
from agents.fixtures import scoring as fixtures_scoring
from agents.stats.model_runtime import StatsModel
from agents.template import scoring as template_scoring
from shared.contracts import AgentArgument, ChipRecommendation, PlayerEntry, Recommendation

DISABLE_LLM_ENVS = [
    "STATS_AGENT_DISABLE_LLM", "FIXTURES_AGENT_DISABLE_LLM", "NEWS_AGENT_DISABLE_LLM",
    "CONTRARIAN_AGENT_DISABLE_LLM", "TEMPLATE_AGENT_DISABLE_LLM", "CHIPS_AGENT_DISABLE_LLM",
    "MANAGER_NARRATE_DISABLE_LLM", "MANAGER_REACT_DISABLE_LLM",
]


def disable_all_llm_calls() -> None:
    for env in DISABLE_LLM_ENVS:
        os.environ[env] = "1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_predictions(conn, gameweek: int, stats_argument: AgentArgument, model_version: str) -> None:
    """Writes Stats' predicted_points into predictions_log for this
    gameweek - Contrarian and Chips both read it back (see their own
    scoring.py modules), matching how the live system decouples them from
    a direct call to the Stats service.
    """
    rows = [
        (r.player_id, gameweek, r.predicted_points, model_version, _now())
        for r in stats_argument.recommendations
        if r.predicted_points is not None
    ]
    if rows:
        conn.executemany(
            "INSERT INTO predictions_log (player_id, gw, predicted_points, model_version, logged_at) VALUES (%s,%s,%s,%s,%s)",
            rows,
        )


def write_team_state(conn, gameweek: int, squad_ids: frozenset[int], bank: float) -> None:
    """Our squad, in the raw FPL entry-picks shape Chips' scoring.py reads
    (``[{"element": player_id}, ...]``) - see
    agents/chips/scoring.py::_our_squad_player_ids's docstring.
    """
    picks = [{"element": pid} for pid in sorted(squad_ids)]
    conn.execute(
        """INSERT INTO my_team_state (gw, picks, bank, updated_at)
           VALUES (%s,%s,%s,%s)
           ON CONFLICT (gw) DO UPDATE SET picks=EXCLUDED.picks, bank=EXCLUDED.bank, updated_at=EXCLUDED.updated_at""",
        (gameweek, json.dumps(picks), bank, _now()),
    )


class SpecialistBridge:
    """Callable matching ``orchestrator.graph.build_graph``'s
    ``agent_caller`` signature: ``(agent_name, gameweek, player_pool) -> AgentArgument``.
    """

    def __init__(self, conn, stats_model: StatsModel):
        self.conn = conn
        self.stats_model = stats_model

    def __call__(self, agent_name: str, gameweek: int, player_pool: list[dict]) -> AgentArgument:
        entries = [PlayerEntry.model_validate(p) for p in player_pool]

        if agent_name == "stats":
            picks = self.stats_model.score(entries, top_k=len(entries))
            recs = [
                Recommendation(player_id=p.player_id, conviction=p.conviction, predicted_points=p.predicted_points)
                for p in picks
            ]
            return AgentArgument(agent="stats", recommendations=recs, reasoning="offline model output (backtest)")

        if agent_name == "news":
            return AgentArgument(
                agent="news",
                reasoning="No historical availability data for backtest - News contributes nothing this gameweek (documented limitation, see data/backtest_seed.py).",
            )

        if agent_name == "chips":
            verdict = chips_scoring.evaluate_chip_timing(self.conn, gameweek)
            return AgentArgument(
                agent="chips",
                chip_recommendation=ChipRecommendation(
                    chip=verdict.chip, confidence=verdict.confidence,
                    reasoning="; ".join(verdict.reasoning_factors) or "no strong chip signal",
                ),
                reasoning="; ".join(verdict.reasoning_factors) or "no strong chip signal",
            )

        module = {"fixtures": fixtures_scoring, "contrarian": contrarian_scoring, "template": template_scoring}[agent_name]
        picks = module.rank_players(self.conn, gameweek, entries, top_k=len(entries))
        recs = [
            Recommendation(player_id=p.player_id, conviction=p.conviction, predicted_points=getattr(p, "predicted_points", None))
            for p in picks
        ]
        return AgentArgument(agent=agent_name, recommendations=recs, reasoning=f"{agent_name} scoring output (backtest)")
