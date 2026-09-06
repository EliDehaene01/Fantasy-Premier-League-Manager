"""The 'numbers' half of the Stats agent.

Everything that must be **reproducible, testable and cheap** lives here:
loading the trained model, scoring a player pool, ranking, deriving
conviction, and pulling the SHAP factors behind each pick.

This module never calls an LLM. That separation is deliberate - see the note
in ``service.py`` - because an LLM is none of "reproducible, testable, cheap"
and must never be the thing that decides which players to recommend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import shap
from sklearn.pipeline import Pipeline

from .data import REPO_ROOT

MODEL_PATH = REPO_ROOT / "models" / "stats_model.pkl"

# How many SHAP factors to attach to each recommendation. 3 is enough to
# write a specific sentence without the prompt becoming a data dump.
TOP_FACTORS_PER_PLAYER = 3

# Human-readable phrasing for the model's feature names, used both in the LLM
# prompt and in the offline fallback sentence. Anything not listed here falls
# back to a prettified version of the raw name.
FEATURE_PHRASES: dict[str, str] = {
    "pts_roll3": "points over the last 3 gameweeks",
    "pts_roll5": "points over the last 5 gameweeks",
    "pts_roll10": "points over the last 10 gameweeks",
    "fixture_adj_xp": "fixture-adjusted recent form",
    "fixture_multiplier": "how favourable the fixture is",
    "pts_per90_roll5": "points per 90 minutes",
    "xgi_per90_roll5": "expected goal involvements per 90",
    "bps_per90_roll5": "all-round match involvement (BPS) per 90",
    "threat_per90_roll5": "goal threat per 90",
    "creativity_per90_roll5": "chance creation per 90",
    "price_now": "price",
    "price_delta3": "price movement over the last 3 gameweeks",
    "price_delta5": "price movement over the last 5 gameweeks",
    "price_momentum_sign": "price trend direction",
    "is_home": "playing at home",
    "rest_days": "rest since the last match",
    "minutes_roll3": "minutes over the last 3 gameweeks",
    "minutes_roll5": "minutes over the last 5 gameweeks",
    "start_rate_roll5": "how often he has started recently",
    "minutes_trend": "his minutes trend",
    "is_double_gw": "a double gameweek",
    "bonus_roll5": "recent bonus points",
    "ict_roll5": "recent ICT index",
    "opp_def_conceded_roll5": "the opponent's recent goals conceded",
    "opp_def_xgc_roll5": "the opponent's expected goals conceded",
    "opp_attack_roll5": "the opponent's attacking output",
    "team_attack_roll5": "his team's attacking form",
    "team_cs_rate_roll5": "his team's clean-sheet rate",
    "team_xgc_roll5": "his team's defensive solidity",
    "conceded_roll5": "goals conceded while he has been on the pitch",
    "ownership_lag1": "his ownership",
    "transfer_balance_lag1": "net transfer momentum",
    "pts_season_avg": "season points-per-game",
    "games_played": "how much history we have on him",
    "pos_DEF": "being a defender",
    "pos_FWD": "being a forward",
    "pos_GK": "being a goalkeeper",
    "pos_MID": "being a midfielder",
}


def _phrase(feature: str) -> str:
    return FEATURE_PHRASES.get(feature, feature.replace("_", " "))


@dataclass
class Factor:
    """One SHAP-derived reason behind a single player's prediction."""

    feature: str
    phrase: str          # human-readable version of `feature`
    direction: str       # "raises" or "lowers" the prediction
    shap_value: float    # signed; magnitude = how much it moved the prediction


@dataclass
class Pick:
    """A scored, ranked player - the model's full opinion on one player."""

    player_id: int
    name: str | None
    position: str | None
    predicted_points: float
    conviction: float
    factors: list[Factor]


class StatsModel:
    """Loads ``stats_model.pkl`` once and scores player pools with it.

    Instantiate this exactly once (at service startup) and reuse it. Building
    the SHAP explainer is not free, and the model is stateless across
    requests, so there is no reason to reload per call.
    """

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features: list[str] = list(bundle["features"])
        self.model_name: str = bundle.get("model_name", type(self.model).__name__)

        # Unwrap a sklearn Pipeline so SHAP sees the bare estimator; keep the
        # preprocessing step so we can transform inputs the same way.
        if isinstance(self.model, Pipeline):
            self._pre = self.model[:-1]
            estimator = self.model[-1]
        else:
            self._pre = None
            estimator = self.model

        # TreeExplainer is exact and fast for the tree models (RF / XGBoost),
        # which is what the winner is. If the winner were ever a linear / MLP
        # model this would raise, and we fall back to no per-player factors.
        try:
            self._explainer = shap.TreeExplainer(estimator)
        except Exception:  # pragma: no cover - only hit if the winner changes type
            self._explainer = None

    # ------------------------------------------------------------------
    def _matrix(self, players) -> np.ndarray:
        """Turn the incoming player feature dicts into the model's input
        matrix, in the exact column order the model was trained on. A missing
        feature becomes 0.0 - the same thing training did with ``fillna(0.0)``.
        """
        return np.array(
            [[float(p.features.get(f, 0.0)) for f in self.features] for p in players],
            dtype=float,
        )

    def _factors_for_row(self, shap_row: np.ndarray) -> list[Factor]:
        order = np.argsort(np.abs(shap_row))[::-1][:TOP_FACTORS_PER_PLAYER]
        out: list[Factor] = []
        for i in order:
            val = float(shap_row[i])
            if val == 0.0:
                continue
            feat = self.features[i]
            out.append(
                Factor(
                    feature=feat,
                    phrase=_phrase(feat),
                    direction="raises" if val > 0 else "lowers",
                    shap_value=val,
                )
            )
        return out

    # ------------------------------------------------------------------
    def score(self, players, top_k: int) -> list[Pick]:
        """Score every player, return the top ``top_k`` as ranked ``Pick``s.

        Ranking is by predicted points. ``conviction`` is then predicted
        points scaled against the strongest pick (see the comment in
        schemas.py).
        """
        if not players:
            return []

        X = self._matrix(players)
        preds = np.asarray(self.model.predict(X), dtype=float)

        # Rank by predicted points, keep the best `top_k`.
        top_k = min(top_k, len(players))
        top_idx = np.argsort(preds)[::-1][:top_k]

        top_preds = preds[top_idx]
        # Conviction is relative to the best pick. Floor negatives at 0 first
        # (the tree model with a Poisson objective won't produce them, but a
        # future model swap might).
        anchor = max(float(top_preds.max()), 0.0)

        # SHAP only for the handful of players we're actually recommending -
        # no point explaining the players we're not mentioning.
        if self._explainer is not None:
            X_top = self._pre.transform(X[top_idx]) if self._pre is not None else X[top_idx]
            shap_top = np.asarray(self._explainer.shap_values(X_top), dtype=float)
        else:
            shap_top = np.zeros((len(top_idx), len(self.features)))

        picks: list[Pick] = []
        for rank, idx in enumerate(top_idx):
            p = players[idx]
            pred = float(preds[idx])
            conviction = (max(pred, 0.0) / anchor) if anchor > 0 else 0.0
            picks.append(
                Pick(
                    player_id=p.player_id,
                    name=p.name,
                    position=p.position,
                    predicted_points=round(pred, 2),
                    conviction=round(min(max(conviction, 0.0), 1.0), 3),
                    factors=self._factors_for_row(shap_top[rank]),
                )
            )

        # Already in descending predicted-points order, which is also
        # descending conviction order; sort explicitly so the guarantee lives
        # in the code, not just implied by argsort.
        picks.sort(
            key=lambda pk: (pk.conviction, pk.predicted_points, -pk.player_id),
            reverse=True,
        )
        return picks
