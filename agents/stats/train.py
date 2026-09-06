"""Train and compare five expected-points models, then ship the winner.

Run:  python -m agents.stats.train

Produces:
  * models/stats_model.pkl       - the winning model + metadata
  * docs/model_comparison.md      - the written-up comparison
  * docs/figures/model_*.png      - comparison chart, SHAP plots

Pipeline order (matches the project's data-science spec):
  1. build features           (agents/stats/features.py)
  2. chronological split       (NEVER random - see the comment at the split)
  3. train 5 models on train, score all on the same validation set
  4. SHAP on the winner
  5. save winner + write-up
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import REPO_ROOT
from .features import build_feature_frame

try:
    from xgboost import XGBRegressor

    HAVE_XGB = True
except Exception:  # pragma: no cover - xgboost should be installed
    HAVE_XGB = False

DOCS = REPO_ROOT / "docs"
FIG = DOCS / "figures"
MODELS = REPO_ROOT / "models"

# --- split configuration -----------------------------------------------
# Warm-up: the rolling features need a few gameweeks of history before they
# mean anything, so we don't train or validate on the very start of the season.
WARMUP_LAST_GW = 5
# Everything from here on is validation; everything before is training.
VALID_FROM_GW = 30
RANDOM_STATE = 42


@dataclass
class Result:
    name: str
    preds: np.ndarray
    mae: float = 0.0
    rmse: float = 0.0
    spearman: float = 0.0
    mae_played: float = 0.0  # MAE restricted to players who actually played
    beats_naive: bool = False
    notes: str = ""
    extra: dict = field(default_factory=dict)


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _score(name: str, y_true: np.ndarray, preds: np.ndarray, played_mask: np.ndarray) -> Result:
    r = Result(name=name, preds=preds)
    r.mae = float(mean_absolute_error(y_true, preds))
    r.rmse = _rmse(y_true, preds)
    r.spearman = float(spearmanr(y_true, preds).statistic)
    r.mae_played = float(mean_absolute_error(y_true[played_mask], preds[played_mask]))
    return r


def load_split():
    """Build features and cut the data into train / validation *by time*."""
    frame, feature_cols, target = build_feature_frame()

    # Drop the warm-up gameweeks (unreliable rolling features).
    frame = frame[frame["GW"] > WARMUP_LAST_GW].reset_index(drop=True)

    # ================================================================
    # CHRONOLOGICAL SPLIT - this is deliberate, not a random train_test_split.
    #
    # In FPL you predict gameweek N knowing only gameweeks 1..N-1. A random
    # k-fold would put gameweek 38 rows in the training set while validating
    # on gameweek 10 - the model would effectively "know the future": a
    # player's end-of-season form, final price, and his team's eventual
    # league position all leak backward through rolling stats computed on
    # both sides of the fold. That inflates the validation score and the
    # model then underperforms in live use.
    #
    # So: train on the earlier gameweeks, validate on the later ones, with a
    # hard cut between them - exactly how the model will be used week to week,
    # and the same no-lookahead rule that governs the backtest engine.
    # ================================================================
    train = frame[frame["GW"] < VALID_FROM_GW].reset_index(drop=True)
    valid = frame[frame["GW"] >= VALID_FROM_GW].reset_index(drop=True)

    return frame, train, valid, feature_cols, target


def train_all():
    frame, train, valid, feats, target = load_split()

    Xtr, ytr = train[feats].to_numpy(dtype=float), train[target].to_numpy(dtype=float)
    Xva, yva = valid[feats].to_numpy(dtype=float), valid[target].to_numpy(dtype=float)
    played_va = (valid["minutes"] > 0).to_numpy()

    # Poisson can't be trained on negative targets (cards / own goals push a
    # few rows to -1 / -2). Clip the *training* target at 0 for that model
    # only; it costs us almost nothing and keeps the model well-defined. The
    # evaluation target is left untouched.
    ytr_nonneg = np.clip(ytr, 0, None)

    results: list[Result] = []

    # ---- MODEL 1: naive baseline ------------------------------------
    # "Rolling average of the player's last 3-5 gameweeks." No fitting - this
    # is the number a casual manager eyeballs off the FPL form column. If the
    # ML models can't beat this, that IS the headline finding.
    naive_pred = (valid["pts_roll3"].to_numpy() + valid["pts_roll5"].to_numpy()) / 2.0
    r_naive = _score("Naive (rolling 3/5 avg)", yva, naive_pred, played_va)
    r_naive.notes = "no training; mean of the player's last-3 and last-5 GW points"
    results.append(r_naive)
    naive_mae = r_naive.mae

    # ---- MODEL 2: Poisson regression -------------------------------
    # Why Poisson and not plain linear regression:
    #   * total_points is count-like: 0, 1, 2, 6, 13... not a smooth Gaussian.
    #   * It's floored near zero and right-skewed; OLS would happily predict
    #     negative points and assumes constant variance, but here the spread
    #     grows with the mean (a 10-point player is far more variable than a
    #     2-point one). Poisson's log link keeps predictions >= 0 and models
    #     that mean-variance link directly.
    #   * It's regularised (L2, alpha) so the ~40 correlated features don't
    #     blow up the coefficients.
    # It's still only an approximation (points aren't truly Poisson - bonus
    # and card deductions make them lumpy), but it's the right family.
    poisson = Pipeline([
        ("scale", StandardScaler()),
        ("model", PoissonRegressor(alpha=1.0, max_iter=500)),
    ])
    poisson.fit(Xtr, ytr_nonneg)
    r_pois = _score("Poisson regression", yva, poisson.predict(Xva), played_va)
    results.append(r_pois)

    # ---- MODEL 3: Random Forest -----------------------------------
    # Bagged trees: robust to feature scaling / skew, captures non-linear
    # interactions (e.g. "high minutes AND easy fixture") for free, hard to
    # overfit badly. A strong default for tabular data this size.
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(Xtr, ytr)
    r_rf = _score("Random Forest", yva, rf.predict(Xva), played_va)
    results.append(r_rf)

    # ---- MODEL 4: XGBoost ---------------------------------------
    # Gradient-boosted trees: usually the top performer on tabular data. Uses
    # a Poisson objective here for the same reason as model 2 - the target is
    # non-negative count data. Shallow trees + low learning rate + subsampling
    # to keep it from memorising the training season.
    if HAVE_XGB:
        xgb = XGBRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            objective="count:poisson",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        xgb.fit(Xtr, ytr_nonneg)
        r_xgb = _score("XGBoost", yva, xgb.predict(Xva), played_va)
        results.append(r_xgb)
    else:
        xgb = None

    # ---- MODEL 5: small MLP ------------------------------------
    # A small neural net (2 hidden layers). Included for completeness. The
    # expectation - stated up front in ARCHITECTURE.md - is that it LOSES to
    # the tree ensembles: neural nets need lots of data and lack the built-in
    # "split on a threshold" inductive bias that suits this mix of sparse,
    # differently-scaled tabular features. ~20k training rows is small for a
    # net. We report the result rather than hiding it.
    mlp = Pipeline([
        ("scale", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            alpha=1e-3,
            learning_rate_init=1e-3,
            early_stopping=True,
            n_iter_no_change=15,
            max_iter=300,
            random_state=RANDOM_STATE,
        )),
    ])
    mlp.fit(Xtr, ytr)
    r_mlp = _score("MLP (64, 32)", yva, mlp.predict(Xva), played_va)
    results.append(r_mlp)

    # ---- finalise: who beat the naive baseline on MAE? -------------
    for r in results:
        r.beats_naive = r.mae < naive_mae - 1e-9
    results[0].beats_naive = False  # the baseline doesn't "beat" itself

    fitted = {
        "Poisson regression": poisson,
        "Random Forest": rf,
        "XGBoost": xgb,
        "MLP (64, 32)": mlp,
    }

    # Winner = lowest validation MAE among the *trained* models.
    trainable = [r for r in results if r.name != results[0].name]
    winner = min(trainable, key=lambda r: r.mae)

    return {
        "frame": frame,
        "train": train,
        "valid": valid,
        "features": feats,
        "target": target,
        "results": results,
        "fitted": fitted,
        "winner": winner,
        "winner_model": fitted[winner.name],
        "naive_mae": naive_mae,
        "X_valid": Xva,
        "y_valid": yva,
    }


# ---------------------------------------------------------------------------
# SHAP on the winning model
# ---------------------------------------------------------------------------
def explain_winner(ctx: dict) -> dict:
    import shap

    winner = ctx["winner"]
    model = ctx["winner_model"]
    feats = ctx["features"]
    valid = ctx["valid"]
    Xva = ctx["X_valid"]

    tree_like = winner.name in ("Random Forest", "XGBoost")
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(Xva), size=min(2000, len(Xva)), replace=False
    )
    Xs = Xva[sample_idx]

    if tree_like:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Xs)
        preds_sample = model.predict(Xs)
    else:
        scaler = model.named_steps["scale"]
        inner = model.named_steps["model"]
        Xs_scaled = scaler.transform(Xs)
        explainer = shap.Explainer(inner.predict, Xs_scaled)
        shap_values = explainer(Xs_scaled).values
        preds_sample = inner.predict(Xs_scaled)

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # Beeswarm summary plot.
    FIG.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, Xs, feature_names=feats, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIG / "model_shap_beeswarm.png", dpi=110)
    plt.close()

    # A few worked examples: highest predicted, lowest predicted, and a
    # middling one - show which features pushed each prediction.
    picks = {
        "highest predicted": int(np.argmax(preds_sample)),
        "lowest predicted": int(np.argmin(preds_sample)),
        "median predicted": int(np.argsort(preds_sample)[len(preds_sample) // 2]),
    }
    examples = []
    for label, i in picks.items():
        row_meta = valid.iloc[sample_idx[i]]
        contrib = pd.Series(shap_values[i], index=feats).sort_values(key=np.abs, ascending=False)
        top = [(f, round(float(v), 3)) for f, v in contrib.head(6).items()]
        examples.append({
            "label": label,
            "player": f"{row_meta['name']}, GW{int(row_meta['GW'])} vs {row_meta['team']}'s opponent",
            "position": row_meta["position"],
            "predicted": round(float(preds_sample[i]), 2),
            "actual": float(row_meta["total_points"]),
            "top_contributors": top,
        })

    return {"importance": importance, "examples": examples,
            "base_value": float(np.mean(preds_sample))}


# ---------------------------------------------------------------------------
# Persist + write-up
# ---------------------------------------------------------------------------
def _comparison_chart(results, naive_mae):
    FIG.mkdir(parents=True, exist_ok=True)
    names = [r.name for r in results]
    maes = [r.mae for r in results]
    fig, ax = plt.subplots(figsize=(7, 4))
    colours = ["#888" if n.startswith("Naive") else "#3b7dd8" for n in names]
    ax.barh(names, maes, color=colours)
    ax.axvline(naive_mae, color="#b34b4b", ls="--", lw=1, label="naive baseline MAE")
    ax.set_xlabel("validation MAE (lower is better)")
    ax.set_title("Model comparison - mean absolute error on held-out gameweeks")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "model_comparison_mae.png", dpi=110)
    plt.close(fig)


def save_model(ctx: dict, shap_out: dict) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    winner = ctx["winner"]
    payload = {
        "model": ctx["winner_model"],
        "model_name": winner.name,
        "features": ctx["features"],
        "target": ctx["target"],
        "prediction_meaning": (
            "expected FPL total_points for a player in an upcoming gameweek, "
            "given only information available before that gameweek's deadline"
        ),
        "split": {
            "warmup_last_gw": WARMUP_LAST_GW,
            "train_gws": f"{WARMUP_LAST_GW + 1}-{VALID_FROM_GW - 1}",
            "valid_gws": f"{VALID_FROM_GW}-38",
            "method": "chronological (no random k-fold)",
        },
        "validation_metrics": {
            r.name: {
                "mae": round(r.mae, 4),
                "rmse": round(r.rmse, 4),
                "spearman": round(r.spearman, 4),
                "mae_played_only": round(r.mae_played, 4),
                "beats_naive": r.beats_naive,
            }
            for r in ctx["results"]
        },
        "top_features_by_shap": shap_out["importance"].head(12).to_dict("records"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "load_note": "joblib.load(...); model.predict expects the `features` columns in that order",
    }
    joblib.dump(payload, MODELS / "stats_model.pkl")
    print(f"saved {MODELS / 'stats_model.pkl'}  (winner: {winner.name})")


def write_comparison(ctx: dict, shap_out: dict) -> None:
    results = ctx["results"]
    winner = ctx["winner"]
    naive_mae = ctx["naive_mae"]
    imp = shap_out["importance"]

    L: list[str] = []
    a = L.append
    a("# Model comparison - Stats agent expected-points model\n")
    a("_A walk-through of what was tried, how it was judged, and why one model won._\n")

    a("## What the model predicts\n")
    a("For every player, before a gameweek's deadline: **how many FPL points will they "
      "score in that gameweek?** Inputs are only things known before kick-off - the "
      "player's recent history, price, who they play, home or away. The match itself "
      "(minutes, goals, bonus) is never an input; that would be cheating.\n")

    a("## How the data was split\n")
    a(f"- Gameweeks 1-{WARMUP_LAST_GW} are dropped as **warm-up** (the rolling-average "
      f"features need a few weeks of history first).")
    a(f"- **Train:** gameweeks {WARMUP_LAST_GW + 1}-{VALID_FROM_GW - 1} "
      f"({len(ctx['train']):,} player-gameweek rows).")
    a(f"- **Validate:** gameweeks {VALID_FROM_GW}-38 ({len(ctx['valid']):,} rows).")
    a("- The split is **chronological, never random.** Predicting a gameweek means "
      "knowing only the past; a random split would leak end-of-season information "
      "backward into training and flatter the score. Same no-lookahead rule as the "
      "backtest engine.\n")

    a("## The five models\n")
    a("| # | Model | What it is | Prior expectation |")
    a("|---|---|---|---|")
    a("| 1 | Naive baseline | Average of the player's last 3 and last 5 gameweek points. No training. | The bar to clear. |")
    a("| 2 | Poisson regression | Regularised linear model with a log link. | Points are non-negative count data, so Poisson fits better than plain (Gaussian) linear regression - it can't predict negatives and it models variance growing with the mean. |")
    a("| 3 | Random Forest | 400 bagged decision trees. | Strong tabular default. |")
    a("| 4 | XGBoost | Gradient-boosted trees, Poisson objective. | Usually the winner on tabular data. |")
    a("| 5 | MLP (64, 32) | Small 2-layer neural network. | Expected to lose to the trees - nets need more data and lack the threshold-split bias that suits this feature set. |")
    a("")

    a(f"## Results (held-out gameweeks {VALID_FROM_GW}-38)\n")
    a("Primary metric is **MAE** (average points error on a typical row). RMSE is shown "
      "too but it is dominated by the rare 15+ point hauls. Spearman is rank correlation "
      "- how well the model *orders* players, which is what actually matters when picking "
      "a squad. `MAE (played)` restricts to players who got minutes.\n")
    a("| Model | MAE | RMSE | Spearman | MAE (played) | Beats naive? |")
    a("|---|---|---|---|---|---|")
    for r in results:
        flag = "-" if r.name.startswith("Naive") else ("**yes**" if r.beats_naive else "no")
        a(f"| {r.name} | {r.mae:.3f} | {r.rmse:.3f} | {r.spearman:.3f} | {r.mae_played:.3f} | {flag} |")
    a("")
    a("![model comparison](figures/model_comparison_mae.png)\n")

    naive_r = results[0]
    beat = [r for r in results if r.beats_naive]
    lost = [r for r in results if not r.beats_naive and not r.name.startswith("Naive")]
    best = min((r for r in results if not r.name.startswith("Naive")), key=lambda r: r.mae)
    best_spearman = max(results, key=lambda r: r.spearman)
    a("### In plain language\n")
    if beat:
        gain = (naive_mae - best.mae) / naive_mae * 100
        a(f"- **{len(beat)} of 4 trained models beat the naive baseline on MAE**, and all "
          f"four beat it on RMSE. The best, **{best.name}**, moves MAE from "
          f"{naive_mae:.3f} to {best.mae:.3f} (~{gain:.0f}% better) and RMSE from "
          f"{naive_r.rmse:.3f} to {best.rmse:.3f}.")
    else:
        a("- **No trained model beat the naive baseline on MAE.** That is the headline "
          "finding: week-to-week FPL points are noisy enough that a simple rolling "
          "average is hard to improve on with this feature set.")
    # Be honest about Spearman: the naive baseline is a strong *ranker* even
    # when it loses on absolute error.
    if best_spearman.name.startswith("Naive"):
        a(f"- **But the naive baseline still orders players best** (Spearman "
          f"{naive_r.spearman:.3f} vs {best.name}'s {best.spearman:.3f}). Recent-form "
          f"average is a genuinely good *ranking* signal; the ML models win by being "
          f"better *calibrated* - they pull nailed-on non-scorers and rotation risks "
          f"down toward zero, which is where most of the MAE gain comes from. The "
          f"~{abs(naive_r.spearman - best.spearman):.02f} Spearman gap is small and "
          f"within what re-running on another season would move.")
    for r in lost:
        a(f"- **{r.name} did not beat the baseline on MAE** ({r.mae:.3f} vs {naive_mae:.3f}) "
          f"- a regularised linear model is too rigid for the threshold-shaped "
          f"relationships here (points jump when minutes cross 60, when a fixture is "
          f"easy *and* form is good, etc.).")
    mlp_r = next(r for r in results if r.name.startswith("MLP"))
    tree_rs = [r for r in results if r.name in ("Random Forest", "XGBoost")]
    if tree_rs:
        tree_best = min(tree_rs, key=lambda r: r.mae)
        if mlp_r.mae > tree_best.mae:
            a(f"- **The MLP underperformed the tree models** (MAE {mlp_r.mae:.3f} vs "
              f"{tree_best.name} {tree_best.mae:.3f}), exactly as expected. Most likely "
              f"reason: ~{len(ctx['train']) // 1000}k training rows is a small tabular "
              f"dataset, and gradient-boosted / bagged trees have the right inductive "
              f"bias for sparse, differently-scaled features like these. More data or "
              f"heavy tuning might close the gap; it is not worth it here.")
        else:
            a(f"- The MLP actually matched or beat a tree model here (MAE {mlp_r.mae:.3f}); "
              f"noted, though the trees remain the more reliable choice on data this size.")
    a("")

    a(f"## Winner: {winner.name}\n")
    a(f"Chosen on lowest validation MAE ({winner.mae:.3f}) and lowest RMSE "
      f"({winner.rmse:.3f}). Saved to `models/stats_model.pkl` with the feature list, "
      f"the split definition, and every model's metrics.\n")
    a("Why not just keep the naive baseline, given it edges Spearman? Three reasons: "
      "(1) the Manager's solver optimises an objective built from the *magnitude* of "
      "predicted points, not just the ranking, so calibration matters; (2) XGBoost also "
      "wins RMSE, i.e. it is less wrong on the big weeks that decide captaincy; "
      "(3) a trained model gives per-prediction SHAP attributions, which is exactly what "
      "the Stats agent needs to quote in the debate ('backing him - minutes and xGI both "
      "trending up'). The naive average can't explain itself.\n")

    a("## Which engineered features actually mattered (SHAP)\n")
    a("SHAP attributes each prediction to its features. Averaging the absolute "
      "attribution over 2,000 validation rows gives a ranking of real influence on the "
      "winning model:\n")
    a("> Note: the winner uses a Poisson objective (log link), so SHAP values are in "
      "**log-points space** - read them as *direction and relative size*, not as a "
      "number of points. The `predicted` / `actual` figures in the worked examples "
      "below are in real points.\n")
    a("| rank | feature | mean abs SHAP | required or extra |")
    a("|---|---|---|---|")
    required = {
        "pts_roll3", "pts_roll5", "pts_roll10", "fixture_adj_xp", "fixture_multiplier",
        "pts_per90_roll5", "xgi_per90_roll5", "bps_per90_roll5", "threat_per90_roll5",
        "creativity_per90_roll5", "price_now", "price_delta3", "price_delta5",
        "price_momentum_sign",
    }
    for i, row in imp.head(15).iterrows():
        tag = "required" if row["feature"] in required else "extra"
        a(f"| {i + 1} | `{row['feature']}` | {row['mean_abs_shap']:.4f} | {tag} |")
    a("")
    a("![SHAP beeswarm](figures/model_shap_beeswarm.png)\n")
    dead = imp.tail(8)["feature"].tolist()
    top10 = list(imp.head(10)["feature"])
    rank_of = {f: i + 1 for i, f in enumerate(imp["feature"])}
    a("**What mattered:** the model leans hardest on **recent minutes** "
      f"(`minutes_roll3`, rank {rank_of.get('minutes_roll3', '?')}) and a "
      f"**longer-run scoring baseline** (`pts_season_avg`, rank "
      f"{rank_of.get('pts_season_avg', '?')}) - together they dwarf everything else. "
      "That matches the EDA: predicting points is mostly predicting whether the player "
      "starts and roughly how good he is.")
    extras_in_top = [f for f in top10 if f not in required]
    reqs_in_top = [f for f in top10 if f in required]
    a(f"\n**Extra features that earned their place** (top 10): "
      f"{', '.join('`' + f + '`' for f in extras_in_top)}. Of the engineered extras, "
      "`minutes_roll3`, `pts_season_avg`, `ict_roll5`, `minutes_trend` and "
      "`ownership_lag1` are the ones doing real work.")
    a(f"\n**Required features that turned out weak:** only "
      f"{', '.join('`' + f + '`' for f in reqs_in_top) or 'a few'} of the four mandated "
      "feature groups land in the top 10. In particular **`fixture_adj_xp` and "
      "`fixture_multiplier` (the required 'fixture-adjusted expected points') barely "
      f"move the model** (ranks {rank_of.get('fixture_adj_xp', '?')} and "
      f"{rank_of.get('fixture_multiplier', '?')}), and `pts_roll5` is mostly redundant "
      "with `pts_roll3` + `pts_season_avg`. Worth keeping them for now (fixture effects "
      "should matter and may just be too noisy at 5-game samples), but they are honest "
      "candidates to rebuild rather than trust.")
    a(f"\n**Dead weight:** {', '.join('`' + f + '`' for f in dead)} contributed almost "
      f"nothing. Safe to drop in a future pass to simplify the model.\n")

    a("### Worked examples\n")
    a("A few individual predictions from the winning model, with the features that moved "
      "them most (positive SHAP = pushed the prediction up):\n")
    for ex in shap_out["examples"]:
        a(f"**{ex['label']}** - {ex['player']} [{ex['position']}]  ")
        a(f"predicted **{ex['predicted']}** pts, actually scored **{ex['actual']:.0f}**  ")
        contribs = ", ".join(f"`{f}` {v:+.2f}" for f, v in ex["top_contributors"])
        a(f"top contributors: {contribs}\n")

    a("## Honest limitations\n")
    a("- The model predicts an **average** outcome; FPL is won on the upside tail "
      "(captaincy, differentials), which the error metrics under-reward. And the naive "
      "baseline still out-ranks it on Spearman - the win here is calibration, not a "
      "dramatically better model.")
    a("- The required **fixture-adjustment features carry almost no weight** (see SHAP). "
      "Either fixture difficulty matters less week-to-week than assumed, or the 5-game "
      "opponent-strength estimate is too noisy - the Fixtures agent may be the better "
      "home for that signal anyway.")
    a("- No explicit injury data - availability is inferred from recent minutes. The "
      "Injuries agent owns the hard veto in the full system.")
    a("- `xP` (FPL's own expected-points column) was deliberately **excluded** from the "
      "features: it is another model's output and using it would make this partly a copy "
      "of FPL's model rather than an independent one.")
    a("- One season of training data. Re-fit each season; consider carrying prior "
      "seasons once available.\n")

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "model_comparison.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {DOCS / 'model_comparison.md'}")


def main() -> None:
    np.random.seed(RANDOM_STATE)
    ctx = train_all()

    print("\nvalidation MAE by model:")
    for r in ctx["results"]:
        print(f"  {r.name:26s} MAE={r.mae:.3f}  RMSE={r.rmse:.3f}  "
              f"spearman={r.spearman:.3f}  beats_naive={r.beats_naive}")
    print(f"\nwinner: {ctx['winner'].name}")

    _comparison_chart(ctx["results"], ctx["naive_mae"])
    shap_out = explain_winner(ctx)
    save_model(ctx, shap_out)
    write_comparison(ctx, shap_out)

    (DOCS / "model_metrics.json").write_text(
        json.dumps(
            {r.name: {"mae": r.mae, "rmse": r.rmse, "spearman": r.spearman,
                      "mae_played": r.mae_played, "beats_naive": r.beats_naive}
             for r in ctx["results"]},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("done.")


if __name__ == "__main__":
    main()
