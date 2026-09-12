"""Hyperparameter tuning for the winning model (XGBoost) from train.py.

Run:  python -m agents.stats.tune

Produces:
  * models/xgb_best_params.json     - the tuned hyperparameters ON THEIR OWN
                                       (or an explicit note that the untuned
                                       params are what's deployed), so the
                                       retraining pipeline (built next) can
                                       reuse them without re-searching every
                                       time it retrains on new data.
  * models/stats_model.pkl           - overwritten with the tuned model, but
                                       ONLY if it passes the selection rule
                                       below. If it doesn't, the existing
                                       artifact is left alone and that's
                                       reported plainly - not a failure.
  * docs/model_comparison.md         - the tuning section rewritten (not
                                       appended twice - see the idempotency
                                       note near the bottom of this file).
  * docs/figures/xgb_tuning_rmse.png - untuned vs tuned RMSE, side by side.

This reuses train.py's `train_all()` for the feature build + the exact same
chronological train/validation split - see the CHRONOLOGICAL TUNING comment
below for why that split discipline applies just as much to the *search*
as it does to the original train/validation split.

CORRECTED OBJECTIVE (v2 of this file)
--------------------------------------
The first version of this search optimized MAE, and picked the resulting
model because it improved MAE by ~4.5% - without checking what happened to
RMSE or Spearman. It turned out RMSE got ~9.6% WORSE and Spearman dropped
too: the search had found a model that's better at the boring middle of the
points distribution and worse at exactly the big-haul, captaincy-deciding
gameweeks RMSE is sensitive to (RMSE penalises large errors quadratically;
MAE treats every point of error the same). Since this model's whole purpose
is feeding a squad-selection solver where identifying big-haul candidates
matters more than shaving error off routine bench players, optimizing MAE
was optimizing the wrong thing. See `_objective` below for the fix, and
`_passes_selection_rule` for how the selection criterion was tightened to
catch this failure mode even if it recurs in a different shape.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from . import train as train_module
from .data import REPO_ROOT
from .train import RANDOM_STATE, VALID_FROM_GW, WARMUP_LAST_GW, XGB_UNTUNED_PARAMS, _rmse, _score

DOCS = REPO_ROOT / "docs"
FIG = DOCS / "figures"
MODELS = REPO_ROOT / "models"

# Optuna prints a line per trial by default; keep the console readable.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- search budget --------------------------------------------------------
# Kept modest on purpose: each trial fits XGBoost once per fold (see below),
# and this runs on a laptop, not a cluster. 30 trials x 3 folds = 90 fits,
# which finishes in a few minutes and is plenty for TPE to find a
# meaningfully better region of a ~9-dimensional space than the original
# hand-picked defaults.
N_TRIALS = 30

# --- forward-chaining fold configuration ----------------------------------
# UNCHANGED from the previous run - the fold construction was never the
# problem, only what the objective function measured on those folds was.
MIN_TRAIN_GWS = 10  # first fold needs at least this many GWs of history
VALID_WINDOW_GWS = 4  # each fold validates on this many subsequent GWs
MAX_FOLDS = 3

# --- selection-rule tolerances --------------------------------------------
# See `_passes_selection_rule` below. Both are set well BELOW the size of
# the regression the previous (MAE-optimized) run produced - RMSE ~9.6%
# worse, Spearman ~0.012 worse - specifically so that a repeat of that exact
# failure would still be rejected under this tolerance, not waved through.
RMSE_TOLERANCE_PCT = 0.5    # RMSE allowed to be up to 0.5% worse and still count as "not meaningfully worse"
SPEARMAN_TOLERANCE = 0.005  # Spearman allowed to be up to 0.005 worse and still count as "not meaningfully worse"


# ---------------------------------------------------------------------------
# CHRONOLOGICAL TUNING
# ---------------------------------------------------------------------------
# train.py's `load_split()` comment explains why the ONE outer split (train
# GW6-29, validate GW30-38) has to be chronological, not random: predicting
# gameweek N only ever uses information from gameweeks < N, so a random split
# would let end-of-season information leak backward and flatter the score.
#
# That exact same reasoning applies to hyperparameter tuning, and it is easy
# to miss - most hyperparameter-search tutorials (Optuna's own examples
# included) reach straight for `sklearn.model_selection.KFold` or a random
# train/validation split inside the objective function, because for i.i.d.
# data that's completely fine. Our rows are NOT i.i.d. in time: they carry
# rolling-window features computed from a player's *own* recent history, so
# a randomly-shuffled inner fold has exactly the same leakage problem as a
# randomly-shuffled outer split - a trial's "best" hyperparameters could just
# be the ones that best exploit hindsight, not the ones that generalise
# forward. Optuna's samplers make no assumption either way - they just
# optimise whatever number the objective function returns - so the
# discipline has to be built into the objective function ourselves. Hence:
# no `KFold`, no `cross_val_score`, no shuffling anywhere below.
#
# Concretely, we build FORWARD-CHAINING (expanding-window) folds, entirely
# INSIDE the original training pool (GW6-29) - never touching the GW30-38
# holdout the rest of the model comparison uses:
#
#   fold 1: train on GW 6-15,  validate on GW 16-19
#   fold 2: train on GW 6-19,  validate on GW 20-23
#   fold 3: train on GW 6-23,  validate on GW 24-27 (or however many remain)
#
# Each fold's validation gameweeks are strictly later than everything in its
# training window, exactly mirroring how the model is actually used week to
# week. Reserving the real GW30-38 holdout for the FINAL comparison (rather
# than also handing it to Optuna) matters for a second, related reason: if
# the search saw that exact set during tuning, "beats the untuned model on
# GW30-38" would partly just mean "we picked hyperparameters that fit GW30-38
# a bit better," which is a subtler version of the same leakage problem -
# tuning to the test set instead of the training set.
def _forward_chaining_folds(unique_gws: list[int]) -> list[tuple[set[int], set[int]]]:
    """Expanding-window chronological folds, built from GAMEWEEK ORDER (not
    raw GW arithmetic, in case of gaps) so this stays correct even if a
    blank/postponed gameweek ever breaks numeric contiguity."""
    folds: list[tuple[set[int], set[int]]] = []
    train_end = MIN_TRAIN_GWS
    while len(folds) < MAX_FOLDS:
        valid_slice = unique_gws[train_end: train_end + VALID_WINDOW_GWS]
        if len(valid_slice) < 2:  # not enough gameweeks left for a meaningful check
            break
        folds.append((set(unique_gws[:train_end]), set(valid_slice)))
        train_end += VALID_WINDOW_GWS
    return folds


def _fold_scores(params: dict, train_frame, feats: list[str], target: str, folds) -> dict:
    """Fit one hyperparameter set on each chronological fold, return the
    average RMSE, Spearman and MAE across folds.

    RMSE is what the caller should optimize (see `_objective`'s docstring
    for why); Spearman and MAE are computed here too so every trial's
    ranking quality is visible after the fact, not just the winning one's -
    that's what lets us check whether chasing RMSE is dragging Spearman down
    again, the same way chasing MAE did last time.
    """
    fold_rmse, fold_spearman, fold_mae = [], [], []
    for train_gws, valid_gws in folds:
        ftr = train_frame[train_frame["GW"].isin(train_gws)]
        fva = train_frame[train_frame["GW"].isin(valid_gws)]

        Xtr = ftr[feats].to_numpy(dtype=float)
        ytr = np.clip(ftr[target].to_numpy(dtype=float), 0, None)  # Poisson needs >= 0, see train.py
        Xva = fva[feats].to_numpy(dtype=float)
        yva = fva[target].to_numpy(dtype=float)

        model = XGBRegressor(
            **params, objective="count:poisson", random_state=RANDOM_STATE, n_jobs=-1
        )
        model.fit(Xtr, ytr)
        preds = model.predict(Xva)

        fold_rmse.append(_rmse(yva, preds))
        fold_spearman.append(float(spearmanr(yva, preds).statistic))
        fold_mae.append(mean_absolute_error(yva, preds))

    return {
        "rmse": float(np.mean(fold_rmse)),
        "spearman": float(np.mean(fold_spearman)),
        "mae": float(np.mean(fold_mae)),
    }


def _objective(trial: optuna.Trial, train_frame, feats: list[str], target: str, folds) -> float:
    """Optuna objective: average RMSE across the chronological folds.

    WHY RMSE AND NOT MAE: this model's predictions feed a squad-selection
    solver whose whole point is picking out big-haul / captaincy candidates,
    not just minimizing average week-to-week error. RMSE penalises large
    errors QUADRATICALLY (a 10-point miss costs 100x what a 1-point miss
    costs; MAE would charge it only 10x), so it is directly sensitive to
    whether the model gets the rare big hauls right - exactly the outcome
    that actually matters here. MAE rewards a model that's uniformly
    "pretty close" everywhere, including on the large majority of low-
    scoring, uninteresting rows, which is NOT what this system needs from
    its top picks. The previous version of this file optimized MAE and
    produced a model that was 4.5% better on MAE while being 9.6% WORSE on
    RMSE - a textbook case of optimizing a proxy metric instead of the one
    that reflects the actual downstream use. Don't swap this back to MAE
    without re-reading that history in docs/model_comparison.md.

    Spearman is tracked as a trial user attribute (not the objective) so it
    can be inspected afterwards without being what's being chased - see
    `tune_xgboost`'s diagnostic correlation check.
    """
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    scores = _fold_scores(params, train_frame, feats, target, folds)
    trial.set_user_attr("fold_spearman", scores["spearman"])
    trial.set_user_attr("fold_mae", scores["mae"])
    return scores["rmse"]  # <-- the objective Optuna minimizes


def _trial_rmse_spearman_tradeoff(study: optuna.Study) -> float:
    """Correlation, across every trial, between fold RMSE (the objective)
    and fold Spearman. Positive => trials with WORSE (higher) RMSE tended to
    have BETTER (higher) Spearman too, i.e. optimizing RMSE is dragging
    Spearman down as a side effect - the same failure mode as last time, in
    the other direction. Negative/near-zero => no such trade-off showed up
    across this search.
    """
    rmses = np.array([t.value for t in study.trials if t.value is not None])
    spearmans = np.array([
        t.user_attrs["fold_spearman"] for t in study.trials if t.value is not None
    ])
    if len(rmses) < 2:
        return 0.0
    return float(np.corrcoef(rmses, spearmans)[0, 1])


def tune_xgboost(ctx: dict) -> dict:
    """Search for better XGBoost hyperparameters, then fit and score ONE
    final model on the exact same (full train pool -> GW30-38 holdout) split
    train.py's comparison used, so the "tuned vs untuned" numbers are
    directly comparable to everything else in docs/model_comparison.md.
    """
    train_frame = ctx["train"]
    feats, target = ctx["features"], ctx["target"]

    unique_gws = sorted(train_frame["GW"].unique().tolist())
    folds = _forward_chaining_folds(unique_gws)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda t: _objective(t, train_frame, feats, target, folds),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )
    best_params = study.best_params
    rmse_spearman_corr = _trial_rmse_spearman_tradeoff(study)

    # Final fit: full original training pool (GW6-29), scored on the
    # untouched GW30-38 holdout - identical inputs to every other model in
    # train.py's comparison, via the same `_score` helper.
    Xtr, ytr_nonneg = ctx["train"][feats].to_numpy(dtype=float), np.clip(
        ctx["train"][target].to_numpy(dtype=float), 0, None
    )
    Xva, yva = ctx["X_valid"], ctx["y_valid"]
    played_va = (ctx["valid"]["minutes"] > 0).to_numpy()

    tuned_model = XGBRegressor(
        **best_params, objective="count:poisson", random_state=RANDOM_STATE, n_jobs=-1
    )
    tuned_model.fit(Xtr, ytr_nonneg)
    tuned_result = _score("XGBoost (tuned)", yva, tuned_model.predict(Xva), played_va)
    tuned_result.beats_naive = tuned_result.mae < ctx["naive_mae"] - 1e-9
    tuned_result.notes = f"Optuna TPE, {N_TRIALS} trials, {len(folds)} chronological folds, objective=RMSE"

    return {
        "study": study,
        "best_params": best_params,
        "folds": folds,
        "tuned_model": tuned_model,
        "tuned_result": tuned_result,
        "best_trial_fold_spearman": study.best_trial.user_attrs["fold_spearman"],
        "best_trial_fold_mae": study.best_trial.user_attrs["fold_mae"],
        "rmse_spearman_corr": rmse_spearman_corr,
    }


def _passes_selection_rule(tuned, untuned) -> tuple[bool, str]:
    """Decide whether the tuned model replaces the saved one.

    Rule: the tuned model wins only if it beats the untuned baseline on
    BOTH RMSE and Spearman, OR is within a small explicit tolerance on one
    of them while clearly beating the other. This is the direct fix for
    what went wrong last time - a model was promoted on a single metric
    (MAE) while regressing badly on both RMSE and Spearman. Requiring both
    metrics (with a little slack, since real noise exists) makes that
    specific failure impossible to repeat by construction.

    Returns (passed, human_readable_reason).
    """
    rmse_ok = tuned.rmse <= untuned.rmse
    rmse_within_tol = tuned.rmse <= untuned.rmse * (1 + RMSE_TOLERANCE_PCT / 100)
    rmse_clear_win = tuned.rmse < untuned.rmse * (1 - RMSE_TOLERANCE_PCT / 100)

    spearman_ok = tuned.spearman >= untuned.spearman
    spearman_within_tol = tuned.spearman >= untuned.spearman - SPEARMAN_TOLERANCE
    spearman_clear_win = tuned.spearman > untuned.spearman + SPEARMAN_TOLERANCE

    if rmse_ok and spearman_ok:
        return True, "tuned model beats (or matches) the untuned baseline on both RMSE and Spearman"
    if rmse_within_tol and spearman_clear_win:
        return True, (
            f"RMSE within the {RMSE_TOLERANCE_PCT}% tolerance "
            f"({tuned.rmse:.3f} vs {untuned.rmse:.3f}) while Spearman clearly improved "
            f"({tuned.spearman:.3f} vs {untuned.spearman:.3f})"
        )
    if spearman_within_tol and rmse_clear_win:
        return True, (
            f"Spearman within the {SPEARMAN_TOLERANCE} tolerance "
            f"({tuned.spearman:.3f} vs {untuned.spearman:.3f}) while RMSE clearly improved "
            f"({tuned.rmse:.3f} vs {untuned.rmse:.3f})"
        )
    return False, (
        f"tuned model does not beat the untuned baseline on both metrics within tolerance "
        f"(RMSE {tuned.rmse:.3f} vs {untuned.rmse:.3f}, Spearman {tuned.spearman:.3f} vs "
        f"{untuned.spearman:.3f})"
    )


def _tuning_chart(untuned_rmse: float, tuned_rmse: float) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    names = ["XGBoost (untuned)", "XGBoost (tuned)"]
    rmses = [untuned_rmse, tuned_rmse]
    colours = ["#3b7dd8", "#2ca25f" if tuned_rmse < untuned_rmse else "#b34b4b"]
    ax.barh(names, rmses, color=colours)
    ax.set_xlabel("validation RMSE (lower is better), held-out GWs")
    ax.set_title("XGBoost: untuned vs. Optuna-tuned (objective = RMSE)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "xgb_tuning_rmse.png", dpi=110)
    plt.close(fig)
    # The old MAE-focused chart from the previous (incorrect) run is no
    # longer representative of the selection criterion - remove it rather
    # than leave a stale, misleading artifact next to the new one.
    stale = FIG / "xgb_tuning_mae.png"
    if stale.exists():
        stale.unlink()


def _write_best_params(tune_out: dict, ctx: dict, untuned_result, deployed: str) -> None:
    payload = {
        "model": "XGBoost",
        "objective_metric": "rmse",
        "objective_metric_reason": (
            "RMSE (not MAE) - this model feeds a squad-selection solver where correctly "
            "identifying big-haul/captaincy candidates matters more than average-case "
            "error; RMSE penalises large errors quadratically and MAE does not. See "
            "tune.py's module docstring for the full history of why this was corrected."
        ),
        "fixed_params": {"random_state": RANDOM_STATE, "n_jobs": -1, "objective": "count:poisson"},
        "search": {
            "method": "Optuna TPESampler",
            "n_trials": N_TRIALS,
            "cv_method": "forward-chaining chronological folds (no shuffling)",
            "n_folds": len(tune_out["folds"]),
            "folds_gws": [
                {"train_gws": sorted(tr), "valid_gws": sorted(va)}
                for tr, va in tune_out["folds"]
            ],
            "trial_rmse_vs_spearman_correlation": round(tune_out["rmse_spearman_corr"], 4),
        },
        "selection_rule": {
            "description": (
                "tuned model deployed only if it beats the untuned baseline on both "
                "RMSE and Spearman, or is within a small explicit tolerance on one while "
                "clearly winning the other"
            ),
            "rmse_tolerance_pct": RMSE_TOLERANCE_PCT,
            "spearman_tolerance": SPEARMAN_TOLERANCE,
        },
        "untuned_params": XGB_UNTUNED_PARAMS,
        "tuned_params": tune_out["best_params"],
        "untuned_holdout_metrics": {
            "mae": round(untuned_result.mae, 4),
            "rmse": round(untuned_result.rmse, 4),
            "spearman": round(untuned_result.spearman, 4),
        },
        "tuned_holdout_metrics": {
            "mae": round(tune_out["tuned_result"].mae, 4),
            "rmse": round(tune_out["tuned_result"].rmse, 4),
            "spearman": round(tune_out["tuned_result"].spearman, 4),
        },
        # Which params are ACTUALLY in production right now - explicit, not
        # left to be inferred from whichever section someone reads first.
        "deployed": deployed,  # "tuned" or "untuned"
        "deployed_params": tune_out["best_params"] if deployed == "tuned" else XGB_UNTUNED_PARAMS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (MODELS / "xgb_best_params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {MODELS / 'xgb_best_params.json'} (deployed: {deployed})")


def _write_comparison_doc(tune_out: dict, ctx: dict, untuned_result, deployed: str, reason: str) -> None:
    tuned = tune_out["tuned_result"]
    folds = tune_out["folds"]
    corr = tune_out["rmse_spearman_corr"]

    L: list[str] = ["## Hyperparameter tuning (XGBoost)\n"]
    a = L.append
    a("XGBoost won the original five-model comparison above on hand-picked "
      "defaults. This section tunes it with [Optuna](https://optuna.org/) "
      "(TPE sampler) and reports the result against that same untuned model, "
      "on the exact same held-out gameweeks.\n")

    a("### Why the search can't use a random/shuffled CV\n")
    a("The same reason the original train/validation split is chronological, "
      "not random (see \"How the data was split\" above): every row carries "
      "rolling-window features built from a player's own *past* gameweeks. "
      "A random k-fold inside the search - the default in most hyperparameter- "
      "tuning tutorials, and what Optuna's own CV integrations assume you'll "
      "supply - would let a fold's \"validation\" rows sit chronologically "
      "*before* rows in its own \"training\" fold, leaking future information "
      "into the model the search is scoring. So the objective function uses "
      "hand-built **forward-chaining (expanding-window) folds**, entirely "
      "inside the original training pool, never touching the GW"
      f"{VALID_FROM_GW}-38 holdout used for the final number.\n")

    a("### A previous version of this search optimized the wrong metric\n")
    a("The first tuning pass optimized **MAE** and picked the resulting model "
      "because MAE improved ~4.5%, without checking RMSE or Spearman. RMSE "
      "actually got **~9.6% worse** and Spearman dropped too - the search had "
      "found a model that's better at the boring middle of the points "
      "distribution and worse at exactly the big-haul, captaincy-deciding "
      "gameweeks RMSE is sensitive to (RMSE penalises large errors "
      "quadratically; MAE weighs every point of error the same). Since this "
      "model feeds a squad-selection solver where spotting big-haul "
      "candidates matters more than shaving error off routine rows, "
      "optimizing MAE was optimizing the wrong thing. **This run optimizes "
      "RMSE instead**, and tracks Spearman for every trial (not just the "
      "winner) specifically to catch the same failure mode recurring in the "
      "other direction.\n")

    a("### Search setup\n")
    a(f"- **Method:** Optuna, TPE sampler, {N_TRIALS} trials, **objective = RMSE** "
      "(previously MAE - see above).")
    a(f"- **Validation:** {len(folds)} forward-chaining folds inside the "
      f"GW{WARMUP_LAST_GW + 1}-{VALID_FROM_GW - 1} training pool (unchanged from "
      "the previous run):")
    for i, (tr, va) in enumerate(folds, 1):
        a(f"  - fold {i}: train GW{min(tr)}-{max(tr)}, validate GW{min(va)}-{max(va)}")
    a("- **Search space:** `max_depth` [3, 8], `learning_rate` [0.01, 0.3] (log), "
      "`n_estimators` [100, 500], `subsample` [0.5, 1.0], `colsample_bytree` "
      "[0.5, 1.0], `min_child_weight` [1, 10], `reg_alpha` / `reg_lambda` "
      "[1e-8, 10] (log). `objective=count:poisson` and `random_state` held "
      "fixed, matching the untuned model.")
    a(f"- **Best params found:** `{json.dumps(tune_out['best_params'])}`")
    a(f"- **Across-trial diagnostic:** correlation between each trial's fold "
      f"RMSE and fold Spearman = **{corr:+.3f}**. " + (
          "Positive - trials with worse RMSE tended to have better Spearman, i.e. "
          "there IS some RMSE-vs-Spearman tension in this search space, the same "
          "shape of problem as before, just smaller (see the selection rule below "
          "for how that's guarded against)."
          if corr > 0.15 else
          "Small/negative - no meaningful sign that chasing RMSE dragged Spearman "
          "down across this search; the two moved together often enough not to "
          "be a systematic trade-off here."
      ) + "\n")

    a(f"### Tuned vs. untuned (held-out gameweeks {VALID_FROM_GW}-38)\n")
    a("| Model | MAE | RMSE | Spearman | MAE (played) | Beats naive? |")
    a("|---|---|---|---|---|---|")
    for r in (untuned_result, tuned):
        flag = "**yes**" if r.beats_naive else "no"
        a(f"| {r.name} | {r.mae:.3f} | {r.rmse:.3f} | {r.spearman:.3f} | {r.mae_played:.3f} | {flag} |")
    a("")
    a("![untuned vs tuned XGBoost RMSE](figures/xgb_tuning_rmse.png)\n")

    a("### Selection rule\n")
    a(f"The tuned model is deployed only if it beats the untuned baseline on **both** "
      f"RMSE and Spearman, or is within a small explicit tolerance on one "
      f"({RMSE_TOLERANCE_PCT}% for RMSE, {SPEARMAN_TOLERANCE} for Spearman - both set "
      "well below the size of the regression the MAE-optimized run produced) while "
      "clearly winning the other. No single-metric selection this time - that's "
      "exactly what caused the previous problem.\n")

    rmse_gain_pct = (untuned_result.rmse - tuned.rmse) / untuned_result.rmse * 100
    spearman_gain = tuned.spearman - untuned_result.spearman
    mae_change_pct = (tuned.mae - untuned_result.mae) / untuned_result.mae * 100

    a("### Verdict\n")
    a(f"**{reason}.**\n")
    if deployed == "tuned":
        a(f"`models/stats_model.pkl` and `models/xgb_best_params.json` were updated to "
          f"the tuned model (MAE {tuned.mae:.3f}, RMSE {tuned.rmse:.3f}, Spearman "
          f"{tuned.spearman:.3f}).")
        # The selection rule can technically pass on a razor-thin margin -
        # say so plainly rather than let "PASSED" read as a bigger win than
        # it is. This is the same discipline that was missing last time.
        if abs(rmse_gain_pct) < 1.0 and abs(spearman_gain) < 0.01:
            a(f"\nHonest read on the size of this: the margin is **razor-thin** - "
              f"RMSE moved {rmse_gain_pct:+.2f}% and Spearman moved {spearman_gain:+.3f}, "
              "both well within what re-running the search with a different seed would "
              "shift. This technically clears the selection bar (it does not regress "
              "either metric), but it should be read as *\"roughly equivalent "
              "performance, re-tuned\"*, not *\"a clear win\"* - the previous run's "
              "mistake was overstating a small number in one direction; the fix isn't to "
              "overstate a small number in the other.")
        if mae_change_pct > 0.5:
            a(f"\nOne real trade made deliberately: MAE moved from {untuned_result.mae:.3f} "
              f"to {tuned.mae:.3f} ({mae_change_pct:+.1f}%) - worse than the untuned model. "
              "That's expected and accepted, not a regression to worry about: MAE is no "
              "longer what the search optimizes for (see \"A previous version of this "
              "search optimized the wrong metric\" above), so a model selected on "
              "RMSE/Spearman has no reason to also be MAE-optimal, and isn't here.")
    else:
        a(f"`models/stats_model.pkl` is **unchanged** - it still holds the untuned "
          f"XGBoost (MAE {untuned_result.mae:.3f}, RMSE {untuned_result.rmse:.3f}, "
          f"Spearman {untuned_result.spearman:.3f}). `models/xgb_best_params.json` "
          "records the untuned params as what's actually deployed, plus the searched "
          "(but rejected) tuned params for reference. This is a legitimate outcome, "
          "not a failed run: the search tried a real ~9-dimensional space under an "
          "honest selection rule and the hand-picked defaults held up.")
    a("")

    # Idempotent, not just append-only: re-running this script should
    # replace the tuning section with a fresh one, not stack duplicates on
    # top of the base comparison train.py wrote. Splitting on the header
    # text itself (rather than matching exact prior whitespace) keeps this
    # robust across runs, including this correction overwriting the
    # previous (MAE-based) tuning section entirely rather than appending
    # alongside it.
    marker = "## Hyperparameter tuning (XGBoost)"
    doc_path = DOCS / "model_comparison.md"
    existing = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""
    base = existing.split(marker)[0]
    doc_path.write_text(base.rstrip("\n") + "\n\n---\n\n" + "\n".join(L), encoding="utf-8")
    print(f"wrote tuning section to {doc_path}")


def main() -> None:
    np.random.seed(RANDOM_STATE)

    print("building features + running the original 5-model comparison (train.py)...")
    ctx = train_module.train_all()
    untuned_result = next(r for r in ctx["results"] if r.name == "XGBoost")
    print(f"untuned XGBoost: MAE={untuned_result.mae:.3f} RMSE={untuned_result.rmse:.3f} "
          f"spearman={untuned_result.spearman:.3f}")

    print(f"tuning XGBoost with Optuna ({N_TRIALS} trials, objective=RMSE)...")
    tune_out = tune_xgboost(ctx)
    tuned = tune_out["tuned_result"]
    print(f"tuned XGBoost:   MAE={tuned.mae:.3f}  RMSE={tuned.rmse:.3f}  "
          f"spearman={tuned.spearman:.3f}  beats_naive={tuned.beats_naive}")
    print(f"trial RMSE-vs-Spearman correlation: {tune_out['rmse_spearman_corr']:+.3f}")

    passed, reason = _passes_selection_rule(tuned, untuned_result)
    print(f"selection rule: {'PASSED' if passed else 'FAILED'} - {reason}")

    # ALWAYS re-save stats_model.pkl with whichever model actually wins under
    # the corrected rule - including re-saving the UNTUNED model when tuning
    # doesn't pass. That matters concretely right now: the artifact on disk
    # currently holds the model from the previous (flawed, MAE-selected)
    # run, which must not be left in place just because this run's search
    # happened not to win - "no change" here would silently leave a
    # rejected model deployed.
    if passed:
        ctx["winner_model"] = tune_out["tuned_model"]
        ctx["winner"] = tuned
        deployed = "tuned"
    else:
        ctx["winner_model"] = ctx["fitted"]["XGBoost"]
        ctx["winner"] = untuned_result
        deployed = "untuned"

    # explain_winner() (train.py) works off `ctx["winner_model"]`/`ctx["winner"]`,
    # set just above, so SHAP, the pickle, and the beeswarm plot all describe
    # whichever model is actually being deployed.
    shap_out = train_module.explain_winner(ctx)
    train_module.save_model(ctx, shap_out)
    print(f"models/stats_model.pkl saved as the {deployed} model.")

    _tuning_chart(untuned_result.rmse, tuned.rmse)
    _write_best_params(tune_out, ctx, untuned_result, deployed)
    _write_comparison_doc(tune_out, ctx, untuned_result, deployed, reason)
    print("done.")


if __name__ == "__main__":
    main()
