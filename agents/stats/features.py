"""Archive-side entry point for feature building: load the vaastav CSV, then
hand it to the shared engineering module.

All the actual feature logic (rolling windows, fixture adjustment, per-90,
price momentum, cold-start handling) lives in ``features/engineering.py`` now
- it's imported by both this training-side path and the live Stats agent /
weekly ingestion path (``ingestion/silver.py``), so the two can never drift
apart. See ARCHITECTURE.md section 3a ("Shared feature engineering").

This module's only job is the archive-specific part: turning
``merged_gws_2025-26.csv`` into the common "gameweek frame" shape
``build_features_from_frame`` expects (one row per player per gameweek,
double-gameweeks collapsed) - that parsing lives in ``data.py``.
"""

from __future__ import annotations

from features.engineering import build_features_from_frame

from .data import load_gameweeks


def build_feature_frame(csv_path=None) -> tuple:
    """Return ``(frame, feature_columns, target_column)`` for the archive CSV.

    Thin wrapper kept for backward compatibility with ``train.py`` and the
    existing tests, which import ``build_feature_frame`` by name.
    """
    df = load_gameweeks(csv_path) if csv_path else load_gameweeks()
    return build_features_from_frame(df)


if __name__ == "__main__":
    frame, feats, tgt = build_feature_frame()
    print(f"{len(frame):,} rows, {len(feats)} features, target = {tgt!r}")
    print(frame[["name", "GW", "pts_roll5", "fixture_adj_xp", tgt]].head(12))
