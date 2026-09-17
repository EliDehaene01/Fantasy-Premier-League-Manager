"""Stats agent: offline model building, live serving, and the model
lifecycle loop that keeps it honest over a season.

Three parts, see agents/stats/README.md for the full breakdown:

  * data science (``eda.py``, ``features.py``, ``train.py``, ``tune.py``) -
    EDA, a time-respecting train/validation split, a five-model comparison,
    hyperparameter tuning, and SHAP explainability. Produces
    ``models/stats_model.pkl``.
  * the FastAPI service (``service.py``, ``model_runtime.py``,
    ``reasoning.py``, using the shared contract in
    ``shared/contracts.py``) - loads that model and answers
    ``POST /argue`` in the gameweek debate, calling out to a Foundry LLM for
    the prose only.
  * the model lifecycle loop (``predictions_log.py``, ``monitor.py``,
    ``retrain.py``) - logs every live prediction to Postgres, watches
    rolling error against the backtest baseline, and retrains automatically
    on sustained degradation. See docs/monitoring.md.
"""
