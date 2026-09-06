"""Stats agent data-science pipeline.

This package contains the *offline* model-building code for the Stats agent:
EDA, feature engineering, a time-respecting train/validation split, a
five-model comparison, and SHAP explainability. The trained model it
produces (``models/stats_model.pkl``) is what the (separate) LLM-facing
Stats agent will load and quote from during the gameweek debate.

Nothing in here talks to an LLM or the FPL API - it is pure data science.
"""
