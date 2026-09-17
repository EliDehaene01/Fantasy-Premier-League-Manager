"""The embedding-based availability-severity classifier - the production
path for Tier 2's availability judgment, per the evaluation in
``severity_finetune_eval.py`` (embedding classifier macro-F1 0.58 vs
zero-shot's 0.192 on the same held-out real test set: see
docs/severity_classifier.md). Kept separate from that eval script so
production code (``tier2.py``) depends only on this small module, not on
an evaluation harness.

SCOPE: availability severity (OUT/DOUBT/FIT) only - see
severity_finetune_eval.py's module docstring for why this never extends to
the notable-positive-coverage judgment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from agents.stats.data import REPO_ROOT
from shared.contracts import VetoStatus

from . import embeddings

MODEL_PATH = REPO_ROOT / "models" / "severity_classifier.joblib"  # same models/ dir as stats_model.pkl

# --- labeling rule (see severity_finetune_eval.py's docstring for the full
# reasoning: formulaic FPL status text -> an explicit, auditable rule
# rather than one-by-one manual labeling) ------------------------------
_PCT_RE = re.compile(r"(\d+)% chance of playing")


def _label_chunk(chunk_text: str) -> str:
    m = _PCT_RE.search(chunk_text.lower())
    if m:
        pct = int(m.group(1))
        if pct <= 25:
            return "OUT"
        if pct <= 50:
            return "DOUBT"
        return "FIT"
    return "OUT"


@dataclass
class LabeledExample:
    player_name: str
    chunk_text: str
    label: str  # ground truth: OUT/DOUBT/FIT


def label_corpus(conn) -> list[LabeledExample]:
    rows = conn.execute(
        "SELECT chunk_text FROM news_passages WHERE corpus = 'team_news' ORDER BY id"
    ).fetchall()
    examples = []
    for r in rows:
        chunk = r["chunk_text"]
        name = chunk.split(" (", 1)[0].strip()
        examples.append(LabeledExample(player_name=name, chunk_text=chunk, label=_label_chunk(chunk)))
    return examples


def train(examples: list[LabeledExample]) -> LogisticRegression:
    X = np.array(embeddings.embed_texts([ex.chunk_text for ex in examples]))
    y = [ex.label for ex in examples]
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, y)
    return clf


def save(clf: LogisticRegression, path: Path = MODEL_PATH) -> None:
    joblib.dump(clf, path)


_cached_clf: LogisticRegression | None = None


def _load(path: Path = MODEL_PATH) -> LogisticRegression:
    global _cached_clf
    if _cached_clf is None:
        _cached_clf = joblib.load(path)  # raises FileNotFoundError if never trained/deployed - caller falls back
    return _cached_clf


def classify_availability(player_name: str, passages: list[str]) -> dict:
    """Same return contract as ``tier2.classify_availability_zero_shot``:
    ``{"status": VetoStatus, "confidence": float, "grounding_snippet": str | None}``.

    Uses only the top (most relevant) retrieved passage - retrieval already
    ranks by similarity, and the classifier was trained on single chunks, so
    this matches training conditions rather than inventing a multi-passage
    combination rule that was never evaluated.
    """
    clf = _load()
    text = passages[0]
    X = np.array(embeddings.embed_texts([text]))
    proba = clf.predict_proba(X)[0]
    idx = int(np.argmax(proba))
    return {
        "status": VetoStatus(clf.classes_[idx]),
        "confidence": float(proba[idx]),
        "grounding_snippet": text,
    }
