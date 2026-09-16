"""Fine-tuned (embedding-based) availability-severity classifier for the
News agent, evaluated against the zero-shot LLM classification already live
in Tier 2 (``tier2.classify_availability_zero_shot``).

Run: ``python -m agents.news.severity_finetune_eval``

SCOPE: availability severity (OUT/DOUBT/FIT) ONLY - never extended to the
notable-positive-coverage judgment. That judgment is inherently fuzzy and
qualitative (see ``tier2.POSITIVE_COVERAGE_SYSTEM_PROMPT``: "be conservative",
"would make a football fan sit up") - there is no formulaic rule to derive
ground-truth labels from the way there is for availability status text, so
there is nothing honest to fine-tune or evaluate against. Availability status
text ("25% chance of playing", "Suspended until...") is short and formulaic
enough to label by an explicit, auditable rule (see
``severity_classifier._label_chunk``) - exactly the kind of task a small
classifier is suited to.

WHY COMPARE AGAINST ZERO-SHOT AT ALL (not just "does the fine-tune work")
--------------------------------------------------------------------------
A fine-tuned classifier that merely functions is not automatically worth
deploying - it adds a second code path and a model artifact to keep in sync,
for a task an already-capable general model may already handle just as well.
The only way to know whether that complexity earns its keep is a head-to-head
on the SAME held-out cases: if the fine-tune doesn't clearly beat zero-shot,
the honest outcome is "don't deploy it", not "deploy it because it runs".

WHY AN EMBEDDING CLASSIFIER, NOT A HOSTED LLM FINE-TUNE
---------------------------------------------------------
Checked directly against this project's real Foundry project first:
``client.fine_tuning.jobs.list()`` succeeds - fine-tuning is reachable here,
not categorically unavailable. But the real labeled corpus (188 real
team_news passages) is small and, once split, leaves very few DOUBT/FIT
examples (most real entries are unambiguous OUT: suspensions, "unknown
return date", transfers out). Committing a remote, billed, hours-scale
hosted fine-tune to a dataset this thin is not a responsible use of that
mechanism. ARCHITECTURE.md's own wording sanctions the alternative used
here: "fine-tune a small classifier (OR AN EMBEDDING MODEL)". A linear
classifier on top of already-computed text-embedding-3-small vectors is
cheap, fast, needs no hosted training job, and is a legitimate instance of
that sanctioned alternative - not a workaround dodging the real comparison.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from ingestion.db import get_connection

from . import embeddings, severity_classifier
from .severity_classifier import LabeledExample, label_corpus  # re-exported: this eval script's own labeling rule
from .tier2 import classify_availability_zero_shot


def train_test_split(examples: list[LabeledExample], *, test_frac: float = 0.25, seed: int = 42):
    """Stratified BY LABEL so the small DOUBT/FIT classes appear in both
    splits rather than vanishing from a plain random split. Held out BEFORE
    anything else touches this data - never used to train or tune below.
    """
    rng = np.random.RandomState(seed)
    by_label: dict[str, list[LabeledExample]] = {}
    for ex in examples:
        by_label.setdefault(ex.label, []).append(ex)

    train, test = [], []
    for group in by_label.values():
        idx = rng.permutation(len(group))
        n_test = max(1, round(len(group) * test_frac)) if len(group) > 1 else 0
        test_idx = set(idx[:n_test])
        train += [group[i] for i in range(len(group)) if i not in test_idx]
        test += [group[i] for i in range(len(group)) if i in test_idx]
    return train, test


def predict_embedding_classifier(clf, examples: list[LabeledExample]) -> list[str]:
    X = np.array(embeddings.embed_texts([ex.chunk_text for ex in examples]))
    return list(clf.predict(X))


def predict_zero_shot(examples: list[LabeledExample]) -> list[str]:
    """Calls the exact production function, one player at a time (matching
    how Tier 2 actually invokes it: one player, their own retrieved passages).
    """
    preds = []
    for ex in examples:
        try:
            result = classify_availability_zero_shot(ex.player_name, [ex.chunk_text])
            preds.append(result["status"].value)
        except Exception:
            preds.append("DOUBT")  # same spirit as tier2's real fallback: don't guess OUT/FIT on failure
    return preds


def evaluate(y_true: list[str], y_pred: list[str]) -> dict:
    labels = ["OUT", "DOUBT", "FIT"]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        lbl: {"precision": round(float(p), 3), "recall": round(float(r), 3),
              "f1": round(float(f), 3), "support": int(s)}
        for lbl, p, r, f, s in zip(labels, precision, recall, f1, support)
    }
    accuracy = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)
    return {"per_class": per_class, "accuracy": round(accuracy, 3), "macro_f1": round(macro_f1, 3)}


# Deploy only if the embedding classifier clearly beats zero-shot - a small
# stated margin, not "any improvement", so noise on a small test set doesn't
# flip the decision. See requirement 5: "only wire in if a real, meaningful
# improvement."
DEPLOY_MARGIN = 0.05


def main() -> None:
    from agents.stats.data import REPO_ROOT

    conn = get_connection()
    examples = label_corpus(conn)
    train, test = train_test_split(examples)

    label_counts = {lbl: sum(1 for e in examples if e.label == lbl) for lbl in ("OUT", "DOUBT", "FIT")}
    print(f"labeled {len(examples)} real team_news passages "
          f"({len(train)} train / {len(test)} held-out test)")
    print("label distribution:", label_counts)

    clf = severity_classifier.train(train)
    y_true = [ex.label for ex in test]

    clf_metrics = evaluate(y_true, predict_embedding_classifier(clf, test))
    print("embedding classifier:", json.dumps(clf_metrics, indent=2))

    zs_metrics = evaluate(y_true, predict_zero_shot(test))
    print("zero-shot (production):", json.dumps(zs_metrics, indent=2))

    deploy = clf_metrics["macro_f1"] > zs_metrics["macro_f1"] + DEPLOY_MARGIN
    if deploy:
        # Re-train on ALL labeled data (train+test) for the artifact that
        # actually goes into production - the held-out split above only
        # existed to measure whether this is worth deploying at all; once
        # that's decided, there's no reason to withhold real labeled signal
        # from the deployed model.
        production_clf = severity_classifier.train(examples)
        severity_classifier.save(production_clf)
        print(f"deployed classifier to {severity_classifier.MODEL_PATH}")
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"total": len(examples), "train": len(train), "test": len(test), "label_distribution": label_counts},
        "embedding_classifier_metrics": clf_metrics,
        "zero_shot_metrics": zs_metrics,
        "deploy_margin": DEPLOY_MARGIN,
        "deploy_embedding_classifier": deploy,
    }
    (REPO_ROOT / "docs" / "severity_classifier_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("deploy embedding classifier as production Tier 2 path:", deploy)
    conn.close()


if __name__ == "__main__":
    main()
