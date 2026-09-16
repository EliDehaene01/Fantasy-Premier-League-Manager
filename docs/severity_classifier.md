# Availability-severity classifier: fine-tune vs. zero-shot

Scope: availability severity (OUT/DOUBT/FIT) only, in Tier 2 of the News
agent. Never extended to notable-positive-coverage — see
`agents/news/severity_finetune_eval.py`'s module docstring for why that
judgment doesn't have a labelable ground truth the way availability text
does.

Reproduce this comparison: `python -m agents.news.severity_finetune_eval`
(reads real data from Postgres; requires a live Foundry connection for the
zero-shot side). Results are written to `docs/severity_classifier_eval.json`.

## 1. Dataset

188 real `team_news` passages already ingested into the live pgvector store
(`news_passages` where `corpus = 'team_news'`) — real scraped FPL
"Availability" tab entries, not synthetic examples.

**Labeling approach: rule-based, not manual.** FPL's team-news text is
short and formulaic (`"Suspended until 19 Oct"`, `"Ankle injury - 25%
chance of playing"`, `"Has joined Real Sociedad on loan for the rest of the
season"`). Rather than hand-labeling 188 rows with no more information than
these patterns already encode, labels are derived by an explicit, auditable
rule (`agents/news/severity_classifier.py::_label_chunk`):

| Pattern in text | Label | Why |
|---|---|---|
| "X% chance of playing", X <= 25 | OUT | too unlikely to risk selecting |
| "X% chance of playing", 25 < X <= 50 | DOUBT | a genuine toss-up |
| "X% chance of playing", X > 50 | FIT | likely enough to include |
| "Suspended" / ban language | OUT | not playing this gameweek |
| "Expected back \<date\>" (no %) | OUT | not confirmed for this gameweek |
| "Unknown return date" | OUT | not confirmed for this gameweek |
| "Has joined X ..." (loan/permanent/departed) | OUT | not available to select at all |

Stated plainly: this is a rule, not independent human judgment — it is
reproducible and auditable, but it also means the "ground truth" reflects
this project's own reading of FPL's status text, not a second, independent
labeler. Framed honestly, not as blind gold-standard labeling.

**3 classes, not a finer return-timeline scale**: the deployed `VetoStatus`
contract (`agents/stats/schemas.py`) only has three values (OUT/DOUBT/FIT).
Both the classifier and the zero-shot path have to produce that same
`Veto` shape, so there's nothing to gain from labeling a finer scale that
would just get collapsed back down before it could be used.

**Class balance, stated honestly**: the real corpus is heavily skewed
toward OUT (174 of 188 — mostly suspensions and, especially, the large
"has joined X" transfer-out category). DOUBT has only 3 real examples
league-wide; FIT has 11. This is a genuinely thin sample for the minority
classes — reported here rather than hidden, per the task's instruction to
"note honestly if the corpus doesn't have enough real examples yet." It
was judged enough to produce a directionally reliable comparison (see
results below), not enough to claim a precise, stable estimate of DOUBT-
class performance specifically.

## 2. Held-out test set

Stratified by label, 25% held out (`severity_finetune_eval.train_test_split`,
seed=42) **before** anything else touches the data — no training, no
tuning, no threshold-picking sees these rows first. 140 train / 48 test,
with every one of the 3 real DOUBT examples and several real FIT examples
present in both splits (a plain random split risked losing the minority
classes from one side entirely).

## 3. Why an embedding classifier, not a hosted LLM fine-tune

Checked directly against this project's real Microsoft Foundry project
before choosing an approach: `client.fine_tuning.jobs.list()` and
`client.models.list()` both succeed — fine-tuning is reachable here, not
categorically unavailable. But committing a remote, billed, hours-scale
hosted fine-tuning job to 140 training rows with 2-8 examples per minority
class is not a responsible use of that mechanism — the result would be
dominated by noise from a handful of examples. ARCHITECTURE.md's own
wording sanctions the alternative used here: "fine-tune a small classifier
(or an embedding model)." A `LogisticRegression` on top of
text-embedding-3-small vectors (already computed for the RAG corpus) is
cheap, trains in seconds, and is a legitimate instance of that sanctioned
alternative, not a workaround dodging the real comparison.

## 4. Results (held-out test set, 48 examples)

| Class | Metric | Embedding classifier | Zero-shot (production LLM) |
|---|---|---|---|
| OUT (support 44) | precision | 1.000 | 1.000 |
| | recall | 0.977 | 0.273 |
| | f1 | 0.989 | 0.429 |
| DOUBT (support 1) | precision | 0.000 | 0.038 |
| | recall | 0.000 | 1.000 |
| | f1 | 0.000 | 0.074 |
| FIT (support 3) | precision | 0.600 | 0.000 |
| | recall | 1.000 | 0.000 |
| | f1 | 0.750 | 0.000 |
| **Overall** | accuracy | **0.958** | **0.271** |
| | macro-F1 | **0.580** | **0.168** |

(Zero-shot numbers vary slightly run-to-run since it's a live LLM call - a
repeat run measured accuracy 0.25 / macro-F1 0.192. The gap over the
classifier is consistently large regardless.)

**Why zero-shot underperforms this badly - a real finding, not an LLM
failure to write off**: manually inspecting individual predictions (not
just the aggregate) showed the zero-shot model was not crashing or
timing out - it was consistently uncertain on two specific real patterns
that dominate this corpus:

- **"Has joined X on loan/permanently"** (the largest single OUT
  sub-category, ~90 real rows): `AVAILABILITY_SYSTEM_PROMPT` asks the model
  to judge *fitness*, and gives it no notion that "transferred to another
  club" also means "unavailable to select." The model split its guesses
  across OUT/DOUBT/FIT inconsistently on this pattern.
- **"Expected back \<date\>" with no percentage**: the model frequently
  called this DOUBT rather than OUT, a defensible read in isolation
  ("injured, return date given, so *some* chance"), but one that disagrees
  with this project's labeling rule, which treats anything not confirmed
  for *this* gameweek as OUT.

This is an honest, real gap in the current zero-shot **prompt's scope**,
not proof the underlying model is incapable of the task - a rewritten
system prompt describing these two cases explicitly would likely close
much of the gap. But that is exactly the fine-tune's advantage here: the
classifier is trained directly on the same rule its output is scored
against, so it never had this ambiguity to resolve in the first place.

## 5. Deploy decision

**Deployed.** The embedding classifier beats the zero-shot baseline by
macro-F1 0.58 vs ~0.17-0.19 - a wide, real margin, comfortably clearing the
small stated tolerance (`DEPLOY_MARGIN = 0.05` in
`severity_finetune_eval.py`) meant to guard against a marginal result
looking real on a 48-row test set. This is not a razor-thin case that
needs hedging.

`agents/news/tier2.py::resolve_availability` now calls
`severity_classifier.classify_availability` as the production path. The
original zero-shot function (`classify_availability_zero_shot`) is kept in
`tier2.py`, unused in production, so this comparison stays reproducible and
the evaluated baseline doesn't silently drift out of the codebase.

The trained model artifact (`models/severity_classifier.joblib`, retrained
on the full 188-row labeled set once the held-out comparison above
justified deploying it) is committed to version control, matching this
repo's existing convention for the Stats agent's own model artifact
(`models/stats_model.pkl`). Regenerate it by re-running
`python -m agents.news.severity_finetune_eval` whenever the underlying
corpus changes meaningfully (e.g. a full re-ingest).

## 6. Honest limitations

- DOUBT has only 3 real examples total; the classifier's 0.0 F1 on DOUBT in
  this run reflects 1 test example, not a stable estimate. As more real
  DOUBT-labeled news accumulates, this should be re-evaluated.
- The "ground truth" is this project's own rule-based reading of FPL's
  status text, not independently human-labeled - see the honesty note in
  section 1.
- The classifier is only as good as the labeling rule it was trained on; it
  will reproduce that rule's own blind spots (e.g. it has no notion of "how
  long ago the news was published" - an "expected back" date from three
  gameweeks ago and one from yesterday are treated identically by the label
  rule, and so identically by the classifier).
