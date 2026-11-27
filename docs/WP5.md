# WP5 — Frustration detection

**Status:** planned, not started.
**Done when:** `predict(text)` returns a calibrated frustration score for one
caller turn, measured on a split that does not leak paraphrases.

The point is to notice a caller losing patience *before* they have to say
"passez-moi quelqu'un", so the handover happens while it still reads as
service rather than as a last resort.

## Scope

| In | Out |
|---|---|
| A text classifier over sentence embeddings | Wiring into `escalation.check()` (WP7) |
| Grouped cross-validation and per-kind error analysis | Dialogue state and turn counting (WP6) |
| Calibrating `FRUSTRATION_ESCALATE_THRESHOLD` | `MAX_TURNS_BEFORE_ESCALATION` (needs WP6's state) |
| A Gradio panel showing the score | The handover wording (WP7's fixed script) |
| | Darija and Arabizi |

## What the corpus actually contains

`data/raw/frustration.csv`, 476 rows, columns `id,text,label,lang,kind,group`.

| lang | rows | | label | rows |
|---|---:|---|---|---:|
| fr | **180** | | 0 (calm) | 231 |
| tn | 138 | | 1 (frustrated) | 245 |
| tn_latn | 102 | | | |
| mix | 56 | | | |

`kind` maps **deterministically** onto `label` — every `frustrated` row is a 1,
every `neutral` row a 0, with no exceptions. So it is not extra training
signal. It is the error-analysis axis, and that is where its value is.

### The French set, which is what WP5 reports

180 rows: **85 frustrated, 95 calm.** Balanced enough to need no resampling.

| kind | label | rows |
|---|---|---:|
| frustrated | 1 | 75 |
| frustrated_hard | 1 | 10 |
| neutral | 0 | 54 |
| worry | 0 | 12 |
| correction | 0 | 11 |
| positive | 0 | 11 |
| other_complaint | 0 | 7 |

The four small negative kinds are the whole difficulty of the task. A model
that scores 90 % overall while confusing `correction` with frustration is
useless, because corrections are the most common thing a caller does in a
booking.

## Decision 1 — the split must be grouped, or the number is fiction

The `group` column marks paraphrases of the same complaint. In the French set:
**119 groups, 14 of them holding more than one row, and 75 of the 180 rows —
42 % — sitting inside a multi-row group.** The largest holds six.

One such group, from the Darija side where they are easiest to see at a glance:

```
والله ضيّعتلي وقتي!
يا خويا ضيّعتلي وقتي!!
يا خويا ضيّعتلي وقتي، عدّيني لعبد
يزّي ضيّعتلي وقتي، عدّيني لعبد
والله ضيّعتلي وقتي، باش نسكّر
ضيّعتلي وقتي
```

Six wordings of "you wasted my time". A plain `StratifiedKFold` puts some of
those in training and the rest in test, and then reports that the model
generalises. It does not — it has seen the sentence.

**So: `StratifiedGroupKFold` on `group`.** And report *both* numbers, grouped
and ungrouped, because the gap between them is the only honest measure of how
much the paraphrases were inflating the score. Hiding the ungrouped number
hides the evidence that the grouped one was necessary.

This is the third time this project has met the same error in a different
costume — in-index similarity in WP3 (fixed with leave-one-out), a
catalogue-derived test set in WP4 (fixed by commissioning `exam_test_t3.csv`).
Here the corpus ships the column that prevents it, which is a credit to
whoever built the data.

## Decision 2 — the stub's features do not exist, and cannot yet

`src/hermes/frustration.py` declares:

```python
class TurnFeatures:
    turn_index: int
    repeated_question: bool
    failed_matches: int
    audio_retries: int
    words: int
    exclamations: int
```

**Four of those six cannot be computed from this corpus at all.** It is 476
standalone sentences. There is no turn index, no repeat count, no record of a
failed match or an audio retry — and no dialogue to derive them from. Three of
them (`repeated_question`, `failed_matches`, `audio_retries`) need the dialogue
state that WP6 owns and WP3 deliberately deferred, so they could not be
populated at run time either.

Only `words` and `exclamations` are derivable from the text. They are worth an
**ablation** rather than an assumption: measure the embedding alone against the
embedding plus those two, and keep them only if they earn it. A single
exclamation mark is a weak signal in a corpus where the calm rows include
"merci beaucoup !".

**So WP5 is text-only**, and `TurnFeatures` stays in the signature as an
optional argument that WP6 fills in. This is the third stub in the project
written before its data — the WP3 intent enum had no `medical_question`, the
WP4 `Exam` dataclass had four fields `exams.csv` does not contain. Worth
naming as a pattern: the skeletons are a sketch of intent, not a contract, and
each one should be checked against the corpus before it is built on.

`FrustrationScore` is imported by `agent.py`, `escalation.py`, `pipeline.py`
and `logging_utils.py`, so **that name does not change.**

## Decision 3 — the plan says MLP; this project has already tested that claim

`docs/README.md` and the plan both call WP5 an "MLP on sentence embeddings".
WP3 ran exactly that comparison on the identical embeddings for the intent
gate, and `scripts/train_intent_classifier.py` records the outcome in its own
docstring: the MLP had **the best raw accuracy and was still the wrong
choice**, losing on macro F1 and on medical recall.

So WP5 compares logistic regression against the MLP and picks on the metric
that matters here, rather than inheriting the plan's architecture. With 180
rows and 384 dimensions, the prior strongly favours the linear model — more
parameters than examples is how you fit paraphrase groups instead of
frustration.

## Decision 4 — French only, and the reason differs from WP4's

Same conclusion as WP4, different argument, so it needed its own measurement.

In WP4 the `mix` rows were usable: the exam name was French and only the
greeting was Darija, so the part being matched was in scope. **Here the signal
itself is usually Darija.**

```
votre machine ma tefhem chay !!          frustration = "ma tefhem chay"
yezzi passez-moi quelqu'un, fadit mel robot    = "yezzi", "fadit mel robot"
merci barcha !                                  = "barcha"
```

Measured: **53 of the 56 `mix` rows** contain Arabic script or Arabizi, and the
remaining three still lean on Darija ("lezem", "rokba limin"). A
French-language reading of those rows has almost nothing to work from.

So `mix` is excluded here, where WP4 included it. The embedder is
multilingual (`paraphrase-multilingual-MiniLM-L12-v2`), so it may cover the
Arabic-script rows to some degree — but Arabizi is Latin-script Darija, which
multilingual models handle poorly, and measuring it would be measuring the
embedder's Darija coverage rather than frustration detection.

The 296 non-French rows stay in the file for whenever Darija happens.

## Decision 5 — which error costs more? This one needs a decision

Every earlier module in this project had an obvious asymmetry: a clarifying
question is cheap, a wrong action is not. **Frustration detection does not,
because both errors are actions.**

| error | what happens | cost |
|---|---|---|
| false positive | a calm caller is handed to a human | a human's minute — and WP3 spent real effort cutting human load from 44.8 % to 28.6 % of turns by declining off-topic calls. False positives give that back. |
| false negative | a frustrated caller stays with the bot | the caller gets angrier, and this is the exact failure the feature exists to prevent |

My reading, offered as a recommendation and not acted on: **favour recall.** A
needless transfer costs one bounded human minute, while a missed frustration
compounds over every remaining turn and is what a marker — or a real caller —
would notice. That argues for a threshold below 0.5 rather than the 0.70
currently sitting in `config.py` as an uncalibrated placeholder.

But that is a service-policy judgement about the centre's staffing, not a
measurement, and `FRUSTRATION_ESCALATE_THRESHOLD` is where it gets encoded. It
needs an explicit answer before step 3.

### `correction` is the threshold's real constraint

Whatever the policy, the per-kind table decides whether it is affordable:

- **`correction` (11 rows)** — "non, j'ai dit mardi". Routine, frequent, and
  structurally similar to complaint. Flagging corrections as frustration would
  escalate a large share of ordinary bookings.
- **`worry` (12)** — anxious about the exam, not annoyed with the agent.
  Arguably *should* reach a human, but for a different reason, and conflating
  the two makes the score mean two things.
- **`other_complaint` (7)** — unhappy about something that is not us.
- **`frustrated_hard` (10)** — the subtle positives. Recall here is the real
  test of the model, as opposed to recall on "yezzi, passez-moi quelqu'un".

With 7 to 12 rows per hard kind, these are **indicative, not decisive**, and
the report must say so rather than quoting a percentage of eleven.

## Decision 6 — 180 rows, and what that permits

The same order as WP3's intent calibration (222 French examples), so the
method carries over. One sentence is 0.6 points, which means **no decimal
places in the report** and no claim resting on a two-row difference. Grouped
5-fold leaves roughly 24 groups per fold.

## Plan of work

1. `scripts/train_frustration.py` — embed the 180 French rows, run
   `StratifiedGroupKFold`, compare logistic regression against the MLP, print
   the grouped and ungrouped numbers side by side, and the per-kind breakdown.
   Writes `models/frustration.joblib`.
2. `src/hermes/frustration.py` — `load_dataset`, text-only `featurise`, cached
   `load_model`, `predict` returning `FrustrationScore`. Degrades to a neutral
   score when the weights are missing, as the stub's docstring promises.
3. Calibrate `FRUSTRATION_ESCALATE_THRESHOLD` on the grouped cross-validated
   probabilities, under whichever policy Decision 5 settles on, and record the
   run in `config.py` beside the value.
4. The `words`/`exclamations` ablation. Ship them only if they beat the
   embedding alone on the grouped split.
5. Tests: score is a probability; a calm sentence scores low and an angry one
   high; the threshold decides the flag; missing weights degrade instead of
   raising; and one test per hard kind so a regression on `correction` is
   visible.
6. A Gradio panel on the existing bench, showing the score and the nearest
   labelled neighbours so a demo can be argued with.
7. Write the measurements here.

Thresholds are calibrated at step 3, after the model exists and before
anything consumes the score.

## Open questions

1. **The error asymmetry in Decision 5.** Needed before step 3, because it is
   what the threshold encodes. My recommendation is to favour recall; it is
   your call about the centre's staffing.
2. **`MAX_TURNS_BEFORE_ESCALATION = 8`** is a policy choice with no measurement
   behind it and it needs a turn counter, which is WP6's dialogue state.
   Confirmed out of scope for WP5 unless you want it moved in.
3. **Darija.** 296 of 476 rows are `tn`, `tn_latn` or `mix`. Excluded from the
   reported numbers, consistent with WP4 and with the scope decision. The rows
   stay in the file.

## Acceptance criteria

- [ ] `predict(text)` returns a `FrustrationScore` with a probability in [0, 1]
- [ ] grouped **and** ungrouped cross-validated scores both reported, with the
      gap stated as the cost of the paraphrase leak
- [ ] logistic regression compared against the MLP on identical embeddings,
      with the choice argued on a named metric
- [ ] per-kind breakdown, with `correction`, `worry`, `other_complaint` and
      `frustrated_hard` called out and their row counts shown
- [ ] `FRUSTRATION_ESCALATE_THRESHOLD` calibrated, its run recorded in
      `config.py`, under a stated policy
- [ ] the `words`/`exclamations` ablation measured, and the features shipped
      only if they earn it
- [ ] missing weights degrade to a neutral score rather than raising
- [ ] tests including one per hard kind
