# WP5 — Frustration detection

**Status:** done. `detect(text, lang)` ships, with calibrated soft and hard
levels. Full numbers in [frustration_eval.md](../reports/frustration_eval.md).
**Done when:** ~~`predict(text)` returns a calibrated frustration score for one
caller turn, measured on a split that does not leak paraphrases.~~ Met.

## What shipped

**A torch MLP ensemble** — one hidden layer of 48, dropout 0.3, AdamW weight
decay, five members averaged, **epochs chosen per fold by inner grouped CV**
rather than asserted. Logistic regression and the lexicon are baselines.

| | P | R | F1 | ROC-AUC |
|---|---|---|---|---|
| **torch MLP ×5 (shipped)** | **0.93±0.01** | **0.92±0.02** | **0.93±0.02** | **0.99±0.00** |
| logistic regression | 0.93±0.01 | 0.93±0.02 | 0.93±0.01 | 0.98±0.00 |
| French lexicon, 84 terms in 9 families | 1.00 | 0.61 | 0.76 | — |

Grouped 5-fold over `group`, five seeds. The lexicon's precision is **1.00 —
zero false alarms in 180 rows.**

**Two levels, two actions:**

```
soft  if  prob >= 0.62                                      apologise, simplify
hard  if  prob >= 0.98  OR  (lexicon hit AND prob >= 0.62)  hand to a human
```

| level | recall | precision | correction | worry | neutral | positive |
|---|---|---|---|---|---|---|
| soft 0.62 | 0.91 | 0.94 | 3.4/11 | 0/12 | 0.4/54 | 1.0/11 |
| hard 0.98 | 0.78 | 0.98 | 1.0/11 | 0/12 | 0/54 | 0.2/11 |

`worry` is never flagged at either level — an anxious caller is never mistaken
for an annoyed one, which was the distinction most worth protecting.

**The hard level is one of three paths to a human.** Two consecutive soft hits
transfer, and so does the WP6/WP7 repeat counter. A row missed here is a
delayed handover, not a lost one — which is why recall 0.78 at the hard level
is acceptable where it would not be if this were the only route.

The point is to notice a caller losing patience *before* they have to say
"passez-moi quelqu'un", so the handover happens while it still reads as
service rather than as a last resort.

## Scope

| In | Out |
|---|---|
| A text classifier over sentence embeddings | Wiring into `escalation.check()` (WP7) |
| Grouped cross-validation and per-kind error analysis | Dialogue state and turn counting (WP6) |
| Calibrating the soft and hard thresholds | `MAX_TURNS_BEFORE_ESCALATION` (needs WP6's state) |
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

`src/hermes/understand/frustration.py` declares:

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
gate, and `scripts/intent/train_intent_classifier.py` records the outcome in its own
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

## The hybrid rule, and three attempts to get there

The hard level is not a bare threshold, and the route to that is the most
instructive thing in this package.

A threshold alone lands at **0.98** — because two calm rows are confidently
misread, `Non non, c'est bon, c'est juste une erreur de ma part` at 0.975 and
`Oui ça marche !` at 0.943 — and costs recall 0.71, missing 24 of 85. Meanwhile
the model scores `Bon, je répète, le genou DROIT !!` at **0.116**: explicit,
capitalised, twice exclaimed, and all but invisible to the embedding. The word
list catches that outright.

So the two are complementary, and ORing them should be free. It took three
attempts.

**Attempt 1, gated at the soft threshold** — a rounding error, +0.01 recall.
The gate excludes the very rows that motivated it: 0.116 is nowhere near 0.62.

**Attempt 2, ungated** — *worse* than plain, F1 0.88 against 0.90. The lexicon's
one false alarm was `Vous pouvez répéter s'il vous plaît ?`, which is itself a
`correction`, so admitting the word list spent the exact budget the hard
threshold was competing for and forced τ back to 0.98.

**Attempt 3** — fix the lexicon instead of the rule. Step 1 had already found
the real distinction and then ignored it: *the caller repeating themselves is
frustration; asking the agent to repeat is a correction, and that is
grammatical person, not vocabulary.* The repetition family had both `je répète`
and the bare infinitive in it. Restricting it to the first person removed the
word list's only false alarm, took its precision to 1.00, cost **no recall at
all**, and made the gated hybrid satisfiable.

Result: at the same τ and the same false alarms as the probability alone, the
hybrid catches **six more frustrated callers**. The gain came from correcting a
word list, not from tuning a threshold.

## Measured and not shipped

**A looser hard bar.** Allowing calm kinds to be flagged twice reaches τ 0.86
with recall 0.85 and F1 0.90. Not taken: the hard level transfers the call, and
the hybrid recovers most of that recall without spending the budget.

**Negation-contrast training rows.** `Oui ça marche !` scoring 0.943 looked like
general blindness to negation, and I recommended 40–60 polarity pairs on that
basis. Probing fresh pairs showed the opposite — `ça marche` 0.024 against `ça
ne marche pas` 0.700, `c'est bon comme ça` 0.014 against `ce n'est pas bon du
tout` 0.912, all separating correctly. The two corpus failures are those two
rows, not the phenomenon, so the data task was dropped.

**`words` and `exclamations` as features.** No gain: F1 0.93 → 0.93 for the
linear model, and inside its own standard deviation for the MLP. `merci
beaucoup !` is a calm row with an exclamation mark. `turn_features()` stays in
the module for WP6's use but is out of the feature vector.

## EmoWOZ: a measured negative result

EmoWOZ (`hhu-dsml/emowoz`, **CC-BY-NC-4.0**, Feng et al. LREC 2022) was added
as auxiliary English training data — user turns only, `dialogue_id` as group,
~1600 rows sampled balanced by `scripts/frustration/build_emowoz.py`. The French test folds
were held byte-identical and EmoWOZ rows entered **training folds only**,
enforced by an explicit loop rather than `cross_val_predict`, because a helper
that splits one matrix cannot express "these rows train but never test".

Every arm lost to French-only:

| arm | LR F1 | MLP F1 |
|---|---|---|
| French only | **0.93** | **0.92** |
| +1600 unweighted | 0.81 | 0.85 |
| +200 / +400 at equal total weight | 0.89 / 0.89 | 0.89 / 0.89 |
| +800 at equal total weight | 0.88 | 0.84 |
| pretrain on EmoWOZ, fine-tune on French | — | 0.75 |

**Weighting recovered 0.08 of the 0.12 drop, so swamping was real** — 1600
English rows against ~144 French ones. But 0.04 survives perfect weighting, and
that residue is genuine negative transfer: a **label-boundary mismatch**.
EmoWOZ's `dissatisfied` is annoyance inside a restaurant-booking dialogue, and
mapping it onto this project's label shifts the boundary rather than sharpening
it.

The mechanism is visible in one column — `neutral` false flags:

```
French only              1/0/0/0/1  of 54
+400 at equal weight     7/8/5/9/7
+1600 unweighted        12/13/12/13/13
pretrain + fine-tune    18/17/17/19/14
```

Recall *rises* 0.93 → 0.97 while precision falls 0.93 → 0.82. It is a bias
shift toward flagging, not a better detector.

**The MLP was more robust than LR to unweighted mixing** (0.85 against 0.81) —
dropout and the class-weighted loss absorbing some of the swamping — which is a
point in favour of the shipped model, not of the data.

Two expectations of mine that the measurement refuted: I predicted EmoWOZ's 200
`apologetic` rows would fix the `correction` weakness, and they did not — the
best any arm managed was leaving it unchanged. And I expected the paraphrase
leak to be large given 42 % of rows sit in a group; it is **+0.02**.

## Limitations — read these before quoting any number

**Recall 0.91 holds for corpus-like phrasing and not necessarily beyond it.**
The corpus's frustrated rows run on a narrow vocabulary — `répète`, `marre`,
`nul`, `n'importe quoi`, `perdre mon temps`. Probes written outside it are hit
and miss: `vous me faites tourner en bourrique` scores 1.000, while `ça
commence à m'énerver sérieusement` and `franchement c'est inacceptable ce
service` were both missed before the lexicon's `angry` and `fed up` families
were extended. **The hand-written held-out set
(`data/raw/frustration_test_user.csv`) is the real measure**, and it is scored
once at the end of the evaluation, never used for training or thresholds.

**The learning curve is flat for the shipped model.** 0.90 → 0.91 → 0.91 → 0.92
at 25/50/75/100 % of each training fold, inside its own ±0.01–0.02. An earlier
draft of this page called the model data-limited on the strength of logistic
regression's +0.02 over the same range; that was reading the wrong row. More
French rows are not the obvious lever.

**Probabilities are heavily saturated.** Five members agreeing produces many
exact 0.000 and 1.000 values, so real decisions happen in a thin band — which
makes `oui j'ai compris` at 0.597 uncomfortably close to the soft threshold.

**180 rows.** One row is 0.6 points. No figure here means anything past two
decimals, and the hard kinds (7 to 12 rows each) are reported as counts rather
than percentages throughout.

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
would notice.

**Settled, and the answer was neither.** The two-level design dissolves the
question: `soft` is recall-oriented at 0.62 and costs an apology, `hard` is
precision-oriented at 0.98 and costs a transfer. Each error is priced against
its own action instead of one threshold trying to serve both.

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

1. `scripts/frustration/train_frustration.py` — embed the 180 French rows, run
   `StratifiedGroupKFold`, compare logistic regression against the MLP, print
   the grouped and ungrouped numbers side by side, and the per-kind breakdown.
   Writes `models/frustration.joblib`.
2. `src/hermes/understand/frustration.py` — `load_dataset`, text-only `featurise`, cached
   `load_model`, `predict` returning `FrustrationScore`. Degrades to a neutral
   score when the weights are missing, as the stub's docstring promises.
3. Calibrate `FRUSTRATION_SOFT_THRESHOLD` and `FRUSTRATION_HARD_THRESHOLD` on
   the grouped cross-validated probabilities, and record each run in
   `config.py` beside the value.
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

1. **`data/raw/frustration_test_user.csv` is not written yet.** The evaluation
   skips it with a message. It is the only number in this package free of the
   paraphrase-template hazard, so the limitations above stay partly
   unquantified until it exists.
2. **`MAX_TURNS_BEFORE_ESCALATION = 8`** is a policy choice with no measurement
   behind it and it needs a turn counter, which is WP6's dialogue state. Out of
   scope for WP5.
3. **Darija.** 296 of 476 rows are `tn`, `tn_latn` or `mix`. Excluded from the
   reported numbers, consistent with WP4 and with the scope decision. The rows
   stay in the file.

## Reproducing it

```powershell
uv run python scripts/frustration/train_frustration.py     # models/frustration.pt
uv run python scripts/frustration/eval_frustration.py      # reports/frustration_eval.md
uv run python -m pytest tests/understand/test_frustration.py -q
```

The evaluation defaults to running the EmoWOZ arms, which needs
`scripts/frustration/build_emowoz.py` first and takes ~20 minutes. `--no-emowoz
--no-curve` gives the shipped numbers in a few minutes.

## Acceptance criteria

- [x] `detect(text, lang)` returns `{level, prob, lexicon_hits, source}`, with
      the probability in [0, 1] and `level` in `none`/`soft`/`hard`
- [x] grouped **and** ungrouped cross-validated scores both reported; the
      paraphrase leak is **+0.02** for both models
- [x] logistic regression compared against the torch MLP on identical
      embeddings, five seeds; the MLP ships on AUC 0.99 vs 0.98 and equal F1
      within noise, having been chosen for the brief's architecture
- [x] per-kind breakdown throughout, as counts; `worry` is 0/12 at both levels
- [x] soft and hard thresholds calibrated on out-of-fold probabilities, their
      runs recorded in `config.py`, under a stated policy
- [x] the `words`/`exclamations` ablation measured and **not** shipped
- [x] missing weights degrade to level `none` with source `no_model` rather
      than raising; a model trained with a different embedder is refused
- [x] tests: the three polarity pairs, a correction below hard, a lexicon hit
      reaching hard, unsupported language, empty text, and the level rule as a
      pure function — 36 tests
- [ ] the hand-written held-out set scored once (pending the file)
