# WP5 — frustration detection

**Scope: French only.** 180 rows of `frustration.csv` with `lang == fr`, 85 frustrated and 95 calm. The 296 Darija, Arabizi and mixed rows are not used — measured reason: 53 of the 56 `mix` rows carry their frustration in Darija ("yezzi", "fadit mel robot", "ma tefhem chay"), so a French reading of them has almost nothing to work from.

Every split is `StratifiedGroupKFold` on `group`. 75 of the 180 rows sit in a paraphrase group, so an ungrouped split trains on some wordings of a complaint and tests on the rest.

## Models

| training set | model | split | P | R | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| lexicon baseline | 84 terms in 9 families | grouped | 1.00 | 0.61 | 0.76 | — |
| (a) French only | logistic regression | grouped | 0.93±0.01 | 0.93±0.02 | 0.93±0.01 | 0.98±0.00 |
| (a) French only | torch MLP x5 | grouped | 0.93±0.01 | 0.92±0.01 | 0.92±0.01 | 0.99±0.00 |
| (a) French only | logistic regression | ungrouped | 0.94±0.01 | 0.96±0.00 | 0.95±0.00 | 0.99±0.00 |
| (a) French only | torch MLP x5 | ungrouped | 0.93±0.01 | 0.96±0.01 | 0.95±0.01 | 0.99±0.00 |
| (ship) inner-CV epochs | torch MLP x5 | grouped | 0.93±0.01 | 0.92±0.02 | 0.93±0.02 | 0.99±0.00 |

**Shipped: the torch MLP ensemble** — one hidden layer of 48, dropout 0.3, AdamW weight decay, 5 members averaged, epochs chosen per fold by inner grouped CV. Logistic regression and the lexicon are baselines.

### The paraphrase leak

| model | grouped | ungrouped | inflation |
|---|---|---|---|
| logistic regression | 0.93 | 0.95 | +0.02 |
| torch MLP x5 | 0.92 | 0.95 | +0.02 |

### Epochs, chosen rather than asserted

Inner grouped CV inside each outer training fold, 25 selections (5 seeds x 5 folds).

| epochs | chosen | mean inner validation loss |
|---|---:|---:|
| 50 | 24/25 | 0.2812 |
| 100 | 0/25 | 0.2999 |
| 200 | 1/25 | 0.3207 |
| 400 | 0/25 | 0.3502 |

The loss rises monotonically with epochs, so a fixed 200 — the first draft's value — was four times too many.

## Thresholds

Swept at 0.02 on the shipped model's out-of-fold probabilities. Counts are means over 5 seeds.

| tau | recall | precision | F1 | frustrated | frust_hard | correction | worry | neutral | positive |
|---|---|---|---|---|---|---|---|---|---|
| 0.10 | 0.98 | 0.88 | 0.93 | 74.4/75 | 9.0/10 | 6.8/11 | 0.8/12 | 2.6/54 | 1.0/11 |
| 0.20 | 0.97 | 0.90 | 0.93 | 74.0/75 | 8.2/10 | 6.2/11 | 0.0/12 | 1.8/54 | 1.0/11 |
| 0.40 | 0.93 | 0.92 | 0.92 | 72.2/75 | 6.8/10 | 4.6/11 | 0.0/12 | 1.2/54 | 1.0/11 |
| 0.80 | 0.87 | 0.96 | 0.91 | 68.2/75 | 5.6/10 | 2.4/11 | 0.0/12 | 0.0/54 | 1.0/11 |

### The two levels, and the rule

```
soft  if  prob >= 0.62
hard  if  prob >= 0.98  OR  (lexicon hit AND prob >= 0.62)
```

| rule | tau | recall | precision | F1 | correction | worry | positive | missed |
|---|---|---|---|---|---|---|---|---|
| hard alone, calm <= 1 | 0.98 | 0.71 | 0.98 | 0.83 | 1.0/11 | 0.0/12 | 0.2/11 | 24.4/85 |
| hard alone, calm <= 2 | 0.86 | 0.85 | 0.96 | 0.90 | 1.8/11 | 0.0/12 | 1.0/11 | 12.8/85 |
| HYBRID: prob OR (lexicon AND >= soft) | 0.98 | 0.78 | 0.98 | 0.87 | 1.0/11 | 0.0/12 | 0.2/11 | 18.6/85 |

The hybrid's second arm is worth its place: at the same threshold and the same false alarms as the probability alone, it catches six more frustrated callers. It only became affordable once the lexicon's repetition family was restricted to the first person — while it also matched the bare infinitive it wrongly flagged "Vous pouvez répéter s'il vous plaît ?", a `correction`, and the threshold was already spending its whole correction budget.

**The hard level is one of three paths to a human.** Two consecutive soft hits transfer, and so does the WP6/WP7 repeat counter. A row missed here is a delayed handover, not a lost one.

## Limitations

- **Recall 0.91 holds for corpus-like phrasing.** The corpus's frustrated rows run on a narrow vocabulary, and probes with wording outside it are hit and miss. The hand-written held-out set is the real measure.
- The corpus is visibly templated — `Sérieusement, / Oh là là, / Écoutez,` prefixed onto a core complaint. That is what `group` encodes and why the grouped split is load-bearing.
- Two calm rows are confidently misread: `Non non, c'est bon` at 0.975 and `Oui ça marche !` at 0.943. Both are correctly labelled, so this is model error. It is **not** general negation blindness: on fresh polarity pairs the model separates `ça marche` 0.024 from `ça ne marche pas` 0.700.
- Probabilities are heavily saturated. Real decisions happen in a thin band, which makes `oui j'ai compris` at 0.597 uncomfortably close to the soft threshold.
- 180 rows. One row is 0.6 points, so **no figure here is meaningful to more than two decimals**, and the hard kinds (7 to 12 rows each) are reported as counts, never percentages.
