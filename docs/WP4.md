# WP4 — Exam identification (bot level 1)

**Status:** steps 1–9 complete.
**Done when:** `identify(text)` returns one of the 173 exams, or a clarifying
question, measured on a test set that does not quote the catalogue.
**Graded block:** "Bot et recherche floue" — 4 pts.

UC2 (book a new appointment) cannot offer a slot until it knows which exam the
caller wants: the exam fixes the duration, the machine and the preparation.
This package only identifies the exam. It never books — that is WP6.

## Scope

| In | Out |
|---|---|
| Normalise + `lexicon.repair()` from WP3 | Booking and dialogue (WP6) |
| Modality / laterality / contrast extraction | Frustration detection (WP5) |
| RapidFuzz ranking over a keyword gazetteer | Live telephony |
| Clarify on close pairs, confusables, missing side | The agenda (already built) |
| LLM fallback over the top 5 candidates only | Choosing *when* to book the exam |
| `escalate` exams always transfer | Darija / Arabizi matching and tests |

## Language scope — French only

Settled: **French is the only language in scope for WP4**, and Darija is the
last thing the project will do, if it is done at all.

This is not just a decision about which tests to write — it changes what the
gazetteer indexes. The obvious shortcut is to index all three keyword columns
because they are already in the file, but `keywords_tn_latn` holds strings like
`irm ras`, `IRM mta3 mo5` and `IRM mta3 cerveau`. Indexing those puts five
extra near-duplicate entries per exam into the pool, and a French utterance can
win against an Arabizi keyword for the wrong exam. That does not just risk a
wrong match, it distorts every `margin` in the ranking — the number Decision 1
depends on.

So `build_exam_index.py` indexes **`keywords_fr` only**, behind a `--langs fr`
flag rather than a hard-coded column name. Adding Darija later is then a flag
and a rebuild, not a rewrite. Same reasoning as the Qdrant trade in WP3: pay
for where the data might go, but do not carry the cost today.

## What the catalogue actually contains

Measured from `data/raw/exams.csv` (173 rows) and `confusables.csv` (64 rows):

| Modality | Exams | | Laterality | Exams | | Contrast | Exams |
|---|---|---|---|---|---|---|---|
| radiographie | 46 | | aucune | 77 | | non | 149 |
| échographie | 41 | | droite | 39 | | oui | 18 |
| IRM | 39 | | gauche | 39 | | possible | 6 |
| scanner | 35 | | bilatéral | 18 | | | |
| mammographie | 6 | | | | | **escalate = 1** | **5** |
| dentaire | 4 | | | | | | |
| densitométrie | 2 | | | | | | |

Confusable pairs by type: **latéralité 39**, région 12, variante 7, modalité 6.

Three numbers in that table drive the design: 78 of 173 exams exist only as a
left/right pair, 39 of 64 confusable pairs are that same pair, and exactly
**five** exams are flagged `escalate`.

## Decision 1 — side and contrast are slots, not part of the name

The obvious implementation ranks the utterance against all 173 exams. Measured
objection: strip the side and injection words from `name_fr`, group on
(modality, remainder), and 173 exams collapse to **132 bases**:

| Bases | What they are |
|---|---|
| 91 | one exam, nothing to resolve |
| 39 | exactly a droite/gauche pair |
| 2 | exactly a ±injection pair (`IRM cérébrale`, `Scanner cérébral`) |

An earlier count in this document said 135 bases and 38 pairs. That was wrong:
the laterality pattern was written `\bdroit\b` and missed the plurals, so
`EX105 Radiographie des doigts droits` and `EX106 …gauches` were not recognised
as a pair.

Those 39 are coin flips for a fuzzy matcher. `IRM du genou droit` and `IRM du
genou gauche` differ by one short token out of five, so they score within a
point or two of each other on any string metric. Asking a confidence threshold
to separate them is asking the wrong question — the scores *should* be equal,
because the utterance genuinely does not say which one.

Contrast behaves the same way but is far narrower than expected, and this was
worth measuring before building a slot for it. The catalogue has exactly **one**
contrast surface form — "avec injection", on 14 strings — and **15 of the 18
exams whose `contrast` column says `oui` carry no marker in their name at
all**: `IRM hépatique`, `IRM cardiaque`, `Uroscanner`, `Coroscanner`, `Scanner
abdominal`.

So contrast is a property the radiologist decides, not something a caller
distinguishes by name. It gets the same slot treatment as laterality, but it
only ever fires on those 2 bases. Building a general contrast question would
have been machinery for a case that does not exist.

**So: match the base exam, then resolve side and contrast as slots.** Two
consequences worth stating in advance:

- The candidate set drops from 173 to 132, and 39 guaranteed ties leave the
  ranking entirely.
- `margin` becomes a meaningful number. A tie *between bases* is a real
  ambiguity (`IRM du rachis lombaire` vs `Scanner du rachis lombaire`) and
  calls for a question naming both. A tie *within* a base is a missing slot,
  which needs a different question ("Droit ou gauche ?") and must not be
  reported as low confidence — the match was certain, one field was absent.

Collapsing these two cases into one "scores are close" branch is the main
modelling mistake available here, and it would make the confidence threshold
uncalibratable, because it would be fitting one number to two unrelated
phenomena.

## Decision 2 — the test set is the hard part, not the matcher

The labelled data is the `new_appointment` rows of `intents.csv` that carry an
`exam_id`: **42 rows**, split `fr` 16, `mix` 6, `tn` 12, `tn_latn` 8. Under the
French-only scope, **22 are usable and 20 are out.**

The 6 `mix` rows count, because the part that matters is French:

```
سلام، je voudrais un rendez-vous pour irm mammaire bilatérale
عسلامة, le médecin m'a demandé scanner du genou droit, وقتاش تنجمو؟
```

The exam phrase is French in every one of them; only the greeting and the
closing question are not. That makes them a free robustness case rather than a
scope violation — non-French carrier tokens must not drag the ranking, and
these are the only rows that test it.

That leaves two problems, and the second is worse than the first.

**It is too small to carry the target.** On the 22 usable rows, one sentence is
4.5 points. "≥ 85 %" is 18.7 of 22. Any result stated to a tenth of a percent
on that set is noise.

**It quotes the catalogue.** The sentences were built by templating catalogue
names into carrier phrases, and it shows:

```
Bonjour, n7eb na5ou rdv pour pangonogramme (membres inférieurs entiers) svp
Bonsoir, pour scanner angioscanner jambes angio c'est quand le prochain créneau ?
Est-ce que je peux avoir un rdv pour écho des artères rénales doppler cette semaine ?
```

The first carries the catalogue's own parenthetical. No caller says
"pangonogramme (membres inférieurs entiers)". A gazetteer built from
`keywords_fr` will score near-perfectly on these, and the number will mean
nothing about a caller who says "une IRM pour mon dos".

This is the WP3 lesson repeating itself. Exam-name stripping scored **100 % on
a drag test built from the catalogue and 62.5 % on the real test set**, which is
why it was measured and not shipped. A circular test set does not just
overstate the score, it points the tuning in the wrong direction.

### So the evaluation is tiered

| Set | What it is | What it answers | Size |
|---|---|---|---|
| **T1** | catalogue leave-one-out: query each `keywords_fr` variant with its normalized forms removed from the index | does the gazetteer generalise across all 173 exams | 905 |
| **T2** | the 22 `fr` + `mix` corpus rows | the plan's own metric, reported with its circularity stated | 22 |
| **T3** | `exam_test_t3.csv` — hand-written caller language: "une radio des poumons", "une écho pour ma glande au cou", "une radio d'une seule dent" | the only honest accuracy number | 42 |
| **T4** | T2 and T3 with observed substitutions replayed from the accepted Whisper transcripts, followed by `lexicon.repair()` | how the matcher handles those measured text errors | 64 |

Leave-one-out for T1 is the same correction already applied to the intent
gate's similarity sweep in WP3: every example is in the index, so scoring
against an index that contains the query measures nothing.

**The headline numbers for the report are T3 and T4.** T1 is the generalisation
check, T2 is reported for comparability with the brief.

### T3 as delivered, and what "no leak" actually means

`data/raw/exam_test_t3.csv` — 42 hand-written sentences, 35 expecting `matched`
and 7 expecting `clarify`, with columns `exam_id`, `alt_ids`,
`expected_status`, `note`. The 7 clarify rows cover missing side, missing exam
type, missing spine level, a patient who is unsure between two modalities, a
trimester boundary, and the coroscanner / score-calcique pair.

The set was checked for copying before delivery. **The audit was re-run here
with a different metric, and the result needs stating precisely, because "no
leak" is two different claims:**

| Claim | Measured |
|---|---|
| no sentence quotes the catalogue's formal `name_fr` | **1 of 42** |
| no sentence contains any `keywords_fr` entry verbatim | **7 of 42** |

A plain similarity ratio misses the second, because the carrier words dilute
it: `fuzz.ratio` between "Une radio des côtes, je suis tombé dans les
escaliers" and `radio des côtes` is middling, while `partial_ratio` is 100. The
substring is what matters, so `partial_ratio` is the right metric for a leak
audit.

**But six of those seven are not leaks**, and the reason is the point of the
whole exercise. They match a `keywords_fr` entry that is *itself* written in
patient language, while staying far from the formal name:

| Sentence contains | keyword | `name_fr` | distance to name |
|---|---|---|---|
| radio des côtes | ✓ | Radiographie du grill costal | **48.9** |
| IRM de la prostate | ✓ | IRM prostatique | 75.0 |
| panoramique des dents | ✓ | Panoramique dentaire | 80.0 |
| radio du pied droit | ✓ | Radiographie du pied droit | 77.6 |

Nobody asks for a radiograph of their *grill costal*. The keyword lists exist
to hold the colloquial form, so a test set written in colloquial French
**cannot** be disjoint from them — and a test set that *was* disjoint would be
testing phrasings no caller uses. The disjointness goal was the wrong target;
distance from `name_fr` is the right one, and on that measure the set is clean.

That table is also the argument for Decision 4: at 48.9 similarity, a gazetteer
built from `name_fr` would never retrieve EX076 from a real caller.

The one sentence that does quote `name_fr` is `EX034` — "Un scanner ou une IRM
du genou droit, je ne sais plus" — and it is a `clarify` row. Its purpose is a
caller naming two modalities at once, not measuring retrieval, so the quote
costs nothing.

**Consequence for the report:** T3 accuracy is reported twice — over all 35
`matched` rows, and over the **30** with no verbatim keyword overlap. If the two
diverge, the 5 overlapping rows were flattering the score. One extra column,
and it closes the question rather than arguing about it.

One hard case is kept deliberately: "Un scanner du cœur" sits close to the
keyword `scanner du cou` of a *different* exam. That is a trap for the fuzzy
matcher, not a leak, and `cœur`/`cou` is exactly the kind of near-collision the
margin logic has to survive.

### Cases the corpus does not contain at all

- **Missing laterality.** Every lateral row in the corpus names the side
  ("coude droit", "poignet côté gauche"). There are *zero* examples of the
  case the clarify path exists for, so those must be constructed.
- **The escalate exam.** `EX139 Échographie appendiculaire` has no labelled
  sentence.
- **Unknown exam.** No row asks for something the centre does not do.

Constructed cases are fine for the tests. They are not fine for the accuracy
table, and will be kept out of it.

## Decision 2b — the modality filter must be soft, or the dental exams vanish

Step 2 of the pipeline extracts the modality and uses it to filter candidates.
Taken literally — filter on the `modality` column — it makes **all four dental
exams unreachable**, because every one of them is written in radio or scanner
vocabulary:

| Exam | `modality` | its own `keywords_fr` |
|---|---|---|
| `EX172` Radiographie rétro-alvéolaire | dentaire | radio d'une dent, radio de la dent |
| `EX171` Cone beam dentaire (CBCT) | dentaire | scanner dentaire cone beam, scanner 3D des dents |
| `EX170` Panoramique dentaire | dentaire | radio panoramique, radio dentaire panoramique |
| `EX173` Téléradiographie du crâne de profil | dentaire | radio du crâne de profil pour orthodontie |

`exam_test_t3.csv` contains "Il me faut une radio d'une seule dent" → `EX172`,
expected `matched`. A caller saying *radio* would have that exam filtered out
by its own `dentaire` label.

**Fix, read off the data rather than special-cased:** each base accepts the
modalities *its own phrases mention*, unioned with its column value. `EX172`'s
base ends up accepting `{dentaire, radiographie}`; `EX171`'s accepts
`{dentaire, scanner}`. No exam is reachable only through a label no caller
would use.

This generalises beyond dentistry — `Angioscanner`, `Uroscanner`,
`Entéroscanner` and `Coroscanner` are single tokens, so a word-boundary match
on "scanner" does not see them either, and six exams are named that way.

## Step 2–3 as built: choosing the scorer by measurement

The scorer was not chosen by reputation. The shapes being compared are
lopsided — the query is a whole sentence, a phrase is three or four words — and
scorers behave very differently under that mismatch. Top-1 and top-5 over the
35 matching rows of `exam_test_t3.csv`, ranking bases:

| scorer | top-1 | top-5 | mean margin |
|---|---|---|---|
| **token_set_ratio** | **27/35** | **34/35** | **18.2** |
| token_sort_ratio | 26/35 | 34/35 | 9.4 |
| WRatio | 24/35 | 28/35 | 4.1 |
| ratio | 21/35 | 32/35 | 6.6 |
| partial_ratio | 21/35 | 27/35 | 8.9 |
| partial_token_sort_ratio | 21/35 | 29/35 | 10.2 |
| partial_token_set_ratio | 3/35 | 10/35 | 0.0 |

Two things in that spread are worth more than the winner.

`ratio` and `WRatio` lose to the length mismatch — a sentence and a three-word
phrase read as dissimilar. `WRatio` is the general-purpose default and came
fourth.

The `partial_*` family fails the opposite way, and `partial_token_set_ratio`
catastrophically: any short phrase that is a token subset scores 100, so
"panoramique dentaire" won 11 of 42 rows outright and the mean margin went to
zero. **Same attractor effect the FAQ retrieval hit in WP3**, from the same
cause — short entries match everything.

### Filler words come off both sides

| | top-1 | top-5 | margin |
|---|---|---|---|
| base phrases | 25/35 | 31/35 | 17.0 |
| **filler removed** | **27/35** | **34/35** | **18.2** |

The top-5 gain is the valuable half, because **top-5 is the ceiling the LLM
fallback can ever reach** — it is shown five candidates and nothing else. The
index drops from 768 phrases to 480, since removing filler collapses wordings
that differed only by "faire une" and "du".

### Two fixes measured and not shipped

Both were my own proposals, both looked principled, both cost more than they
bought.

**Strip the modality word from the score**, on the argument that the modality
filter already uses it so scoring on it counts the same evidence twice. Margin
18.2 → 32.2, top-1 unchanged at 27/35, **top-5 34/35 → 32/35**. Without the
modality word, bases of different modalities become identical strings — "echo
cou" and "scanner du cou" both reduce to "cou" — and the filter cannot
compensate because it does not fire when the caller names no modality.

**Veto any candidate whose only shared word is the modality**, keeping its
contribution to the score but refusing a match that rests on it alone. Top-1
28/35, **top-5 32/35**, margin collapsed to 2.8, and six utterances left with
no candidate at all.

The reason it fails is the real shape of the remaining errors: when the
caller's word is absent from the gazetteer, the veto removes the **correct**
base as readily as the wrong one. For "une écho pour mon estomac",
`echographie abdominale` shares no content word either — "estomac" is not in
its keywords — so vetoing "echo cou" takes the right answer out with it.

### The margin is load-bearing, not a refinement

"Je veux un scanner" scores **100.0** against every one of the 35 scanners,
because the token intersection `{scanner}` compared against itself is a perfect
match. A score threshold cannot reject that at any value that still accepts
real utterances. What catches it is the margin being 0.0.

So `EXAM_FUZZY_THRESHOLD` and `EXAM_MARGIN` are not two views of the same
thing, and calibrating the threshold alone would leave the bot confidently
booking a dental cone beam for anyone who says "scanner".

### Where the gazetteer alone stands

**top-1 27/35 = 77.1 %, top-5 34/35 = 97.1 %** on T3, before any threshold
tuning and with no LLM.

Of the 8 residual misses, 6 are a **vocabulary gap** rather than a ranking
failure — the caller's word is not in `keywords_fr` at all:

| caller says | catalogue says | exam |
|---|---|---|
| estomac | abdomen | `EX121` |
| cou | cervical | `EX007` |
| cou | troncs supra-aortiques | `EX144` |
| bas-ventre | pelvienne | `EX126` |
| contrôle annuel | dépistage | `EX162` |
| mammographie et écho | bilan sénologique | `EX165` |

**These must not be fixed by adding T3's words to `keywords_fr`.** That would
turn the test set into a training set and make the headline number
meaningless — the exact circularity T3 was commissioned to avoid, arriving
through the back door. Recorded here because it is the tempting fix and it is
the wrong one.

Two legitimate routes, and the first is already in the plan:

1. **Let the LLM close it at step 6.** Mapping "estomac" to "abdominale" is
   French knowledge the model has and a gazetteer cannot. Top-5 is 34/35, so
   the ceiling is 97 %, and the gain is exactly the "gazetteer alone vs
   gazetteer + LLM" column the brief asks for.
2. **A colloquial anatomy synonym list written without reading T3** — general
   French body-part vocabulary, not these six rows. Non-circular, but it is a
   separate data task and it is not needed for the report.

## Step 4 as built: the decision layer

`identify()` runs six layers and the order carries the policy: nothing said →
rank → score floor → margin → slots → escalate. The initial no-LLM run
used floor 82 and margin 5 on the 42 T3 rows:

| | count |
|---|---|
| matched, right exam | **23** |
| matched, **wrong exam** | **0** |
| expected a match, asked instead | 12 |
| expected a question, asked | 5 of 7 |
| expected a question, matched instead | 2 |

Among the 35 rows that expected a match, every resolved exam was correct.
Two of the seven rows that expected a question were resolved prematurely.
The step 5 run reproduced those counts. It also verified all five flagged
exams transfer when named explicitly and that the listed `EX080`/`EX081`
pair produces the catalogue's two-choice question.

### Escalate is checked after resolution, and that position is load-bearing

Two of the five escalate exams are angio scanners sitting next to ordinary
scanners in the ranking, so filtering them out of the candidate pool would
rank the non-escalate neighbour instead and the transfer would never fire.

**Correction to an earlier claim in this document:** none of the five escalate
exams appear in `confusables.csv` at all, so `EX057` is not "both escalate and
confusable" as previously written here, and the clarify-versus-transfer
ordering question it was used to settle does not arise from that pair.

A question about "Un scanner du cœur" would let the caller distinguish
`EX057` (coroscanner, escalate) from `EX058` (score calcique, bookable).
The escalate check applies once the caller's exam resolves. The current
gazetteer resolves this utterance to `EX058`, as shown below.

### Contrast silence means the plain variant, not a question

Two T3 rows — "Une IRM de la tête, c'est pour des migraines" and "un scanner
de la tête après un choc" — land on the 2 bases that split on injection and
expect `matched`. So silence resolves to the plain variant. Three reasons:

- 149 of 173 exams are `contrast=non`.
- The error is cheap and self-correcting. A plain slot booked where injection
  turns out to be needed is caught when the prescription is read; an injected
  slot booked needlessly costs a longer appointment and reads out preparation
  instructions (bilan rénal, allergy and metformin warnings) for nothing.
- **Whether to inject is not the caller's to answer.** It is decided from the
  prescription by the radiologist, so asking the patient asks them something
  they usually cannot answer, and invites exactly the medical conversation
  this agent is built to transfer.

### A question is not automatically better than admitting confusion

The first build answered "une échographie pour voir le bébé, je suis enceinte
de trois mois" with *"Vous souhaitez Échographie cervicale ou Échographie de
la main ?"* — the right `clarify` status reached through two candidates nobody
asked for. Naming them invites a "yes" to the wrong exam, which is worse than
saying nothing useful.

The fix needed no new threshold: **only candidates that clear the accept floor
themselves may be named.** That row leads at 84.6 over a runner-up at 81.5,
and the floor is 82 — so one candidate qualifies, two are required, and it
falls through to the open question.

A bare request such as "Je veux un scanner" still scores 100 for several
bases and can ask about two arbitrary exams. The margin prevents a match,
but the question itself remains a limitation.

Step 8 also reproduced a limitation with shared wording. "IRM ou scanner
cérébral" resolves to `EX040`, while "IRM cérébrale ou scanner cérébral"
asks which modality the caller wants.

### The two remaining misses are both one missing data file

Both rows that should have asked and instead matched confidently are pairs
absent from `confusables.csv`:

| utterance | matched | should ask between | listed? |
|---|---|---|---|
| "Un scanner du cœur" | `EX058` Score calcique @ 100.0 | `EX057` Coroscanner / `EX058` | **no** |
| "Je dois faire une radio, c'est pour mon dos" | `EX082` rachis dorsal @ 100.0 | `EX080` / `EX082` / `EX083` | **no** |

Neither is a scoring failure. `EX082`'s own keyword list contains `radio dos`
and `EX083`'s contains `rx bas du dos`, so "mon dos" legitimately resolves to
the dorsal spine — the catalogue says it does. What is missing is the centre's
statement that the three rachis levels are a set a caller must choose between.
`confusables.csv` lists the two *dynamic* variants (`EX080`/`EX081`,
`EX083`/`EX084`) but no cross-level pair.

Raised as a data question below rather than patched, because `confusables.csv`
is a policy file — it records which pairs the centre wants confirmed — and
changing it to make a test pass would be the same circularity trap as editing
`keywords_fr`, with the difference that here the pairs are arguably genuinely
missing.

## Step 5 — no-LLM calibration

Run `uv run python scripts/calibrate_exam.py --floor 82 --margin 8` to
reproduce the sweep. T1 runs once per distinct catalogue keyword for each
exam, across all 173 exams. Each run removes the query's normalized and
lexicon-repaired forms from every base before ranking. This also removes aliases
shared by left/right or contrast variants. T2 uses the full French index and
the 22 labelled French and mixed corpus rows. T3 is a cross-check on caller
phrasing and expected clarifications.

| Setting (floor / margin) | T1 correct / wrong / withheld | T2 correct / wrong / withheld | T3 correct exam / premature / withheld | Named transfers |
|---|---:|---:|---:|---:|
| 75 / 8 | 265 / 141 / 499 | 20 / 0 / 2 | 25 / 2 / 15 | 5 / 5 |
| 82 / 5 (previous) | 280 / 166 / 459 | 20 / 0 / 2 | 23 / 2 / 17 | 5 / 5 |
| **82 / 8** | **231 / 137 / 537** | **20 / 0 / 2** | **23 / 2 / 17** | **5 / 5** |
| 85 / 8 | 218 / 136 / 551 | 20 / 0 / 2 | 23 / 2 / 17 | 5 / 5 |
| 82 / 10 | 199 / 119 / 587 | 19 / 0 / 3 | 23 / 2 / 17 | 5 / 5 |

At floor 82, margin 8 withholds 29 wrong T1 resolutions and 49 correct
ones compared with margin 5. It retains the T2 and T3 results and all five
named transfers. Margin 10 costs one correct T2 match. Raising the score
floor to 85 at margin 8 gives up 13 correct T1 resolutions to avoid one
wrong result. The two T2 misses (`S016` and `S001`) both tie at 100 and ask
for clarification. T1 measures catalogue synonym retrieval and shows how
often that retrieval can still be wrong. T2 includes catalogue wording and
its 20/22 should be read with that overlap in mind.

`EXAM_LLM_THRESHOLD = 65` sets the lower bound of the planned fallback band
`[65, 82)`. Of the 312 T1 and eight T3 cases in that band, the correct exam
is resolvable from the top five bases in 189 and six cases, respectively.
Using only three candidates drops T1 to 173 and keeps T3 at six. Raising the
lower bound to 70 drops T3 to five recoverable exams across six eligible
cases. T2 has zero cases in the band. These figures measure candidate
availability through slot resolution. Live recovery and latency are measured
after step 6. The candidate cap remains five and the fallback reuses
`LLM_TIMEOUT_S`.

## Decision 4 — the gazetteer is not the WP3 lexicon

They read the same file and must stay separate artefacts:

| | reads | produces | used to |
|---|---|---|---|
| `build_lexicon.py` (WP3) | `modality`, `region_fr`, `name_fr` | single tokens | **rewrite** a mangled token, conservatively |
| `build_exam_index.py` (WP4) | `keywords_fr` (see Language scope) | whole phrases | **rank** 135 base candidates |

Order is fixed and one-directional: repair runs first (pipeline step 1), then
matching. Which means **a wrong repair is a wrong exam** — already the stated
reason `REPAIR_MIN_RATIO` sits at 0.88 and why domain repair refuses to touch a
real French word.

## Step 6 — LLM fallback

Reuses `llm.py` exactly as WP3 built it: hard wall-clock budget through
`run_within()`, disk cache, usage counters, `complete_or()` for the fallback.

- The default fallback band is `[65, 82)`. A listed close pair or an unresolved
  leading base keeps the clarification path. The request sends at most one
  slot-resolved exam ID and name from each of the five highest-ranked bases.
- The model returns a JSON object with `exam_id` (an offered ID or `null`)
  and a string `reason`. Every reply, including a cache hit, is checked
  against the offered IDs. A selected exam goes through the same escalation
  check as a gazetteer match.
- Timeout, model error, malformed output, abstention, or an unoffered ID
  returns the open clarification question. `trace` records the outcome.
- The cache key includes the normalized utterance, ordered candidate IDs and
  names, system instructions, and model. `identify(..., use_llm=False)`
  reproduces the gazetteer-only path used for calibration.

One planning input carried over, not a footnote: WP3 measured the Gemini
timeout rate at **44–62 % under load** with a 2 s budget, and the same prompt
returning in 0.89 s once and 33.4 s another time. **The no-LLM path has to be
good on its own.** The LLM is a recovery layer, and the gazetteer-alone column
in the accuracy table is the one that matters for the demo.

Budget note (Rule 4, one LLM call per turn): WP3 spends the turn's call on the
FAQ answer. A booking turn does not call that, so the exam fallback is that
turn's one call. WP6 merges both into a single prompt and schema — recorded in
[WP6.md](WP6.md).

## Step 7 — matcher evaluation

Run `uv run python scripts/eval_exam.py` from the project root, with
`reports/stt_results.csv` and the generated lexicon available. The script
runs both modes on the same inputs and writes a summary and per-case CSV.
Use `--no-llm` for an offline run with separate output filenames.

T4 starts with the 64 T2 and T3 requests. It replays the longest applicable
measured substitution per request, using Whisper small/raw transcripts that
passed the audio gate. It keeps adjacent alignment edits together and retains
unchanged rows as controls. The report also measures the changed rows on
their own. This is a text simulation with limited coverage of recognition
errors. The replay changes 14 of the 64 inputs in the current corpus.

The [full evaluation](../reports/exam_eval.md) records both modes on all four
tiers. On T3 the fallback raised correct IDs from 23/35 to 24/35. On T4 it
raised them from 37/57 to 38/57, including one additional correct match among
the 14 changed inputs. Both modes still resolved two requests that called for
clarification. The T3 keyword-disjoint subset improved from 18/30 to 19/30.

Only 12 of 307 fallback invocations received live replies. There were 41
timeouts and 254 backend errors, all of the latter during T1. These results
measure the fallback's end-to-end effect under the service conditions of
this run. They provide limited evidence about candidate selection when
Gemini answers. The detailed cases are in the generated
`reports/exam_eval.csv`.

## Clarify, not guess

Three reasons, three different questions, kept distinct in `trace`:

| Reason | Question | Source |
|---|---|---|
| missing side on a lateral exam | "Droit ou gauche ?" | the exam's `laterality` |
| confusable pair | names both | `confusables.csv` `label` |
| low score after the LLM | names the top candidates | the ranking |

The asymmetry is the same one that shapes the safety layer: a wrong exam books
the wrong machine for the wrong duration and the caller finds out on arrival; a
question costs one turn. **So clarifying is cheap and guessing is not**, and the
wrong-exam rate is the number to minimise even at the cost of the clarify rate.

## `escalate` — now a rule, not a single flag

Five exams carry it, and `exams.csv` gained an `escalate_reason` column so the
rule is readable from the data rather than inferred from which ids are flagged.

| Exam | Reason |
|---|---|
| `EX139` Échographie appendiculaire | urgence : suspicion d'appendicite |
| `EX042` Angioscanner cérébral | urgence : suspicion d'AVC |
| `EX052` Angioscanner thoracique | urgence : suspicion d'embolie pulmonaire |
| `EX057` Coroscanner | validation médicale : prémédication pour ralentir le cœur |
| `EX020` IRM cardiaque | validation médicale : protocole adapté au patient |

The stated rule: **the bot never books an exam that is urgent by nature, or
that needs a doctor's validation first.** The centre is fictional, so this is a
defensible operating rule rather than a clinical criterion, and it is written
down as such.

Two things follow:

- The criterion becomes measurable instead of being a single regression guard.
- Two of the five are `scanner` angio exams, which sit next to ordinary
  scanners in the ranking. So `escalate` has to be checked **after** the match
  resolves and before anything is offered — not as a filter on the candidate
  pool, or the matcher would silently rank the non-escalate neighbour instead
  and the transfer would never fire.

`EX057 Coroscanner` transfers when the caller identifies it. The current
confusable list contains none of the five flagged exams.

## Plan of work

1. ~~`scripts/build_exam_index.py` — gazetteer from `keywords_fr`, with the
   base/side/contrast decomposition from Decision 1.~~ **Done.** Builds
   `data/processed/exam_index.json` (136 KB) in under a second: 173 exams →
   132 bases, 768 deduplicated phrases, 64 confusable pairs, 5 escalate exams.
   The builder asserts every exam lands in exactly one base and that no base is
   phraseless, because a hole there would surface only as an unexplained miss
   that threshold tuning would chase forever. `src/hermes/exam_bot.py` carries
   the normalisation primitives both the builder and the matcher need, plus the
   `ExamMatch` result type.
2. ~~`src/hermes/exam_bot.py` — normalise, extract attributes, rank.~~
   **Done.** `prepare()` runs `lexicon.repair()` then reads modality,
   laterality and contrast; `rank()` scores every candidate base and
   `resolve()` turns a base into one exam id or names the missing slot.
3. ~~Candidate filtering by modality and resolved slots.~~ **Done**, and the
   modality filter is soft: it narrows the pool when it can and steps out of
   the way when the narrowed pool would be empty.
4. **Done.** Verified the decision layer, clarification reasons and all five
   transfers. The two missing ambiguity rules are recorded above.
5. **Done.** Evaluated T1 and T2, cross-checked T3 and recorded the four
   settings and measurements in `config.py`.
6. **Done.** LLM fallback, shortlist validation and caching. Offline tests
   cover valid choices, rejected replies, timeouts, ambiguity and transfers.
7. **Done.** Compared gazetteer and live fallback on T1–T4 and wrote the final
   tables to `reports/`.
8. **Done.** Offline tests cover exact names, aliases, a typo, abbreviations
   (TDM, écho), missing side, confusable pairs, an unknown exam, transfers,
   contrast and a mixed-language request with a French exam phrase. The
   local catalogue check rebuilds the index and verifies all five transfers.
9. **Done.** The Gradio **Examens** tab shows the identified exam,
   clarification or transfer, the ranked candidates, and the decision trace.
   Run `uv run python app/gradio_app.py` and open **Examens**. Try
   `IRM du genou droit`, `IRM du genou`, or `Coroscanner`. The LLM starts
   disabled in this tab and can be enabled for an individual test.

Thresholds are calibrated at step 5, before the LLM exists, so the
gazetteer-alone number is measured on its own and not after the fact.

## Data state

Two policy questions remain above for the centre to confirm, covering cardiac
CT wording and unspecified spine level.

| Asked for | Delivered |
|---|---|
| ~40 non-circular French test sentences | `exam_test_t3.csv`, 42 rows, hand-written, audited above |
| 5–8 `escalate` rows under a stated rule | 5 rows plus an `escalate_reason` column |
| Language scope | French only, `keywords_fr` only, no Darija tests |

`exams.csv` is otherwise unchanged — 173 rows, same ids and names, only the
`escalate` flag and the new column differ. Nothing the agenda or the WP3
lexicon reads was touched: `build_lexicon.py` takes `modality`, `region_fr` and
`name_fr`, and the agenda takes `duration_min`, so neither needs rebuilding.

## Acceptance criteria, with the measurement attached

- [x] `identify()` implements the gazetteer decision layer and returns `ExamMatch`
- [x] base/side/contrast decomposition removes 39 within-base lateral ties
- [x] confusable pairs and missing laterality produce distinct questions
- [x] the LLM sees at most 5 candidates and the bot rejects ids outside that shortlist
- [x] all five `escalate` exams transfer, checked after the match resolves
- [x] calibration for the four settings recorded in `config.py`
- [x] accuracy table: gazetteer alone vs gazetteer + LLM, on T1 / T2 / T3 / T4
- [x] T3 reported over all 35 `matched` rows and the 30 with no keyword overlap
- [x] clarify rate and **wrong-exam rate** reported
- [x] LLM usage rate, latency and timeout rate reported
- [x] tests for exact name, typo, abbreviation, missing side, confusable pair,
      unknown exam, escalate exam
