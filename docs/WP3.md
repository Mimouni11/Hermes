# WP3 — Intent gate and FAQ retrieval

**Status:** planned, not started.
**Done when:** an audio question gets a spoken FAQ answer, end to end.

This is the first use case to run the whole chain: audio in, quality gate, STT,
routing, retrieval, spoken reply. WP2 supplies both ends; this package supplies
the middle.

## Scope

| In | Out |
|---|---|
| Intent gate: 6 intents, kNN over embeddings | Exam identification (WP4) |
| Out-of-scope detection with a calibrated τ | The LLM agent (WP6) |
| Ambiguity handling when the top 2 intents are close | Frustration detection (WP5) |
| FAQ retrieval: top-3 with τ_faq | Booking (the agenda is already built) |
| Medical questions routed to a human, always | The fixed safety script (WP7) |
| A Gradio page to test all of it | |

## Deviation from the plan: Qdrant

The plan says "embed intent examples (MiniLM); kNN cosine routing". It names no
vector database, and at today's size one is not needed — 245 intent sentences
and 24 FAQ entries fit in a numpy matrix, where a cosine against all of them is
a single matrix multiply and faster than a network round trip to a service.

**Qdrant is chosen for where the data is going, not where it is.** The Darija
side is not written yet, the corpora are expected to grow substantially, and
the FAQ is the part most likely to keep growing after the project is handed in.
Moving to a vector store later means rewriting the retrieval layer at the point
when there is least time; doing it now costs a container.

Recorded as a deliberate trade: slower and heavier today, in exchange for not
having to change it twice.

**Running it:** a Docker container, `qdrant/qdrant`, on `localhost:6333`.

## Embeddings

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, already the
configured `EMBEDDING_MODEL`. 384 dimensions, cosine distance.

Multilingual matters here, and not as a nice-to-have. The corpora carry four
language tags:

| lang | meaning | intent rows | example |
|---|---|---|---|
| `fr` | French | 100 | "Je veux prendre un rendez-vous" |
| `tn` | Darija, Arabic script | 65 | "قداش تحلّو وقداش تسكّرو؟" |
| `tn_latn` | Arabizi, Latin script | 53 | "wa9tech t7ellou w tsakrou?" |
| `mix` | French–Darija code-switching | 27 | |

MiniLM handles French and Arabic script. **Arabizi is the open risk** — `t7ellou`
and `wa9tech` use digits for Arabic letters, which no multilingual model is
trained on in quantity. Expect the `tn_latn` rows to retrieve worst, measure it
separately, and report the gap per language as the plan requires.

## Collections

### `hermes_intents`

One point per intent example — all 245 rows, every language.

```
vector   384, cosine
payload  { id, text, intent, sub_intent, lang, exam_id }
```

### `hermes_faq`

One point **per question variant**, not per entry. `faq.csv` has 24 entries,
each with a French, Darija and Arabizi question, so up to 72 points all
pointing at the same 24 answers. A Darija question has to match a Darija
question vector, not a French one.

```
vector   384, cosine
payload  { faq_id, question, lang, answer_fr, answer_tn }
```

The answer is carried in the payload rather than fetched separately: at this
size there is no reason for a second lookup.

## The intent gate, as built

### The labels are the corpus's

An earlier sketch of `intent_gate.py` named its own intents — `faq`, `book`,
`reschedule`, `results`, `small_talk` — which matched nothing in
`intents.csv` and had **no `medical_question` at all**, the one category the
brief makes safety-critical. Its `results` label was the second hazard: "où je
récupère mes résultats" is logistics and "mon résultat est normal ?" is a
diagnosis request, and one label cannot hold both. The enum is now exactly the
six labels in the data.

### Routing is a trained classifier, not the neighbours

Measured on the 209 French examples, 5-fold cross-validated:

| method | accuracy | macro F1 | medical recall |
|---|---|---|---|
| kNN k=5 (cosine) | 68.9% | 0.650 | 79.5% |
| kNN k=1 (cosine) | 72.7% | 0.703 | 74.4% |
| logistic regression | 78.0% | 0.759 | 87.2% |
| **logistic regression, class-balanced** | **78.9%** | **0.793** | **92.3%** |
| MLP 256 | 80.4% | 0.785 | 84.6% |

The MLP has the best raw accuracy and is still the wrong choice: it loses on
macro F1 and on medical recall. Accuracy rewards getting the big class right,
and `info` is 87 of the 209 French examples. `class_weight="balanced"` is what
buys back the recall that matters.

Qdrant keeps its job: the classifier says *which* intent, the nearest
neighbour's cosine says *whether the caller said anything we have seen before*.
Those are different questions and one number cannot answer both.

### Medical safety is three layers, not one

92.3% recall means roughly one medical question in twelve reaches the booking
branch. The three that leaked were:

```
"Je suis enceinte, est-ce que la radio est dangereuse pour le bébé ?"
"Est-ce que mon résultat est normal ?"
"Vous pouvez m'expliquer mon compte rendu ?"
```

1. **Deterministic rules, before any routing** (`escalation.py`). Patterns for
   implants, pregnancy, requests for a judgement, report interpretation,
   medication and allergies, in French and Darija. Alone they catch **69.2%**
   of medical questions at a **3.5%** false-alarm rate — and they catch all
   three leaks above, so the two layers together cover 100% of the held-out
   medical questions.
2. **An asymmetric probability floor.** If `P(medical) ≥ 0.25` the turn is
   medical even when another intent scores higher. A false alarm costs one
   transfer; a miss costs safety.
3. **The post-LLM check**, which CLAUDE.md already requires, as the last line.

The rules deliberately do *not* fire on results logistics — "comment récupérer
mes résultats", "mes résultats sont prêts quand", "vous envoyez par mail" are
all `info` and stay `info`. Strong patterns (an implant, a pregnancy) override
that; weak ones (the word "résultat") do not.

### Short utterances never reach the embedder

"oui", "hmm", "euh...", "allô ?" all retrieved "Opérateur !" at about **0.78**,
higher than almost any genuine in-scope example — short strings embed near
short strings whatever they mean. Six such sentences, 19% of the out-of-scope
set, sat at the very top of the score range and made τ uncalibratable. They are
now rejected on length before anything is embedded.

A short reply *inside* a booking ("oui", "le deuxième") is a different thing
and belongs to dialogue state, which is not built yet.

### Measured end to end

| | result |
|---|---|
| out-of-scope handled safely (transfer, rephrase or clarify) | **90.6%** (29/32) |
| medical questions transferred | **100%** (39/39) |
| intent accuracy, cross-validated | **78.9%** |
| cost: `info` questions transferred unnecessarily | ~14% |

**The 100% medical figure is honest but the routing figure needs care.** The
classifier is fitted on the same 209 examples, so its accuracy *on them* is
92% and meaningless; 78.9% is the cross-validated number and the one to quote.
The medical figure survives because the rules that close the gap are not
fitted to anything.

The three out-of-scope sentences that still get routed are the hardest by
design — "Je veux annuler mon abonnement Ooredoo" and "Je veux confirmer ma
commande Jumia" are `unrelated_lookalike`, built to look like a cancellation
and a confirmation.

The 14% of `info` questions pulled into a transfer is the price of the 0.25
floor, paid deliberately.

## Routing

For one transcript:

1. embed it;
2. search `hermes_intents`, k = 5;
3. **best score < τ** → out of scope. Ask the caller to rephrase **once**, then
   transfer to a human;
4. **top-1 and top-2 within `INTENT_MARGIN`** → ask a clarifying question
   rather than guessing;
5. intent is `medical_question` → **transfer immediately**. Never answered,
   whatever the confidence. This is the brief's guiding principle, not a
   threshold;
6. intent is `info` → search `hermes_faq`, top 3. Best < τ_faq → say so and
   transfer rather than answer from a weak match;
7. otherwise → hand to the branch that owns the intent.

Dialogue state is needed so a short reply ("oui", "le deuxième") is read inside
the booking that is already in progress instead of being routed from scratch.

## Thresholds to calibrate

All four are **placeholders** today, carried over from the plan. None has been
measured.

| Key | Placeholder | Calibrate against | Target |
|---|---|---|---|
| `INTENT_ACCEPT_THRESHOLD` (τ) | 0.55 | `oos_validation.csv`, 60 rows | out-of-scope recall as high as possible |
| `INTENT_MARGIN` | 0.08 | the intent set itself | fewer clarifying questions without more wrong routes |
| `FAQ_ACCEPT_THRESHOLD` (τ_faq) | **0.575, calibrated** | `faq_test.csv`, 31 French rows | see below |

`oos_validation.csv` carries an `oos_type` column, so out-of-scope recall can
be reported per kind of question rather than as one number.

The same approach as the audio gate: bin the metric, find where it crosses the
usable limit, write the number into `config.py` with the measurement beside it.

### FAQ retrieval, measured

French only, 31 test queries: **24 answerable, 7 unanswerable by design**. The
unanswerable ones are not failures — refusing them is the "don't know" rate the
plan asks for, and counting them as misses understates accuracy badly.

| | result |
|---|---|
| top-1 on answerable | **75.0%** |
| top-3 on answerable | **87.5%** |
| τ_faq | **0.575** |
| answered at that threshold | 12 of 24, at 85.7% precision |
| unanswerable refused | 100% |

Filtering the search to the query's language helps and is kept: top-3 goes from
64.5% to 67.7%, since French queries were otherwise matching Darija variants.

The score bands overlap — correct retrievals average 0.615 but bottom out at
0.468, while wrong ones reach 0.642 — so no threshold is clean. 0.575 is chosen
on the safe side: below it the call goes to a human.

**The honest reading: half of answerable info questions get transferred.** That
is a retrieval ceiling, not a threshold choice, and the levers are more question
paraphrases per entry, or a stronger embedding model.

## What gets measured

From the plan's evaluation section:

- **Intent gate** — F1 per intent, confusion matrix, out-of-scope recall, and
  the false-reject against false-accept curve as τ moves.
- **FAQ RAG** — retrieval top-1 and top-3 accuracy, plus the "don't know" rate
  on questions that have no answer.
- **Per language.** Every metric split `fr` / `tn` / `tn_latn` / `mix`, so the
  French–Darija gap is visible rather than averaged away.

## The Gradio testing app

Not a demo page — a bench for seeing every stage work.

- **Input:** an audio file, or the microphone. Both paths identical downstream.
- **Quality gate:** verdict, estimated SNR, and the reason when rejected.
- **STT:** the transcript, confidence, `no_speech_prob`, decode latency.
- **Intent:** the chosen intent, the top-5 with scores, and which rule fired —
  accepted, out of scope, ambiguous, or straight to a human.
- **FAQ:** the top-3 matches with scores and their `faq_id`.
- **Reply:** the text, and Piper's audio to play and download.
- **Latency:** a breakdown per stage, since the brief's 5-second budget is
  already tight at 2.46 s for STT alone.
- **Text-only mode:** type a sentence and see the routing, so the retrieval can
  be debugged without speaking.

## Files

| Path | State |
|---|---|
| `src/hermes/embeddings.py` | new — load the model once, encode, cache |
| `src/hermes/vector_store.py` | new — Qdrant client, collections, upsert, search |
| `src/hermes/intent_gate.py` | stub → implement |
| `src/hermes/faq_rag.py` | stub → implement |
| `scripts/build_intent_centroids.py` | stub → repurpose as the intent indexer |
| `scripts/build_faq_index.py` | stub → implement as the FAQ indexer |
| `scripts/calibrate_intent.py` | new — τ and the margin, like `calibrate_gate.py` |
| `app/gradio_app.py` | stub → the testing app |
| `tests/test_intent_gate.py`, `tests/test_faq_rag.py` | stubs → implement |
| `docker-compose.yml` | new — the Qdrant service |

## Config additions

`QDRANT_HOST`, `QDRANT_PORT`, the two collection names, `INTENT_TOP_K`,
`FAQ_TOP_K`, and the embedding dimension. Same rule as everything else: a
default in `config.py`, an optional `.env` override, no `os.getenv` elsewhere.

## Order of work

1. Qdrant running, `vector_store.py` and `embeddings.py`, with the tests
   pointing at an in-memory or throwaway collection.
2. The two index builders, run once over the corpora.
3. The intent gate, then calibrate τ and the margin.
4. FAQ retrieval, then calibrate τ_faq.
5. The Gradio bench.
6. Per-language metrics for the report.

Steps 3 and 4 each end with numbers written into `config.py`, not with a
threshold someone liked the look of.

## Open questions

- **Arabizi.** If `tn_latn` retrieval is much worse than `tn`, the options are
  transliterating Arabizi to Arabic script before embedding, or keeping a
  separate lexicon. Measure before deciding.
- **One collection or one per language?** Starting with `lang` in the payload
  and filtering if needed; splitting later is cheap, merging is not.
- ~~Embed the FAQ question only, or question and answer together?~~
  **Settled: question only.** Both were built and measured on the 31 French
  rows of `faq_test.csv`. At the operating point where each refuses every
  unanswerable question, question-only answers 12 of 24 correctly at 85.7%
  precision, question+answer 11 at 78.6%. The answer text does pull in false
  matches, as the usual reason predicts — an answer listing "8h à 18h" and
  "samedi" hijacks any query mentioning a day or a time. `--embed
  question+answer` remains in the builder so the comparison can be re-run.
- **Does the language need detecting**, or does the caller's language come from
  the STT result? Whisper returns a language code; whether it is reliable on a
  code-switched sentence is untested.
