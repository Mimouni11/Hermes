# hermes-voice — documentation

A voice agent for a medical imaging centre call line. The caller speaks, the
agent identifies which of 173 exams they mean, books or changes the
appointment, and hands the call to a human the moment it leaves its safe scope.

These pages document what is built and the decisions behind it. They are
written for someone picking the project up — a teammate, a marker, or the
author in a week's time.

| Page | What it covers |
|---|---|
| [configuration.md](configuration.md) | `config.py`, the `.env` keys, every threshold and where its value came from |
| [agenda.md](agenda.md) | The appointment book: slots, multi-slot exams, holds, the booking rules |
| [audio-test-set.md](audio-test-set.md) | How the 45-clip French test set is built and regenerated |
| [stt.md](stt.md) | The audio quality gate, the Whisper wrapper, and the measured results |
| [tts.md](tts.md) | The agent's voice: Piper, the cache, and reading a booking aloud |
| [WP3.md](WP3.md) | Intent gate, the three-layer medical safety split, FAQ retrieval and the answer path |
| [WP4.md](WP4.md) | Exam identification: the gazetteer, fuzzy matching, the four test sets and the LLM fallback |
| [WP5.md](WP5.md) | Frustration detection: the lexicon baseline, the torch MLP ensemble, the two levels, and EmoWOZ as a negative result |
| [WP6.md](WP6.md) | What was deferred on purpose: dialogue state, the merged LLM call |

## Where the project stands

`Hermes_plan_projet_3.pdf` is the **initial** plan, written 21 Nov. It is a
snapshot, not a contract, and it is not kept up to date.

**These pages are the living record.** Any change of scope, any decision that
departs from the plan, and any threshold that gets recalibrated is written down
here, with the reason. Where this documentation and the PDF disagree, this
documentation is right.

The plan splits the work into seven packages.

| WP | What it is | State |
|---|---|---|
| WP1 | Data: catalogue, intents, FAQ, frustration set, test audio | **done** |
| WP2 | Speech: STT with confidence and WER, SNR quality check, TTS reply | **done** — the audio chain runs end to end |
| WP3 | Intent gate (6 intents) + FAQ RAG | **done** — routing, medical safety, retrieval and the spoken answer |
| WP4 | Bot level 1: gazetteer + RapidFuzz exam identification | **done** — 0 wrong exams on every hand-written test set |
| WP5 | Frustration detector on sentence embeddings | **done** — torch MLP ×5, F1 0.93 / AUC 0.99, soft and hard levels |
| WP6 | Agent level 2: one LLM call per turn, JSON out | not started |
| WP7 | Escalation, fixed safety script, anonymised logs | not started |

The appointment book is built and tested ahead of WP6, which needs it.

## Layout

```
src/hermes/        the package: one module per pipeline stage
scripts/           everything that builds data or measures something
tests/             pytest; 139 passing, the rest skipped until implemented
data/raw/          the corpora (not in git)
data/audio/        the test clips and their manifest (not in git)
models/            Piper voice, trained weights (not in git)
reports/           evaluation output: *.md summaries in git, *.csv tables not
```

## Running it

```powershell
uv venv --python 3.11
uv sync --extra dev
uv run python -m pytest -q
```

Then build the derived artefacts. **None of these are in git** — they are
generated from `data/raw/`, so a fresh clone has to make them before the
agent can route a call, answer a question or identify an exam:

```powershell
docker compose up -d                              # Qdrant, for the two indexes
uv run python scripts/build_lexicon.py            # data/processed/lexicon.json
uv run python scripts/build_exam_index.py         # data/processed/exam_index.json
uv run python scripts/build_intent_index.py       # the hermes_intents collection
uv run python scripts/build_faq_index.py          # the hermes_faq collection
uv run python scripts/train_intent_classifier.py  # models/intent_classifier.joblib
```

Without the lexicon the repair layer raises; without `exam_index.json` the
Examens tab reports an identification error; without the collections the
intent gate and FAQ tabs skip. The tests skip rather than fail when an
artefact is missing, so a green suite does not mean the bench will run.

uv throughout, never pip. Python 3.11. Every dependency is free, and the
Gemini key lives in `.env`, which is gitignored.

## What is not in git

Nothing under `data/` or `models/`, no project handout — only the `.gitkeep`
files that carry the directory layout.

`reports/` is split by audience. The `.md` summaries **are** committed: they
are short, they change only when a measurement changes, and the docs and the
final report quote their numbers, so they deserve review like any other prose.
The `.csv` per-case tables are not: they run to thousands of rows and are
rewritten in full on every run. The corpora have to
be put back by hand on a fresh clone before the build scripts will run. The
audio set is the exception: it regenerates itself, because
`pick_audio_sentences.py` is seeded and returns the same 20 sentences and the
same split on any machine.
