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
| WP3 | Intent gate (6 intents) + FAQ RAG | not started |
| WP4 | Bot level 1: gazetteer + RapidFuzz exam identification | not started |
| WP5 | Frustration detector (MLP on sentence embeddings) | not started |
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
reports/           generated evaluation tables (not in git)
```

## Running it

```powershell
uv venv --python 3.11
uv sync
uv run python -m pytest -q
```

uv throughout, never pip. Python 3.11. Every dependency is free, and the
Gemini key lives in `.env`, which is gitignored.

## What is not in git

Nothing under `data/` or `models/`, no generated report, no project handout —
only the `.gitkeep` files that carry the directory layout. The corpora have to
be put back by hand on a fresh clone before the build scripts will run. The
audio set is the exception: it regenerates itself, because
`pick_audio_sentences.py` is seeded and returns the same 20 sentences and the
same split on any machine.
