# hermes-voice

A voice agent for a medical imaging centre call line: it listens to a caller, answers questions about exams and preparation, books or cancels appointments, and hands over to a human when it should not be the one answering.

School project. Not for clinical use — it never triages, interprets results or gives medical advice.

## Status

Scaffold only. Every module in `src/hermes/` currently holds its docstring and its function signatures; there is no implementation yet, and the test suite is a skipped specification of what each part has to do.

## How a turn works

```
caller audio
  → audio_quality   SNR check — too noisy? ask them to repeat, stop here
  → stt             transcribe (faster-whisper)
  → escalation      emergency wording? read the safety script, transfer, stop
  → intent_gate     route by embedding similarity, or refuse as out of scope
  → faq_rag         retrieve the centre's own FAQ text
    exam_bot        match what they called the exam to the catalogue
  → frustration     score how patient the caller still sounds
  → agent           ONE LLM call, JSON out: reply + action + slots
  → escalation      re-check the rules against the decision
  → agenda          carry out the booking in SQLite
  → tts             speak the reply (piper)
  → logging_utils   log the turn, anonymised
agent audio
```

One LLM call per turn is the design constraint: it keeps the line responsive and the cost predictable. Everything else is a cheap local model or a lookup.

## Setup

Python **3.11** is required (`torch`, `faster-whisper` and `piper-tts` wheels). Dependencies are managed with [uv](https://docs.astral.sh/uv/) — not pip.

```powershell
# 1. virtual environment, in .venv (uv downloads 3.11 if it has to)
uv venv --python 3.11

# 2. dependencies, and the package itself so `import hermes` works anywhere
uv sync --extra dev

# 3. configuration
Copy-Item .env.example .env
#    then put your free Gemini API key in GEMINI_API_KEY
```

The commands are the same on macOS and Linux. `uv run <cmd>` uses `.venv` without activating it; if you prefer a shell in it, `.\.venv\Scripts\Activate.ps1` (Windows) or `source .venv/bin/activate`.

To work on one part only, skip the heavy wheels and install what you need: `uv pip install pytest`.

For the offline fallback, install [Ollama](https://ollama.com) and pull the model named in `.env`:

```powershell
ollama pull mistral
```

## Data

Input CSVs go in `data/raw/` by hand — they are not generated. The build scripts turn them into the embeddings, gazetteer and database in `data/processed/`:

```powershell
uv run python scripts/build_intent_centroids.py   # intent centroids
uv run python scripts/build_faq_index.py          # embedded FAQ index
uv run python scripts/build_exam_gazetteer.py     # exam surface forms → exam ids
uv run python scripts/build_agenda.py --reset     # data/processed/agenda.db
```

## How to run

```powershell
# the demo UI
uv run python app/gradio_app.py

# the tests
uv run pytest
```

## Layout

| Path | What is in it |
| --- | --- |
| `src/hermes/` | the package: one module per stage of a turn |
| `scripts/` | `build_*.py`, raw CSVs → `data/processed/` |
| `app/gradio_app.py` | the demo UI |
| `data/raw/` | input CSVs, copied in by hand |
| `data/audio/` | test clips (wav files gitignored) |
| `data/processed/` | embeddings, caches, `agenda.db` (gitignored) |
| `notebooks/` | training and evaluation |
| `models/` | trained weights (gitignored) |
| `reports/figures/` | charts for the report |
| `tests/` | pytest suite |

## Privacy

Nothing identifiable reaches disk. Transcripts are scrubbed of names, phone numbers, dates of birth, social security numbers, emails and addresses before being logged, callers are identified by a salted hash, and the audio clips are gitignored.
