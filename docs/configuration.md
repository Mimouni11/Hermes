# Configuration

Everything tunable lives in `src/hermes/config.py`. No other module calls
`os.getenv`, and none builds a filesystem path by hand.

## How it resolves

`load_settings()` reads `.env` through python-dotenv with `override=False`, so
a variable already set in the real environment beats the file. Every key has a
default in `DEFAULTS`, so the project runs with no `.env` at all — which is what
the test suite relies on.

`project_root()` walks up from `config.py` looking for `pyproject.toml` rather
than counting parent directories, so the package still finds its root when
installed in editable mode.

```python
from hermes.config import get_settings

settings = get_settings()      # cached for the process
settings.min_snr_db            # 11.2
settings.paths.agenda_db       # <root>/data/processed/agenda.db
```

For tests, `set_settings()`, `reset_settings()` and `with_overrides(**changes)`
replace the cache without touching the environment.

## Paths

`build_paths()` returns a frozen `Paths` and creates the writable directories
(`processed`, `audio`, `models`, `reports`, `figures`) as a side effect, so no
script has to guard against a missing folder.

## The keys

### Models

| Key | Default | Why |
|---|---|---|
| `STT_BACKEND` | `faster-whisper` | CTranslate2; the openai-whisper path exists as a cross-check |
| `STT_MODEL_SIZE` | `small` | Beats `base` 0.173 vs 0.310 WER head to head — see [stt.md](stt.md) |
| `STT_DEVICE` | `cpu` | Pinned, not probed, so measured latency stays reproducible. `auto` opts into CUDA |
| `STT_COMPUTE_TYPE` | `int8` | What makes `small` realtime on a laptop CPU |
| `TTS_BACKEND` | `piper` | Runs offline; gTTS is the fallback |
| `LANGUAGE` | `fr` | French first; Darija is an extension |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Free tier |
| `OLLAMA_MODEL` | `mistral` | The offline comparison point |

### The audio front end

| Key | Default | Where the number came from |
|---|---|---|
| `TARGET_RMS_DBFS` | `-20.0` | Conventional working level, leaves headroom for the limiter |
| `PEAK_CEILING` | `0.99` | The limiter's ceiling; normalising never clips |
| `MIN_SNR_DB` | `11.2` | **Calibrated.** Rejects 13 of 45 clips at mean WER 0.511, keeps 32 at 0.152 |
| `MIN_CONFIDENCE` | `0.62` | **Calibrated.** Rejects 12 at mean WER 0.523, keeps 33 at 0.159 |
| `MIN_SPEECH_RATIO` | `0.10` | Below this a clip is mostly silence |
| `MAX_CLIPPING_RATIO` | `0.01` | Above this the recording is distorted |
| `MIN_DURATION_S` | `0.30` | Shorter than this there is nothing to transcribe |
| `MAX_SNR_DB` | `60.0` | A clean clip's noise floor is near zero, so the ratio is capped |

The two calibrated values come from `scripts/calibrate_gate.py` run against
measured WER, not from judgement. Re-run it whenever the audio set or the model
changes — the numbers are only valid for the thing they were measured on.

### Still unused

`MIN_CONFIDENCE` is stored and read but nothing enforces it yet. It gates
*after* transcription, so the check belongs in `pipeline.py` next to the
repeat-counting logic, which is still a stub.

Most thresholds are now calibrated against a named test set, and `config.py`
carries the run that produced each one beside the value. This table is a map
of where to look, not a second copy of the numbers — the comments in
`config.py` are the record, because a value and its justification drifting
apart is worse than either alone.

| Key | Value | Calibrated on |
|---|---|---|
| `INTENT_ACCEPT_THRESHOLD` | `0.50` | 222 French examples, leave-one-out, vs 28 held-out out-of-scope — [WP3](WP3.md) |
| `INTENT_MIN_PROBABILITY` | `0.55` | cross-validated, the lever that actually works |
| `FAQ_ACCEPT_THRESHOLD` | `0.625` | 31 rows of `faq_test.csv`; re-measure after any `faq.csv` edit |
| `FAQ_FALLBACK_THRESHOLD` | `0.72` | for **zero** wrong answers on the no-model path |
| `EXAM_FUZZY_THRESHOLD` | `82` | T1 catalogue holdout + `exam_test_t3.csv` — [WP4](WP4.md) |
| `EXAM_MARGIN` | `8.0` | the gate that catches a bare "scanner" scoring 100 |
| `EXAM_LLM_THRESHOLD` | `65` | lower bound of the fallback band `[65, 82)` |
| `LLM_TIMEOUT_FAQ_S` | `2.0` | fallback is the stored answer, so waiting buys only phrasing |
| `LLM_TIMEOUT_EXAM_S` | `3.5` | p50 reply 1.62 s, 13 % timeout rate |
| `INTENT_MARGIN` | `0.08` | **uncalibrated** — superseded by the probability floor |
| `FRUSTRATION_ESCALATE_THRESHOLD` | `0.70` | **uncalibrated** — WP5 |
| `MAX_TURNS_BEFORE_ESCALATION` | `8` | a policy choice, not a measurement |

The exam matcher is implemented: `exam_bot.identify()` ranks 132 bases from a
built gazetteer, resolves laterality and contrast as slots, and transfers the
five `escalate` exams after resolution. See [WP4.md](WP4.md).

Treat the two remaining uncalibrated values as placeholders.

## Secrets

`.env` is gitignored and holds `GEMINI_API_KEY` and `LOG_SALT`.
`require_gemini_key()` raises a message telling you what to do rather than
failing deep inside a request. `.env.example` is committed and carries every
key with a safe value.
