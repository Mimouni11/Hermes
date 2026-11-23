# The agent's voice

`src/hermes/tts.py` turns a reply into speech. Piper by default — local,
offline, no per-call cost — with gTTS behind it as a fallback and as the
comparison point the report needs.

This completes WP2. The audio chain now runs end to end: a clip is gated,
transcribed, and a reply is spoken back.

## Why a file

There is no live telephony — SIP/Twilio is cut to future work — so a call is
simulated one turn at a time in Gradio. Three things need the file rather than
just playable sound:

- **the cache**, since the agent repeats its prompts and the fixed safety
  script on nearly every call;
- **the listening test**, where five people have to rate the same artifacts;
- **evidence**, so the report and the demo video carry the audio the agent
  actually produced.

It is not either/or. `synthesise()` writes a file and returns `Speech`;
`synthesise_samples()` returns the array and skips the disk, which is the path
a live microphone turn should take.

## `prepare_text` is the part that matters

Synthesis is a solved problem; reading a booking aloud is not. A reply carries
a reference code and an appointment time, and a voice reads neither the way a
person would. Everything below is applied before the text reaches Piper:

| written | spoken |
|---|---|
| `K7M2PQ` | `K, 7, M, 2, P, Q` |
| `14h30`, `14:30`, `14 h 30` | `quatorze heures trente` |
| `9h` | `neuf heures` |
| `1h00` | `une heure` (singular — not `un heures`) |
| `2026-11-26` | `vingt-six novembre` |
| `IRM`, `TDM`, `ORL` | `I.R.M.`, `T.D.M.`, `O.R.L.` |
| `rdv` | `rendez-vous` |
| `Dr` | `docteur` |

**Order is load-bearing.** Reference codes are expanded first, before the
acronym pass can rewrite their letters — a code like `TDM4KP` starts with an
acronym and would otherwise come out as `T.D.M.4KP`. Times are expanded before
the bare `h` abbreviation would turn `14h30` into `14 heures 30` with the
digits still unspoken.

The code pattern matches the agenda's alphabet exactly: six characters with no
`I`, `O`, `0` or `1`, so an ordinary six-letter word is never mistaken for a
code.

Dates carry **no article**. `le` is not prepended because the surrounding
sentence already has one, and prepending produces "Le le vingt-six novembre".

## The cache

Keyed on `sha256(spoken text, backend, voice)`, truncated, prefixed with the
backend so a glance at the directory says which engine made each line. Change
the text, the backend or the voice and the line is synthesised again — which is
what makes the cache safe to leave in place while the voice is still being
chosen.

Cached lines go to `data/processed/tts_cache/`. `clear_cache()` empties it and
reports how many it removed. Passing an explicit `out_path` bypasses the cache
entirely.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `TTS_BACKEND` | `piper` | `piper` or `gtts` |
| `TTS_VOICE` | *(unset)* | A `.onnx` path; unset means the first in `models/piper` |
| `TTS_SAMPLE_RATE` | `22050` | Higher than the STT side's 16 kHz — nothing transcribes these, and the listening test does |

An unknown backend raises rather than silently falling back, so a typo in
`.env` fails loudly.

## Measured

Synthesis on CPU, Piper `fr_FR-siwis-medium`:

| reply | spoken length | time to synthesise |
|---|---|---|
| booking confirmation | 9.91 s | 0.99 s |
| two slots offered | 7.12 s | 0.31 s |
| ask to repeat | 3.90 s | 0.14 s |
| fasting instruction | 4.31 s | 0.13 s |

About 0.1× realtime, and every repeated line is free after the first.

### The latency budget is tight

The brief targets under 5 seconds per turn, audio to audio. Measured so far:

| stage | time |
|---|---|
| quality gate | negligible |
| STT (`small`, CPU int8) | 2.46 s mean, 3.32 s worst |
| LLM call | not yet measured |
| TTS | 0.1–1.0 s, 0 s when cached |

That leaves roughly 1–2 seconds for the LLM call before the target is missed.
Measure the full chain as soon as the agent exists rather than discovering it
on demo day.

## Open: does it sound right?

Everything above is verified as *text*. Whether Piper actually says `I.R.M.` as
three letters, and whether `K, 7, M, 2, P, Q` is paced slowly enough for a
caller to write down, can only be settled by listening. Four sample replies are
written to `data/audio/tts_demo/` for exactly that.

If the acronym spelling does not work, the fix is in `ACRONYMS` — a different
written form, not a code change. The listening test in the evaluation plan
(1–5, ≥ 5 listeners, mean ≥ 3.5) is first on the brief's drop list, so it is
optional, but a single listen to the confirmation line is not.

## Tests

25 tests in `tests/test_tts.py`. Most need no voice and no network: the number
speller, the time and date expansion, the reference-code rule, the abbreviation
pass and the cache key. The four synthesis tests skip themselves when no Piper
voice is installed, so the suite still passes on a fresh clone.
