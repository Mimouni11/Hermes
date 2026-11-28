# The French test audio set

45 clips used to measure speech-to-text and to calibrate the quality gate. The
plan asks for ≥ 20 patient turns, some recorded, some synthesised, some noisy.

Nothing here is committed except the scripts. The clips and the manifest are
gitignored, and they regenerate: the selection is seeded, so the same 20
sentences and the same split come back on any machine.

## What it contains

| source | clips | what they are |
|---|---|---|
| `tts` | 8 | Piper, French voice `fr_FR-siwis-medium` |
| `recorded` | 12 | a real voice reading the sentences |
| `noisy` | 25 | 5 of the clips mixed with babble at SNR 20, 10, 5, 0 and −5 dB |

All 16 kHz mono PCM_16 — what Whisper consumes.

## The manifest

`data/audio/manifest.csv` is the ground truth every later script reads.

```
clip_id, text, lang, intent, exam_id, source, noise_snr_db, file
```

`text` is the sentence as it appears in `data/raw/intents.csv`, and is what WER
is scored against.

## The four scripts

### 1. `pick_audio_sentences.py`

Chooses 20 French sentences from the 100 in the intent corpus, with a fixed
seed, and writes the manifest. The selection is constrained, not merely random:

- **all six intents appear**, and the eight TTS clips cover all six between
  them, so the choice of synthesis or recording never hides an intent;
- **at least three sentences carry a hard medical word** — "densitométrie",
  "mammographie", "échographie" are where a French ASR model actually fails, so
  they are in the set by construction rather than by luck;
- the fill is **balanced**, always drawing from the thinnest intent. A purely
  random fill gave 7 clips of one intent and 1 of another, which is a bad
  evaluation set. The result is 4/4/3/3/3/3.

It then prints the 12 sentences for a human to record, with the filename for
each.

### 2. `make_tts_clips.py`

Synthesises the 8 TTS rows. Piper first — local, offline, and the closest to a
French speaker — falling back to gTTS, which sends the sentence to a Google
endpoint and therefore needs the network.

Piper needs a voice model on disk. Put `*.onnx` and its `*.onnx.json` in
`models/piper/` and it is found automatically; free French voices are at
`huggingface.co/rhasspy/piper-voices` under `fr/fr_FR/`.

### 3. `convert_recordings.py`

Turns whatever a phone records into the WAV the manifest names — 13 source
extensions, ffmpeg when it is on PATH, librosa otherwise. libsndfile has no AAC
support, so ffmpeg is effectively required for `.m4a`.

`-map_metadata -1` drops the recorder's tags on the way: a phone writes the
device model and a creation timestamp into an `.m4a`, and neither belongs in the
project. **Originals are never deleted.**

It reports duration, peak, RMS and a clipping check per clip, where clipping
means a run of three or more full-scale samples — a single sample touching 1.0
is normal and counting it would flag healthy audio.

### 4. `add_noise.py`

Mixes 5 clips at each SNR, spreading the choice across intents. Uses whatever
`data/audio/noise/*.wav` holds, or generates babble from a fixed seed and saves
it so later runs mix identical noise: 8 band-limited voices, 300–3400 Hz, with
syllable-rate envelopes. Not a substitute for a real waiting room, but its
energy sits in the same band as the speech it masks, which is what makes an ASR
model struggle.

When a mix would exceed full scale the whole thing is scaled back. **The SNR is
unchanged by that** — speech and noise scale together — and the script prints
the measured SNR alongside the target so you can see it rather than take its
word. All 25 land exactly on target.

The script is idempotent: it rebuilds the noisy rows rather than appending, so
running it twice does not grow the manifest.

## Rebuilding from nothing

```powershell
uv run python scripts/audio/pick_audio_sentences.py   # manifest + the recording list
uv run python scripts/audio/make_tts_clips.py         # the 8 synthetic clips
#   ... record the 12 sentences into data/audio/recorded/ ...
uv run python scripts/audio/convert_recordings.py     # whatever you recorded -> WAV
uv run python scripts/audio/add_noise.py --snr 20 10 5 0 -5
```

The noisy clips are drawn from whatever has audio on disk at the time, so run
`add_noise.py` **after** the recordings are converted — otherwise the noisy set
is synthetic speech only, which tests Piper rather than a caller.
