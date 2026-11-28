# Speech to text and the quality gate

Two modules and two scripts: `audio_quality.py` decides whether a clip is worth
transcribing, `stt.py` transcribes it, `eval_stt.py` measures both, and
`calibrate_gate.py` turns those measurements into thresholds.

## The quality gate

`audio_quality.py` runs before transcription, so a bad turn costs nothing but a
request to repeat.

### `normalize(audio)`

Scales RMS to `target_rms_dbfs` (−20), then limits the peak to `peak_ceiling`
(0.99). When the gain needed to hit the target would push the loudest sample
past full scale, the clip is scaled back and **the RMS ends below target**.
That trade is deliberate: a consistent level is worth less than an undistorted
one, and clipping is not recoverable.

### `estimate_snr(audio)`

Energy-based and reference-free — at run time there is only the caller's audio,
with no clean version to compare against. The clip is cut into 25 ms frames;
the 90th percentile of frame energy is taken as speech, the 10th as the noise
floor, and the estimate is the ratio, capped at 60 dB.

**It reads higher than the SNR a mixer dialled in**, because it compares frame
levels rather than signal against a known reference:

| mixed at | estimated |
|---|---|
| 20 dB | 20.5–23.5 |
| 10 dB | 13.2–16.0 |
| 5 dB | 9.4–11.2 |
| 0 dB | 6.1–7.5 |
| −5 dB | 3.2–4.6 |

What matters is that it orders clips correctly, which it does. Do not read an
estimate as the true SNR.

### `check(audio) -> "ok" | "repeat"`

Reasons are tested most-specific first: `too_short`, `silent`, `clipping`,
`low_snr`, `no_speech`. The ordering is load-bearing — `speech_ratio` needs
about 6 dB of headroom over the noise floor to find speech at all, so a clip
buried in noise looks *empty* to it. Checking SNR first means a noisy caller is
told "there is a lot of noise around you" rather than "I cannot hear you".

Each reason has its own French retry line. The pipeline counts the verdicts:
the first `repeat` asks the caller to say it again, the second hands over.

## The Whisper wrapper

`STT` wraps faster-whisper. `small` on CPU `int8`, read from config rather than
probed, so a measurement does not depend on which machine made it;
`STT_DEVICE=auto` opts back into CUDA. The model loads lazily, and `eval_stt.py`
forces the load before timing so the download never lands inside a latency
figure.

`transcribe(audio)` returns `text`, `confidence`, `no_speech_prob`,
`duration_s`, `latency_s`.

- **confidence** is `exp(avg_logprob)` weighted by segment duration, so a
  half-second "oui" does not count as much as a ten-second sentence.
- **no_speech_prob** takes the **maximum** across segments, not the mean: one
  segment that is confidently silence is the interesting signal, and averaging
  it away is how a half-empty clip passes.

### The initial prompt

Whisper conditions on `initial_prompt` as if it were the text just transcribed.
`build_initial_prompt()` mines the exam catalogue for words of seven characters
or more, longest first, merges them with a hand-written core list, and phrases
the result as a sentence — prose conditions the decoder better than a word list.

**It is not working well enough.** `IRM` is the first term in the prompt and
still came back as `erem` on a clean recording. A likely cause is dilution: 60
French medical terms probably overrun Whisper's 224-token prompt window.
Untested — see *What to try next*.

## Measured results

`scripts/audio/eval_stt.py` runs every clip through every model on raw and normalised
audio, and writes `reports/stt_results.csv`.

### Model comparison — 35 clips, both models

| source | small/raw | small/norm | base/raw | base/norm |
|---|---|---|---|---|
| tts | 0.041 | 0.041 | 0.135 | 0.135 |
| recorded | 0.180 | 0.186 | 0.238 | 0.242 |
| noisy 20 dB | 0.209 | 0.209 | 0.338 | 0.338 |
| noisy 10 dB | 0.209 | 0.209 | 0.547 | 0.484 |
| noisy 5 dB | 0.298 | 0.253 | 0.498 | 0.575 |
| **all** | **0.173** | **0.169** | **0.310** | **0.314** |

| model | mean latency | xRT |
|---|---|---|
| small | 2.46 s | 0.64 |
| base | 0.78 s | 0.20 |

**`small` wins and ships.** It nearly halves the WER, and at 0.64 xRT it is
still realtime, so `base`'s speed advantage buys nothing the 5-second-per-turn
target needs. `base` collapses on exactly the vocabulary this call line runs on:

```
ref:  vous avez de la place pour irm rachis lombaire samedi
base: vous avez de la place pour irrer un hockey un bœur saint gierre
```

### Normalisation does not earn its place

0.173 → 0.169 for `small` is within noise, and `base` got marginally worse.
Whisper normalises internally. It is kept for level consistency, not for WER.

A useful side-effect confirms the implementation: the raw and normalised SNR
estimates are **identical** on all 35 clips, which is correct — SNR is a ratio,
so a uniform gain cancels. `normalize()` is pure gain with no distortion.

### Where it falls over — 45 clips, `small` only

Adding the 0 and −5 dB tiers found the cliff:

| source | mean WER | mean estimated SNR |
|---|---|---|
| tts | 0.041 | 55.8 |
| recorded | 0.180 | 25.2 |
| noisy 20 dB | 0.229 | 22.0 |
| noisy 10 dB | 0.209 | 14.9 |
| noisy 5 dB | 0.320 | 10.6 |
| noisy 0 dB | 0.426 | 6.8 |
| **noisy −5 dB** | **0.622** | 4.0 |

(0.256 overall, not comparable with the 0.173 above — a harder set, and `small`
ran alone.)

### Error analysis

The failures are domain vocabulary, not spelling variants. Across all 140 rows
of the first run there were **zero** accent-only differences and no
`rdv`/number/`I.R.M.` variants — so the WER normaliser needs no extension.

| reference | heard as | count |
|---|---|---|
| `jeun` | `jeu` | 13 |
| `échographie` | `écographie` | 6 |
| `irm rachis` | `erm acquis` / `rm aki` / `rmhk` | 3 |
| `à l iode` | `alleraute` | 2 |
| `rénales doppler` | `aénales dopplères` | 2 |

This is the risk the plan names — "STT mangles medical words" — and it is real.

## The WER normaliser

`normalise_for_wer()` lowercases, removes punctuation, and turns apostrophes
into spaces so `l'échographie` and `l échographie` score alike — Whisper is
inconsistent about elision and scoring it would measure typography.

**Accents are kept.** `echographie` is a different word from `échographie`;
folding them would hide exactly the errors in the table above.

## Calibrating the gate

`scripts/audio/calibrate_gate.py` bins WER against each measurement and finds the
least aggressive cut that still isolates a group at mean WER ≥ 0.5 — half the
words wrong, so any slot filled from it is a guess.

| threshold | value | rejects | their mean WER | kept mean WER |
|---|---|---|---|---|
| `min_snr_db` | **11.2** | 13 of 45 | 0.511 | 0.152 |
| `min_confidence` | **0.62** | 12 of 45 | 0.523 | 0.159 |
| `max_no_speech_prob` | 0.235 | 8 of 45 | 0.510 | 0.201 |

The first two are in `config.py`. **`no_speech_prob` was measured and
rejected** as a gate: its kept group sits at 0.201 against 0.152, and it throws
out clips with perfectly good SNR. It is a noisier signal than the other two.

The two thresholds sit at different points: `min_snr_db` gates before
transcription, `min_confidence` after, since confidence only exists once
Whisper has decoded. The second catches what the first cannot — a clean
recording of a mumble.

`min_confidence` is **not enforced yet**; the check belongs in `pipeline.py`.

## What to try next

1. **Fix the vocabulary biasing.** Count the prompt's tokens, then compare a
   10-term prompt against the current 60-term one, and both against
   faster-whisper's `hotwords` parameter, which biases decoding more directly
   than `initial_prompt`.
2. **Fuzzy-match the transcript** against the gazetteer rather than relying on
   Whisper alone — the plan's own mitigation, and `erem`/`rm aki` are close
   enough to `IRM` for RapidFuzz to recover.
3. **Confirm the exam aloud** before booking, which catches what both miss.
