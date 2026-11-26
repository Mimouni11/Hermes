# WP4 exam identification evaluation

Score floor 82, margin 8, fallback floor 65, budget 2 s per model request.
The configured model is `gemini-3.5-flash-lite`. Client warm-up succeeded: True.
The run started with an empty isolated cache, shared within this evaluation.

| Set | Mode | Correct IDs / labelled | Wrong IDs / labelled | Clarified / expected | Premature IDs | Clarify rate |
|---|---|---:|---:|---:|---:|---:|
| T1 | Gazetteer | 231/905 | 137/905 | — | 0 | 537/905 |
| T1 | LLM enabled | 236/905 | 137/905 | — | 0 | 532/905 |
| T2 | Gazetteer | 20/22 | 0/22 | — | 0 | 2/22 |
| T2 | LLM enabled | 20/22 | 0/22 | — | 0 | 2/22 |
| T3 | Gazetteer | 23/35 | 0/35 | 5/7 | 2 | 17/42 |
| T3 | LLM enabled | 24/35 | 0/35 | 5/7 | 2 | 16/42 |
| T4 | Gazetteer | 37/57 | 0/57 | 5/7 | 2 | 25/64 |
| T4 | LLM enabled | 38/57 | 0/57 | 5/7 | 2 | 24/64 |
| T3 without keyword overlap | Gazetteer | 18/30 | 0/30 | — | 0 | 12/30 |
| T3 without keyword overlap | LLM enabled | 19/30 | 0/30 | — | 0 | 11/30 |
| T4 changed | Gazetteer | 4/14 | 0/14 | — | 0 | 10/14 |
| T4 changed | LLM enabled | 5/14 | 0/14 | — | 0 | 9/14 |

## Fallback usage

| Set | Invoked / rows | Cache hits | Live replies | Timeouts / uncached | Other outcomes | Median uncached turn time |
|---|---:|---:|---:|---:|---:|---:|
| T1 | 292/905 | 0 | 9 | 29/292 | 254 | 0.085 s |
| T2 | 0/22 | 0 | 0 | — | 0 | — |
| T3 | 6/42 | 0 | 1 | 5/6 | 0 | 2.022 s |
| T4 | 9/64 | 0 | 2 | 7/9 | 0 | 2.024 s |

## Reading the results

- T1 removes the normalized and repaired query from all catalogue bases before ranking.
- T2 contains source-corpus wording. T3 uses hand-written requests. The overlap audit compares contiguous keyword sequences after the project's WER normalization.
- T4 is a **text replay**, using one longest observed substitution per matching request and retaining unchanged requests as controls. The rules come from accepted small/raw transcripts in `reports/stt_results.csv`. Adjacent edits are kept together. Standalone insertions and deletions, and clips rejected by the audio gate, are outside this replay. Its edit frequency does not estimate caller STT error rates.
- Changed T4 inputs (14/64): `du coude → de côte` × 1, `quand → qu on` × 2, `rachis → acquis` × 1, `rénales doppler → aénales dopplères` × 1, `veux → vais` × 1, `écho → écouter` × 8.
- The median measures matcher wall time for uncached fallback invocations, including timed-out calls. A live reply can contain a rejection or abstention.
- Observed substitution rules available to the replay: 13.
