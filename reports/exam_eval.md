# WP4 exam identification — the gazetteer

**No model is called in this run.** Every number below is the gazetteer and the decision layer on their own, which is what the demo falls back to whenever the model is slow or unavailable.

The LLM fallback is measured separately, in `exam_llm_eval.md`: mixing the two put a rate-limited API inside the accuracy table and made the gazetteer look worse than it is.

Score floor 82, margin 8, fallback band floor 65.

| Set | Mode | Correct IDs / labelled | Wrong IDs / labelled | Clarified / expected | Premature IDs | Clarify rate |
|---|---|---:|---:|---:|---:|---:|
| T1 | Gazetteer | 231/841 | 137/841 | — | 0 | 473/841 |
| T2 | Gazetteer | 20/22 | 0/22 | — | 0 | 2/22 |
| T3 | Gazetteer | 23/35 | 0/35 | 5/7 | 2 | 17/42 |
| T4 | Gazetteer | 37/57 | 0/57 | 5/7 | 2 | 25/64 |
| T3 without keyword overlap | Gazetteer | 18/30 | 0/30 | — | 0 | 12/30 |
| T4 changed | Gazetteer | 4/14 | 0/14 | — | 0 | 10/14 |

## By modality

T2, T3 and T4 together, keyed on the expected exam.

| Modality | Labelled | Correct | Wrong | Asked |
|---|---:|---:|---:|---:|
| IRM | 28 | 23 | 0 | 5 |
| échographie | 28 | 12 | 0 | 16 |
| radiographie | 24 | 18 | 0 | 6 |
| scanner | 16 | 15 | 0 | 1 |
| mammographie | 8 | 2 | 0 | 6 |
| dentaire | 6 | 6 | 0 | 0 |
| densitométrie | 4 | 4 | 0 | 0 |

## Reading the results

- 42 T3/T4 rows end in a question. The LLM fallback can only act on the subset of those whose top score also sits in the band, counted exactly in `exam_llm_eval.md`. No other row in this table can change when the model is enabled.
- T1 removes the normalized and repaired query from all catalogue bases before ranking, and **excludes the samples whose base that would empty**. For a lateral pair every French keyword collapses to one phrase, so holding it out deletes the base and the exam becomes unfindable by construction.
- That exclusion moved the T1 denominator from 905 to 841 and the clarify count from 537 to 473. It left the correct and wrong counts untouched: every excluded sample had been scored as withheld, not as a wrong resolution. So the margin sweep that chose 8 was reading an inflated clarify rate, not an inflated error rate, and re-running it on the corrected sample selected the same thresholds.
- T2 contains source-corpus wording. T3 uses hand-written requests. The overlap audit compares contiguous keyword sequences after the project's WER normalization.
- T4 is a **text replay**, using one longest observed substitution per matching request and retaining unchanged requests as controls. The rules come from accepted small/raw transcripts in `reports/stt_results.csv`. Adjacent edits are kept together. Standalone insertions and deletions, and clips rejected by the audio gate, are outside this replay. Its edit frequency does not estimate caller STT error rates.
- Changed T4 inputs (14/64): `du coude → de côte` × 1, `quand → qu on` × 2, `rachis → acquis` × 1, `rénales doppler → aénales dopplères` × 1, `veux → vais` × 1, `écho → écouter` × 8.
- Observed substitution rules available to the replay: 13.
