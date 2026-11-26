# WP4 exam LLM fallback

Model `gemini-3.5-flash-lite`. Shortlist capped at 5, fallback band [65, 82).

Two runs, because there are two questions and one run cannot answer both: the quality run removes every obstacle to getting an answer, and the availability run keeps them.

## Quality — can it pick the right exam when it answers?

Throttled 2 s between calls, up to 3 attempts with 4/12 s backoff, 30 s budget, cached.

Only the 44 cases the fallback can act on: the gazetteer resolved no exam **and** the top score sits inside the band. Rows the gazetteer already answers cannot be improved by asking, and a missing-side or close-pair clarification scores above the band and is answered with a question by design, never by asking the model.

| | count |
|---|---:|
| fallback cases asked about | 44 |
| of which carry a labelled exam | 42 |
| model replied | 37 |
| recovery rate among replies | 86% |
| **recovered the right exam** | **32** |
| **gave a wrong exam** | **0** |
| named an exam where a question was expected | 0 |

| Set | Asked | Recovered | Wrong |
|---|---:|---:|---:|
| T1 | 25 | 19 | 0 |
| T3 | 8 | 5 | 0 |
| T4 | 11 | 8 | 0 |

### Why the model declined

| Outcome | count |
|---|---:|
| `llm` | 34 |
| `skipped_unresolved_top` | 7 |
| `cache` | 3 |

## Availability — does it answer inside the production budget?

18 cases, one attempt each, cache cleared before each call, budget 3.5 s (`LLM_TIMEOUT_EXAM_S`). No retries.

| | value |
|---|---:|
| cases sent to the model | 15 |
| answered inside budget | 13 |
| timed out | 2 |
| timeout rate | 13% |
| p50 turn latency, all attempts | 1.79 s |
| p90 turn latency, all attempts | 3.52 s |
| p50 of the replies that landed | 1.62 s |
| p90 of the replies that landed | 3.01 s |

Two pairs of percentiles, because they answer different things. The all-attempts pair is what a caller waits — a timed-out call contributes the budget, so once the timeout rate passes half, the p50 *is* the budget and says nothing about the model. The replies pair is the model's real latency, and comparing it with the budget is what shows whether the budget is set anywhere useful.

## Reading the two together

The quality run says whether the shortlist-plus-model idea works. The availability run says how often a caller gets it. They must not be multiplied into a single accuracy figure, because the gazetteer answer is what ships when the model is late — and that number is in `exam_eval.md`.
