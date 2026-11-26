# WP6 — Agent level 2, booking, and dialogue state

**Status:** draft, not started. This page exists to hold the things WP2, WP3
and WP4 deliberately deferred, so they are not rediscovered late.

> **Raised in WP4, owned by WP7:** the exam selection call sends the caller's
> repaired utterance to Gemini, as the FAQ answer call has since WP3. The model
> cannot choose among five candidates without the words, so this is a question
> about what may leave the building on a live call rather than a prompt to
> redact. It belongs with WP7's anonymised logging, next to the decision about
> what a transcript keeps.

**Done when:** a caller books, moves or cancels an appointment end to end, in
one LLM call per turn, with the exam re-checked against the gazetteer before
anything is written to the agenda.

## Scope

One LLM call per turn, returning JSON — the plan's form is `exam`,
`preferred_day`, `preferred_time`, `choice`, `confirmation`. The model never
touches the database: Python searches, holds and books, and every rule is
enforced in `agenda.py`, which is already built and tested.

## Carried forward from WP3

### Dialogue state

The plan lists this under WP3. It was deferred because a short reply is only
meaningful as the answer to a question the agent asked, and nothing asked one
until the booking branch existed.

**The concrete consequence:** `intent_gate.is_too_short()` rejects "oui" and
"le deuxième" before anything is embedded, which is right at the start of a
call and wrong in the middle of a booking. Measured: "oui", "hmm", "euh..."
and "allô ?" all retrieve "Opérateur !" at about 0.78, higher than almost any
genuine in-scope example, so they cannot be routed by similarity at all.

WP6 needs to give `classify()` a way to know a question is outstanding and
read the reply in that context rather than routing it from scratch.

### Counting the repeats

The plan says "ask to rephrase once, then human". Right now every rephrase is
a first rephrase, because nothing counts them. The same counter serves the
audio gate, which has the same rule.

### `min_confidence` is not enforced

`MIN_CONFIDENCE = 0.62` is calibrated and read, and nothing acts on it. It
gates *after* transcription, so the check belongs here, next to the repeat
counter.

### The FAQ answer call is the agent's info path, built early

`faq_rag.spoken_answer()` already makes one LLM call per `info` turn: it hands
the top-3 retrieved passages to the model and gets back one spoken sentence.

**This is not a second LLM call, and it does not break the one-call-per-turn
rule.** Info turns never reach the agent today, so no turn makes two calls.
What it is, is the agent's info path implemented ahead of the agent.

When the agent lands, the two merge into **one prompt and one JSON schema**,
not two calls stacked. The agent receives the same top-3 passages as grounding
and returns its usual object — `reply`, `answerable`, `used_faq_ids`,
`safety_mentions` — with the reply coming from the passages rather than from
the model's own knowledge.

Two properties must survive the merge:

- **The relevance check.** The model refusing when the passages do not fit is
  what turns a confidently wrong answer into a safe one, and no retrieval
  threshold can replace it (see below). `answerable: false` is how the merged
  schema carries it.
- **The verbatim fallback on timeout.** 33% of calls timed out at a 2-second
  budget on the free tier. When the call fails, the stored FAQ answer is
  spoken unchanged. The merged path needs the same escape hatch, because the
  centre's own text is always a correct thing to say.

Do not refactor this before the agent exists. Merging into a schema that has
not been designed would mean designing it twice.

### `safety_mentions` on the agent's JSON

The lexicon raises safety flags — a word in the transcript that is probably a
misheard implant, allergy or pregnancy term. Today a flag transfers
immediately. When the agent exists, its JSON should carry a `safety_mentions`
field so the model can confirm aloud ("vous avez mentionné des broches,
c'est-à-dire un implant métallique ?") before the handover, rather than the
handover arriving unexplained.

The post-LLM escalation check still runs on the result, as CLAUDE.md requires.
The field informs the conversation; it is not permitted to end it.

## Future work carried from the lexicon layer

### Context-aware repair thresholds

Domain repair currently uses one fixed similarity bar plus a French-word
guard. Safety flagging already uses context — a loose match is only believed
when the sentence is about imaging — because the measurement forced it:

| pair | ratio | |
|---|---|---|
| `brosses` ~ `broches` | 0.71 | must flag |
| `brioche` ~ `broches` | 0.86 | must not |
| `brioches` ~ `broches` | 0.93 | must not |

No single threshold separates those. Domain repair has not been given the same
treatment, and probably should: "erem aki lombaire" is "IRM rachis lombaire"
and nothing in the current design can know it, because the character
similarity is near zero and only the surrounding words carry the meaning.

### A second lexicon pass after the agent

The repair runs once, before routing. A second pass could use what the agent
has established — the exam already confirmed, the body region already
discussed — to resolve manglings the first pass had no context for.

## What already exists and should be reused

| piece | state |
|---|---|
| `agenda.py` | **done** — slots, multi-slot exams, holds with owners, 59 tests |
| `intent_gate.py` | **done** — routing, scope, three medical layers |
| `faq_rag.py` | **done** — retrieval with a calibrated threshold |
| `lexicon.py` | **done** — repair and safety flags between STT and routing |
| `escalation.py` | **partial** — the medical rules are built and measured; `check()`, the scripts and the transfer routing are not |
| `agent.py` | stub |
| `pipeline.py` | stub — the orchestration all of this plugs into |

The bench in `app/gradio_app.py` wires the stages by hand precisely because
`pipeline.run_turn` does not exist. When it does, the bench should call it and
keep its panels.

## Risks

**The latency budget is already tight.** Measured on a warm process: quality
gate negligible, STT 2.4 s, lexicon repair negligible, TTS 0.1–1.0 s and free
when cached. That leaves roughly 1.5 s for the LLM call against the brief's
5-second target. Gemini free-tier latency is not yet measured, and the plan's
own risk table flags rate limits.

**The agent must never see the whole FAQ or the whole catalogue.** The plan is
explicit: the model sees candidate exams only, and the JSON is re-checked
against the gazetteer before booking. That constraint is what keeps an
invented exam out of the database.
