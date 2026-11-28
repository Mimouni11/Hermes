"""Hermes voice agent for a medical imaging centre.

:mod:`hermes.pipeline` orchestrates each turn, from caller audio to spoken reply.

Turn stages:
- :mod:`hermes.listen.audio_quality` checks incoming audio.
- :mod:`hermes.listen.stt` transcribes speech.
- :mod:`hermes.listen.lexicon` repairs the transcript and flags safety terms.
- :mod:`hermes.guard.escalation` checks for safety and transfer before and after the model.
- :mod:`hermes.understand.intent_gate` routes the request.
- :mod:`hermes.understand.faq_rag` retrieves the centre's answers.
- :mod:`hermes.understand.exam_bot` resolves the exam against the gazetteer.
- :mod:`hermes.understand.frustration` scores caller frustration.
- :mod:`hermes.agent` builds and parses the turn's structured model call.
- :mod:`hermes.book.agenda` manages the SQLite appointment calendar.
- :mod:`hermes.speak.tts` synthesizes the reply.
- :mod:`hermes.infra.logging_utils` records anonymised turns.

Shared modules:
- :mod:`hermes.config` owns configuration, paths and thresholds.
- :mod:`hermes.understand.mlp` defines the frustration model.
- :mod:`hermes.infra.llm` handles provider calls and caching.
- :mod:`hermes.infra.embeddings` encodes text.
- :mod:`hermes.infra.vector_store` manages the search index.

CLAUDE.md Rule 2 limits the agent to imaging-centre information and appointments,
with immediate transfer for suspected emergencies. Rule 4 keeps settings in
config and limits each turn to one model call.

School project. Not for clinical use.
"""

__version__: str = "0.1.0"
