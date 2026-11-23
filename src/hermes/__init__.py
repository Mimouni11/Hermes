"""hermes-voice: a voice agent for a medical imaging centre call line.

Handles one caller turn at a time: audio in, audio out. The package is split so
each stage of the turn lives in its own module:

- :mod:`hermes.config`         environment, paths and tunable thresholds
- :mod:`hermes.audio_quality`  signal-to-noise check on the incoming clip
- :mod:`hermes.stt`            speech to text
- :mod:`hermes.tts`            text to speech
- :mod:`hermes.intent_gate`    embedding intent routing and out-of-scope rejection
- :mod:`hermes.faq_rag`        FAQ retrieval
- :mod:`hermes.exam_bot`       exam name gazetteer and fuzzy matching
- :mod:`hermes.frustration`    frustration classifier
- :mod:`hermes.agent`          one LLM call per turn, JSON output
- :mod:`hermes.llm`            Gemini and Ollama clients with a response cache
- :mod:`hermes.agenda`         SQLite appointment calendar
- :mod:`hermes.escalation`     human transfer rules and the safety script
- :mod:`hermes.logging_utils`  anonymised call logs
- :mod:`hermes.pipeline`       orchestration of a single turn

School project. Not for clinical use.
"""

__version__: str = "0.1.0"
