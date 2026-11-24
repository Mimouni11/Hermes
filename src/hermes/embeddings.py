"""Sentence embeddings — one model, every language the call line carries.

The corpora are tagged `fr`, `tn` (Darija in Arabic script), `tn_latn`
(Arabizi) and `mix` (French-Darija code-switching). A multilingual model is not
a convenience here: a Darija question has to land near its Darija neighbours in
the same space the French ones live in, or routing a Darija caller is guesswork.

The model is loaded once per process. The first load downloads the weights,
which is the only step here that touches the network.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from hermes.config import get_settings

#: Loaded models, keyed by name. SentenceTransformer is expensive to construct
#: and thread-safe to reuse, so one per process is enough.
_MODELS: dict[str, Any] = {}

#: Encoding batch size. Large enough to keep the corpora fast, small enough not
#: to spike memory on a laptop.
BATCH_SIZE = 64


def load_model(name: str | None = None) -> Any:
    """Load and cache the sentence embedding model."""
    settings = get_settings()
    model_name = name or settings.embedding_model

    if model_name not in _MODELS:
        from sentence_transformers import SentenceTransformer

        _MODELS[model_name] = SentenceTransformer(model_name)
    return _MODELS[model_name]


def dimension(name: str | None = None) -> int:
    """Return how many dimensions the model emits.

    sentence-transformers renamed this accessor; the old name still works but
    warns, so the new one is preferred and the old kept as the fallback.
    """
    model = load_model(name)
    getter = getattr(model, "get_embedding_dimension", None)
    if getter is None:
        getter = model.get_sentence_embedding_dimension
    return int(getter())


def check_dimension(name: str | None = None) -> int:
    """Verify the model matches ``EMBEDDING_DIM``, and return the true size.

    A mismatch between the configured dimension and the model's own is not an
    error anything else would catch: the collection would be created at the
    wrong width and every later upsert would fail, or worse, succeed against a
    stale collection and retrieve nonsense.
    """
    settings = get_settings()
    actual = dimension(name)
    if actual != settings.embedding_dim:
        raise ValueError(
            f"{name or settings.embedding_model} emits {actual} dimensions, "
            f"but EMBEDDING_DIM is {settings.embedding_dim}. Fix the config "
            "and rebuild the collections — they are created at that width."
        )
    return actual


def encode(
    texts: str | Sequence[str],
    name: str | None = None,
    normalise: bool = True,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Embed one or many sentences, returning a ``(n, dim)`` float32 array.

    Normalised to unit length by default, so a dot product *is* the cosine
    similarity. That keeps the scores comparable between what Qdrant returns
    and anything computed locally, which matters when calibrating a threshold
    against one and applying it to the other.
    """
    items = [texts] if isinstance(texts, str) else list(texts)
    if not items:
        settings = get_settings()
        return np.zeros((0, settings.embedding_dim), dtype="float32")

    model = load_model(name)
    vectors = model.encode(
        items,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalise,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")


def encode_one(
    text: str,
    name: str | None = None,
    normalise: bool = True,
) -> np.ndarray:
    """Embed a single sentence, returning a flat ``(dim,)`` array."""
    return encode(text, name=name, normalise=normalise)[0]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, safe on a zero vector."""
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0:
        return 0.0
    return float(np.dot(a, b) / denominator)
