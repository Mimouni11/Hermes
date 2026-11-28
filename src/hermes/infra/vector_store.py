"""Qdrant: the collections behind the intent gate and the FAQ retrieval.

One thin layer over `qdrant-client` so the rest of the package never imports it
directly. Two collections, both 384-dimensional and both cosine:

- ``hermes_intents`` — one point per labelled intent example
- ``hermes_faq`` — one point per *question variant*, so a Darija question
  matches a Darija vector rather than a French one

The server runs in Docker; see `docker-compose.yml`. Nothing here holds state
beyond a cached client, so the collections can be dropped and rebuilt from the
corpora at any time.

Why a vector store at all, at 245 intent sentences and 24 FAQ entries: a numpy
cosine would be faster today. The corpora are expected to grow, the Darija side
is not written yet, and moving the retrieval layer later would land at the
point when there is least time. Recorded in docs/WP3.md as a deliberate trade.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from hermes.config import get_settings

#: Namespace for turning a corpus id ("S170", "horaires") into a stable point
#: id. Qdrant takes an unsigned integer or a UUID, and the corpora use neither,
#: so each key is hashed to the same UUID every run — which makes re-indexing
#: an overwrite rather than a duplicate.
_NAMESPACE = uuid.UUID("6f1d4e8a-6c2a-4a7a-9f2c-3d5b8e9a1c04")

_CLIENTS: dict[tuple[str, int], Any] = {}


@dataclass(frozen=True)
class Hit:
    """One retrieved neighbour."""

    id: str
    score: float
    payload: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        """Read a payload field."""
        return self.payload.get(key, default)


def point_id(key: str) -> str:
    """Return the stable point id for a corpus key."""
    return str(uuid.uuid5(_NAMESPACE, key))


# --------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------

def client(host: str | None = None, port: int | None = None) -> Any:
    """Return a cached Qdrant client."""
    settings = get_settings()
    where = (host or settings.qdrant_host, port or settings.qdrant_port)

    if where not in _CLIENTS:
        from qdrant_client import QdrantClient

        _CLIENTS[where] = QdrantClient(
            host=where[0],
            port=where[1],
            api_key=settings.qdrant_api_key,
            timeout=30,
            # The client/server version handshake warns on every call when it
            # cannot reach the server, which buries the real error. The pinned
            # compose file is what keeps the versions in step.
            check_compatibility=False,
        )
    return _CLIENTS[where]


#: Remembers whether the server answered, so a suite full of skips does not
#: pay the connection timeout once per test.
_REACHABLE: bool | None = None

#: Seconds to wait when only asking "is it there". Short on purpose: the
#: answer when Qdrant is down should arrive immediately, not after a retry
#: budget meant for real queries.
PING_TIMEOUT = 2.0


def ping(force: bool = False) -> bool:
    """Whether the server answers. Used to fail with advice, not a stack trace.

    The result is cached for the process. ``force=True`` re-checks, which is
    what to use after starting the container without restarting Python.
    """
    global _REACHABLE
    if _REACHABLE is not None and not force:
        return _REACHABLE

    settings = get_settings()
    try:
        from qdrant_client import QdrantClient

        probe = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            timeout=PING_TIMEOUT,
            check_compatibility=False,
        )
        probe.get_collections()
        _REACHABLE = True
    except Exception:  # noqa: BLE001 — any failure means "not usable"
        _REACHABLE = False
    return _REACHABLE


def require_server() -> None:
    """Raise with the command to run when Qdrant is not up."""
    if not ping(force=True):
        settings = get_settings()
        raise RuntimeError(
            f"Qdrant is not answering on {settings.qdrant_host}:"
            f"{settings.qdrant_port}. Start it with:\n"
            "    docker compose up -d"
        )


# --------------------------------------------------------------------------
# Collections
# --------------------------------------------------------------------------

def collection_exists(name: str) -> bool:
    """Whether a collection is already there."""
    return bool(client().collection_exists(collection_name=name))


def ensure_collection(
    name: str,
    dim: int | None = None,
    recreate: bool = False,
) -> None:
    """Create the collection if it is missing, optionally dropping it first.

    ``recreate=True`` is what the index builders use: re-indexing a corpus
    should not leave points behind from rows that have since been deleted.
    """
    from qdrant_client.models import Distance, VectorParams

    settings = get_settings()
    size = dim or settings.embedding_dim
    handle = client()

    if recreate and collection_exists(name):
        handle.delete_collection(collection_name=name)

    if not collection_exists(name):
        handle.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )


def drop_collection(name: str) -> bool:
    """Delete a collection. Returns whether there was one to delete."""
    if not collection_exists(name):
        return False
    client().delete_collection(collection_name=name)
    return True


def count(name: str) -> int:
    """How many points a collection holds."""
    if not collection_exists(name):
        return 0
    return int(client().count(collection_name=name, exact=True).count)


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def upsert(
    name: str,
    vectors: np.ndarray,
    payloads: Sequence[dict[str, Any]],
    keys: Sequence[str] | None = None,
    batch_size: int = 256,
) -> int:
    """Write points, returning how many were sent.

    ``keys`` are corpus ids; each is hashed to a stable point id, so running an
    index builder twice overwrites rather than duplicates. Without them the ids
    are positional, which is only safe for a collection built in one pass.
    """
    from qdrant_client.models import PointStruct

    if len(vectors) != len(payloads):
        raise ValueError(
            f"{len(vectors)} vectors against {len(payloads)} payloads"
        )
    if keys is not None and len(keys) != len(payloads):
        raise ValueError(
            f"{len(keys)} keys against {len(payloads)} payloads"
        )
    if not len(vectors):
        return 0

    points = [
        PointStruct(
            id=point_id(keys[i]) if keys is not None else point_id(f"{name}:{i}"),
            vector=[float(x) for x in vectors[i]],
            payload=dict(payloads[i]),
        )
        for i in range(len(payloads))
    ]

    handle = client()
    for start in range(0, len(points), batch_size):
        handle.upsert(
            collection_name=name,
            points=points[start : start + batch_size],
            wait=True,
        )
    return len(points)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def _match_filter(conditions: dict[str, Any] | None) -> Any:
    """Build an equality filter over payload fields, or None."""
    if not conditions:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    return Filter(
        must=[
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in conditions.items()
        ]
    )


def search(
    name: str,
    vector: np.ndarray,
    limit: int = 5,
    where: dict[str, Any] | None = None,
    score_threshold: float | None = None,
) -> list[Hit]:
    """Return the nearest points, best first.

    ``where`` filters on payload equality — ``{"lang": "fr"}`` to search one
    language. ``score_threshold`` is applied by the server, but the gate
    applies its own threshold on the results instead: knowing the best score
    was 0.41 is the difference between "out of scope" and "no idea", and a
    server-side cut throws that away.
    """
    response = client().query_points(
        collection_name=name,
        query=[float(x) for x in np.asarray(vector).ravel()],
        limit=limit,
        query_filter=_match_filter(where),
        score_threshold=score_threshold,
        with_payload=True,
    )

    return [
        Hit(
            id=str(point.id),
            score=float(point.score),
            payload=dict(point.payload or {}),
        )
        for point in response.points
    ]


def search_text(
    name: str,
    text: str,
    limit: int = 5,
    where: dict[str, Any] | None = None,
) -> list[Hit]:
    """Embed a sentence and search in one call."""
    from hermes.infra.embeddings import encode_one

    return search(name, encode_one(text), limit=limit, where=where)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def describe() -> dict[str, int]:
    """Return the point count of every collection, for the build scripts."""
    handle = client()
    return {
        collection.name: count(collection.name)
        for collection in handle.get_collections().collections
    }


def iter_points(name: str, batch_size: int = 256) -> Iterable[Hit]:
    """Walk every point in a collection, for evaluation and debugging."""
    offset = None
    while True:
        points, offset = client().scroll(
            collection_name=name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            yield Hit(id=str(point.id), score=0.0, payload=dict(point.payload or {}))
        if offset is None:
            return
