"""Dense embedding clients for the IngestionPipeline.

A thin HTTP wrapper around a local `Ollama <https://ollama.com>`_ daemon
running an embedding model (default ``nomic-embed-text``, 768-dim). Embeddings
are serialised as float32 ``bytes`` for storage in the ``embedding`` BLOB
column of ``knowledge_chunks``.

The client degrades gracefully when the daemon is unreachable: ``embed``
returns ``None`` and ``is_available()`` reports ``False`` instead of raising,
so ingestion never fails because a sidecar service is down.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from typing import TYPE_CHECKING, List, Optional
from urllib.parse import urlsplit

import requests

# numpy is imported lazily inside the functions that use it (not at module
# load). The CLI imports this module eagerly via the deep-research command
# chain, so a module-level `import numpy` makes a broken/slow numpy on Windows
# crash every `jarvis` command — including `jarvis serve` (#404, #309).
if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
_MAX_EMBED_BATCH = 32


def _is_loopback_host(host: str) -> bool:
    """Whether an Ollama URL targets a loopback address without credentials."""
    try:
        parsed = urlsplit(host if "://" in host else f"http://{host}")
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
            return False
        if parsed.port == 0:
            return False
        hostname = (parsed.hostname or "").casefold().rstrip(".")
        if hostname == "localhost":
            return True
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


class OllamaEmbedder:
    """Embed text via a local Ollama daemon.

    Parameters
    ----------
    model:
        Ollama model tag (e.g. ``nomic-embed-text``, ``mxbai-embed-large``).
    host:
        Base URL for the Ollama HTTP API. When ``None`` (default) the
        ``OLLAMA_HOST`` environment variable is used, then
        ``http://localhost:11434`` -- the same precedence as ``OllamaEngine``.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_EMBED_MODEL,
        host: Optional[str] = None,
        timeout: float = 30.0,
        allow_redirects: bool = True,
        trust_env: bool | None = None,
    ) -> None:
        self._model = model
        if host is None:
            host = os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._allow_redirects = allow_redirects
        self._trust_env = (
            not _is_loopback_host(self._host) if trust_env is None else bool(trust_env)
        )
        self._session = requests.Session()
        self._session.trust_env = self._trust_env
        self._dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Identity / capability checks
    # ------------------------------------------------------------------

    @property
    def model_version(self) -> str:
        """Stable identifier persisted alongside each embedding row."""
        return f"ollama:{self._model}"

    @property
    def dim(self) -> Optional[int]:
        """Embedding dimensionality, learned after the first successful call."""
        return self._dim

    def is_available(self) -> bool:
        """Return True iff the daemon answers and the model is installed."""
        try:
            resp = self._session.get(
                f"{self._host}/api/tags",
                timeout=2.0,
                allow_redirects=self._allow_redirects,
            )
            if resp.is_redirect:
                return False
            resp.raise_for_status()
        except requests.RequestException:
            return False

        try:
            names = {m.get("name", "") for m in resp.json().get("models", [])}
        except ValueError:
            return False

        # Ollama tags include the ":latest" suffix; match either form.
        return self._model in names or f"{self._model}:latest" in names

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, text: str) -> Optional[bytes]:
        """Embed a single string. Returns float32 bytes or ``None`` on failure."""
        if not text or not text.strip():
            return None
        try:
            resp = self._session.post(
                f"{self._host}/api/embeddings",
                json={"model": self._model, "prompt": text},
                timeout=self._timeout,
                allow_redirects=self._allow_redirects,
            )
            if resp.is_redirect:
                return None
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            logger.warning("OllamaEmbedder.embed: request failed (%s)", exc)
            return None
        except ValueError as exc:
            logger.warning("OllamaEmbedder.embed: bad JSON (%s)", exc)
            return None

        vec = payload.get("embedding")
        if not vec:
            logger.warning(
                "OllamaEmbedder.embed: empty embedding for %d chars", len(text)
            )
            return None

        return self._vector_bytes(vec)

    def _vector_bytes(self, vector: object) -> Optional[bytes]:
        """Validate an Ollama vector and serialize it as float32 bytes."""
        import numpy as np

        try:
            arr = np.asarray(vector, dtype=np.float32)
        except (TypeError, ValueError):
            return None
        if arr.ndim != 1 or not arr.size:
            return None
        if self._dim is None:
            self._dim = int(arr.shape[0])
        elif arr.shape[0] != self._dim:
            logger.warning(
                "OllamaEmbedder: dim drift (expected %d, got %d)",
                self._dim,
                arr.shape[0],
            )
            return None
        return arr.tobytes()

    def embed_batch(self, texts: List[str]) -> List[Optional[bytes]]:
        """Embed text batches, falling back to the legacy single-text API.

        Current Ollama versions accept an ``input`` list at ``/api/embed``.
        Chunking bounds request size. Older versions that do not expose that
        route use ``/api/embeddings`` one item at a time.
        """
        results: List[Optional[bytes]] = [None] * len(texts)
        for offset in range(0, len(texts), _MAX_EMBED_BATCH):
            batch = [
                (index, text)
                for index, text in enumerate(
                    texts[offset : offset + _MAX_EMBED_BATCH], start=offset
                )
                if text and text.strip()
            ]
            if not batch:
                continue
            try:
                resp = self._session.post(
                    f"{self._host}/api/embed",
                    json={"model": self._model, "input": [text for _, text in batch]},
                    timeout=self._timeout,
                    allow_redirects=self._allow_redirects,
                )
                if resp.is_redirect:
                    continue
                resp.raise_for_status()
                payload = resp.json()
            except requests.HTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in {404, 405, 501}:
                    self._embed_batch_legacy(batch, results)
                else:
                    logger.warning(
                        "OllamaEmbedder.embed_batch: request failed (%s)", exc
                    )
                continue
            except requests.RequestException as exc:
                logger.warning("OllamaEmbedder.embed_batch: request failed (%s)", exc)
                continue
            except ValueError as exc:
                logger.warning("OllamaEmbedder.embed_batch: bad JSON (%s)", exc)
                self._embed_batch_legacy(batch, results)
                continue

            vectors = payload.get("embeddings") if isinstance(payload, dict) else None
            if not isinstance(vectors, list) or len(vectors) != len(batch):
                self._embed_batch_legacy(batch, results)
                continue
            for (index, _text), vector in zip(batch, vectors):
                results[index] = self._vector_bytes(vector)
        return results

    def _embed_batch_legacy(
        self,
        batch: List[tuple[int, str]],
        results: List[Optional[bytes]],
    ) -> None:
        """Use the original endpoint when the server lacks ``/api/embed``."""
        for index, text in batch:
            results[index] = self.embed(text)


def default_embedder(
    *, local_only: bool = False, host: Optional[str] = None
) -> Optional[OllamaEmbedder]:
    """Return an ``OllamaEmbedder`` if the daemon and default model are up.

    Ingestion call sites use this so chunks get embedded whenever an embedder
    is actually reachable, and fall back to lexical-only rows (``None``)
    otherwise. Query-side hybrid search does the same probe, so the two ends
    stay in step: if this returns ``None`` at ingest, ``_vector_recall`` finds
    no rows to score either way. Local-only callers require a loopback host,
    disable proxy environment variables, and reject redirects.
    """
    host = host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
    if local_only and not _is_loopback_host(host):
        return None
    embedder = OllamaEmbedder(
        host=host,
        allow_redirects=not local_only,
        trust_env=False if local_only else None,
    )
    return embedder if embedder.is_available() else None


# ---------------------------------------------------------------------------
# Deserialisation helper (used by verification + future retrieval code)
# ---------------------------------------------------------------------------


def decode_embedding(blob: Optional[bytes], *, dtype=None) -> Optional[np.ndarray]:
    """Reconstruct a 1-D vector from a BLOB written by ``OllamaEmbedder.embed``.

    Returns ``None`` when the input is missing or zero-length so callers can
    treat absent embeddings uniformly. ``dtype`` defaults to ``np.float32``
    (resolved lazily; passing ``np.float32`` as a default arg would import
    numpy at module load).
    """
    if not blob:
        return None
    import numpy as np

    return np.frombuffer(blob, dtype=dtype if dtype is not None else np.float32)


__all__ = [
    "OllamaEmbedder",
    "decode_embedding",
    "default_embedder",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_OLLAMA_HOST",
]
