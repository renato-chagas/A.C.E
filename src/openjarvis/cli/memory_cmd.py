"""``jarvis memory`` — memory management subcommands."""

from __future__ import annotations

import re
import sqlite3
import time
import uuid
from collections.abc import Sequence
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from rich.text import Text

from openjarvis.core.config import load_config
from openjarvis.core.paths import ConfigurationError, get_config_dir
from openjarvis.core.registry import MemoryRegistry
from openjarvis.memory.procedures import ProcedureProposal, skill_markdown
from openjarvis.memory.skill_versions import (
    list_skill_versions,
    load_skill_version,
    next_patch_version,
    record_skill_version,
    set_skill_version,
    skill_name_from_markdown,
    skill_version_from_markdown,
    validate_skill_name,
)
from openjarvis.security.file_utils import secure_write_text
from openjarvis.tools.storage.chunking import ChunkConfig
from openjarvis.tools.storage.ingest import ingest_path

if TYPE_CHECKING:
    from openjarvis.memory.episodes import ArchiveClearIntent, EpisodeStore


def _get_backend(backend_key: str | None = None):
    """Instantiate the configured (or overridden) memory backend."""
    config = load_config()
    key = backend_key or config.memory.default_backend

    # Ensure backends are registered
    import openjarvis.tools.storage  # noqa: F401

    if not MemoryRegistry.contains(key):
        raise click.ClickException(
            f"Memory backend '{key}' not found. "
            f"Available: {', '.join(MemoryRegistry.keys())}"
        )

    if key == "sqlite":
        return MemoryRegistry.create(key, db_path=config.memory.db_path)
    return MemoryRegistry.create(key)


@click.group()
def memory() -> None:
    """Manage the memory store."""


@memory.command(name="backup")
@click.argument("destination", type=click.Path(file_okay=False, path_type=Path))
def backup_cmd(destination: Path) -> None:
    """Back up local memory to a new folder."""
    from openjarvis.memory.backup import backup_memory

    try:
        count = backup_memory(load_config().memory, destination)
    except (ConfigurationError, OSError, sqlite3.Error, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Backed up {count} local memory store(s) to {destination}.")


@memory.command(name="restore")
@click.argument("source", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--into",
    "destination",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="New OpenJarvis home outside the source tree.",
)
def restore_cmd(source: Path, destination: Path) -> None:
    """Verify a memory backup and restore it to a new profile."""
    from openjarvis.memory.backup import restore_memory

    try:
        count = restore_memory(source, destination)
    except (ConfigurationError, OSError, sqlite3.Error, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Restored {count} local memory store(s) to {destination}.")
    click.echo("Set OPENJARVIS_HOME to that directory to use the restored profile.")


@memory.command()
@click.argument("path")
@click.option(
    "--backend",
    "-b",
    default=None,
    help="Override the default memory backend.",
)
@click.option(
    "--chunk-size",
    default=512,
    type=click.IntRange(min=1),
    help="Chunk size in tokens.",
)
@click.option(
    "--ocr",
    is_flag=True,
    help="Use local Tesseract OCR for images and scanned PDF pages.",
)
@click.option(
    "--transcribe-media",
    is_flag=True,
    help="Transcribe local audio tracks in supported audio/video files.",
)
@click.option(
    "--chunk-overlap",
    default=64,
    type=click.IntRange(min=0),
    help="Overlap between chunks in tokens.",
)
def index(
    path: str,
    backend: str | None,
    chunk_size: int,
    chunk_overlap: int,
    ocr: bool,
    transcribe_media: bool,
) -> None:
    """Index documents from a file or directory."""
    console = Console(stderr=True)
    target = Path(path)

    if not target.exists():
        console.print(f"[red]Path not found:[/red] {path}")
        raise SystemExit(1)

    t0 = time.time()
    cfg = ChunkConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    console.print(f"[cyan]Indexing[/cyan] {path} ...")
    try:
        ingest_options = {"config": cfg, "ocr": ocr}
        if transcribe_media:
            ingest_options.update(
                transcribe_media=True,
                speech_config=load_config().speech,
            )
        chunks = ingest_path(target, **ingest_options)
    except (ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc

    if not chunks:
        console.print("[yellow]No indexable content found.[/yellow]")
        return

    mem = _get_backend(backend)
    try:
        replace_source = getattr(mem, "replace_source", None)
        if callable(replace_source):
            documents_by_source = {}
            for chunk in chunks:
                documents_by_source.setdefault(chunk.source, []).append(
                    (
                        chunk.content,
                        {
                            "offset": chunk.offset,
                            "index": chunk.index,
                        },
                    )
                )
            for source, documents in track(
                documents_by_source.items(),
                description="Replacing sources...",
                console=console,
            ):
                replace_source(source, documents)
        else:
            for chunk in track(
                chunks,
                description="Storing chunks...",
                console=console,
            ):
                mem.store(
                    chunk.content,
                    source=chunk.source,
                    metadata={
                        "offset": chunk.offset,
                        "index": chunk.index,
                    },
                )
    finally:
        if hasattr(mem, "close"):
            mem.close()

    elapsed = time.time() - t0
    sources = {c.source for c in chunks}
    console.print(
        f"[green]Indexed {len(chunks)} chunks "
        f"from {len(sources)} file(s) "
        f"in {elapsed:.1f}s.[/green]"
    )


@memory.command()
@click.argument("query", nargs=-1, required=True)
@click.option(
    "--top-k",
    "-k",
    default=5,
    type=int,
    help="Number of results to return.",
)
@click.option(
    "--backend",
    "-b",
    default=None,
    help="Override the default memory backend.",
)
def search(
    query: tuple[str, ...],
    top_k: int,
    backend: str | None,
) -> None:
    """Search the memory store."""
    console = Console()
    query_text = " ".join(query)

    mem = _get_backend(backend)
    try:
        results = mem.retrieve(query_text, top_k=top_k)
    finally:
        if hasattr(mem, "close"):
            mem.close()

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f"Search: {query_text}")
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", width=8)
    table.add_column("Source", style="cyan")
    table.add_column("Content")

    for i, r in enumerate(results, 1):
        # Truncate content for display
        preview = r.content[:200]
        if len(r.content) > 200:
            preview += "..."
        table.add_row(
            str(i),
            f"{r.score:.4f}",
            r.source or "-",
            preview,
        )

    console.print(table)


def _get_fact_store():
    """Instantiate the automatic-memory fact store from config."""
    from openjarvis.memory.store import create_fact_store

    config = load_config()
    mem = config.memory
    return create_fact_store(
        getattr(mem, "backend", "local"),
        path=getattr(mem, "facts_path", "~/.openjarvis/memory_facts.jsonl"),
        max_facts=getattr(mem, "max_facts", 1000),
    )


def _prune_fact_embeddings(facts=(), *, clear: bool = False) -> None:
    """Keep the optional semantic sidecar aligned with durable fact changes."""
    try:
        config = load_config().memory
        from openjarvis.memory.fact_embeddings import (
            FactEmbeddingIndex,
            fact_embedding_index_path,
        )

        index_path = fact_embedding_index_path(config.facts_path)
        if not index_path.is_file():
            return
        with FactEmbeddingIndex(
            index_path,
            embedding_model=config.facts_embedding_model,
        ) as index:
            if clear:
                index.clear()
            else:
                index.prune_to(facts)
    except Exception:
        # A vector sidecar is an optional index, never a reason to prevent
        # deletion or trust updates in the authoritative JSONL fact store.
        return


@memory.group(name="facts")
def facts_group() -> None:
    """Review and manage durable memory facts."""


@facts_group.command(name="revise")
@click.argument("existing_fact")
@click.argument("correction")
def revise_fact(existing_fact: str, correction: str) -> None:
    """Replace the exact wording of a fact and keep its previous version."""
    from openjarvis.memory.service import _build_scanner, _build_secret_scanner

    correction = correction.strip()
    if not correction or len(correction) > 1_000 or not correction.isprintable():
        raise click.ClickException("Enter a single line of up to 1,000 characters.")
    for scanner_factory in (_build_secret_scanner, _build_scanner):
        scanner = scanner_factory()
        try:
            clean = scanner is not None and scanner.scan(correction).is_clean is True
        except Exception:
            clean = False
        if not clean:
            raise click.ClickException(
                "The local scanners could not clear this correction for storage."
            )
    store = _get_fact_store()
    try:
        store.revise(existing_fact, correction)
    except (ValueError, NotImplementedError) as exc:
        raise click.ClickException(str(exc)) from exc
    _prune_fact_embeddings(store.list())
    Console().print(
        "[green]Memory revised. The previous wording is kept as history.[/green]"
    )


@facts_group.command(name="reindex")
@click.option(
    "--limit",
    type=click.IntRange(1, 1_000),
    default=100,
    show_default=True,
    help="Maximum number of facts to embed in this run.",
)
def reindex_facts(limit: int) -> None:
    """Create loopback-only semantic vectors for durable memory facts."""
    from openjarvis.memory.fact_embeddings import (
        FactEmbeddingIndex,
        fact_embedding_index_path,
    )

    config = load_config().memory
    facts = _get_fact_store().list()
    try:
        with FactEmbeddingIndex(
            fact_embedding_index_path(config.facts_path),
            embedding_model=config.facts_embedding_model,
        ) as index:
            indexed, remaining = index.index_missing_embeddings(facts, limit=limit)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    Console().print(
        f"[green]Indexed {indexed} fact(s); {remaining} still need vectors.[/green]"
    )


def _get_episode_store() -> EpisodeStore:
    """Open the local episode database using current retention settings."""
    from openjarvis.memory.episodes import EpisodeStore

    memory_config = load_config().memory
    return EpisodeStore(
        memory_config.episodes_path,
        max_episodes=memory_config.episodes_max_count,
        retention_days=memory_config.episodes_retention_days,
        semantic_enabled=memory_config.episodes_semantic_enabled,
        embedding_model=memory_config.episodes_embedding_model,
    )


def _purge_episode_graph_data(
    episode_ids: Sequence[str],
    *,
    conversation_sources: Sequence[str] | None = None,
) -> dict[str, int]:
    """Scrub conversation-derived graph data before deleting its source text."""
    graph_path = Path(load_config().memory.graph_db_path).expanduser()
    if not graph_path.is_file():
        return {
            "entities_removed": 0,
            "entities_updated": 0,
            "relations_removed": 0,
            "relations_updated": 0,
            "evidence_removed": 0,
        }
    from openjarvis.tools.storage.knowledge_graph import KnowledgeGraphMemory

    with KnowledgeGraphMemory(db_path=graph_path) as graph:
        return graph.purge_episode_data(
            episode_ids, conversation_sources=conversation_sources
        )


def _purge_episode_fact_data(episode_ids: Sequence[str]) -> dict[str, int]:
    """Remove facts supported only by deleted episodes and prune their vectors."""
    store = _get_fact_store()
    counts = store.purge_episode_sources(episode_ids)
    if counts.get("unsupported", 0):
        raise RuntimeError(
            "the configured fact store does not support episode-aware cleanup"
        )
    _prune_fact_embeddings(store.list())
    return counts


@memory.group(name="episodes")
def episodes_group() -> None:
    """Inspect and remove retained conversation episodes."""


@episodes_group.command(name="list")
@click.option("--limit", type=click.IntRange(1, 500), default=30)
def list_episodes(limit: int) -> None:
    """List recent substantial conversations retained by ACE."""
    console = Console()
    store = _get_episode_store()
    try:
        episodes = store.list(limit=limit)
    finally:
        store.close()
    if not episodes:
        console.print("[yellow]No conversation episodes stored.[/yellow]")
        return
    table = Table(title=f"Conversation Episodes ({len(episodes)})")
    table.add_column("ID", style="dim")
    table.add_column("Date", style="cyan")
    table.add_column("Source")
    table.add_column("User message")
    for episode in episodes:
        date_text = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(episode.created_at))
        preview = episode.user_text[:180]
        if len(episode.user_text) > 180:
            preview += "…"
        table.add_row(
            episode.episode_id,
            date_text,
            escape(episode.source or "-"),
            escape(preview),
        )
    console.print(table)


@episodes_group.command(name="show")
@click.argument("episode_id")
def show_episode(episode_id: str) -> None:
    """Display one complete retained conversation by its full ID."""
    if not re.fullmatch(r"[0-9a-fA-F]{32}", episode_id.strip()):
        raise click.ClickException("Episode ID must be a full 32-character ID.")
    store = _get_episode_store()
    try:
        episode = store.get_by_id(episode_id)
    finally:
        store.close()
    if episode is None:
        raise click.ClickException(f"Episode not found: {episode_id.strip()}")

    console = Console()
    date_text = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(episode.created_at))
    details = Text()
    details.append("ID: ", style="bold")
    details.append(episode.episode_id)
    details.append("\nDate: ", style="bold")
    details.append(date_text)
    details.append("\nSource: ", style="bold")
    details.append(episode.source or "-")
    console.print(Panel(details, title="Conversation episode", expand=False))
    console.print(Panel(Text(episode.user_text), title="User", expand=False))
    console.print(Panel(Text(episode.assistant_text), title="A.C.E.", expand=False))


@episodes_group.command(name="search")
@click.argument("query", nargs=-1, required=True)
@click.option("--top-k", type=click.IntRange(1, 20), default=5)
def search_episodes(query: tuple[str, ...], top_k: int) -> None:
    """Search retained episodes lexically and, when enabled, semantically."""
    console = Console()
    query_text = " ".join(query)
    store = _get_episode_store()
    try:
        results = store.search(query_text, top_k=top_k)
    finally:
        store.close()
    if not results:
        console.print("[yellow]No matching episodes found.[/yellow]")
        return
    for index, result in enumerate(results, 1):
        episode_id = str(result.metadata.get("episode_id", ""))
        console.print(
            f"[bold cyan]Episode {index}[/bold cyan] "
            f"[dim](ID: {escape(episode_id)})[/dim]"
        )
        console.print(escape(result.content))
        console.print()


@episodes_group.command(name="reindex")
@click.option(
    "--limit",
    type=click.IntRange(1, 1_000),
    default=100,
    show_default=True,
    help="Maximum number of episodes to embed in this run.",
)
def reindex_episodes(limit: int) -> None:
    """Create local semantic vectors for older conversation episodes."""
    store = _get_episode_store()
    try:
        indexed, considered = store.reindex_missing_embeddings(limit=limit)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        store.close()
    Console().print(f"[green]Indexed {indexed} of {considered} episode(s).[/green]")


@episodes_group.command(name="derive")
@click.option(
    "--limit",
    type=click.IntRange(1, 50),
    default=5,
    show_default=True,
    help="Maximum number of queued extraction tasks to process.",
)
@click.option("--model", "model_name", help="Model to use for this run.")
@click.option("--engine", "engine_key", help="Local engine to use for this run.")
def derive_episodes(limit: int, model_name: str | None, engine_key: str | None) -> None:
    """Derive memory from queued episodes using local inference."""
    from openjarvis.engine._discovery import get_configured_engine
    from openjarvis.engine._locality import is_local_engine_for_model
    from openjarvis.memory.service import drain_ask_derivations

    try:
        config = deepcopy(load_config())
    except Exception:
        raise click.ClickException("Could not load memory configuration.") from None

    model = next(
        (
            candidate.strip()
            for candidate in (
                model_name,
                config.memory.extraction_model,
                config.intelligence.default_model,
            )
            if candidate and candidate.strip()
        ),
        "",
    )
    if not model:
        raise click.ClickException(
            "Choose a local model with --model or configure "
            "[memory].extraction_model or [intelligence].default_model."
        )

    config.intelligence.provider = "local"
    config.intelligence.fallback_model = ""
    config.engine.auto_start_local = False
    config.memory.extraction_model = model
    try:
        resolved = get_configured_engine(config, engine_key=engine_key, model=model)
    except Exception:
        raise click.ClickException(
            "Could not resolve a local inference engine."
        ) from None
    if resolved is None:
        raise click.ClickException(
            "No local engine is available for the selected model."
        )

    key, engine = resolved
    try:
        if not is_local_engine_for_model(engine, model, key):
            raise click.ClickException(
                "The selected model requires a verified local route."
            )
        try:
            completed, deferred = drain_ask_derivations(
                config, engine, model, limit=limit
            )
        except Exception:
            raise click.ClickException(
                "Could not process queued memory derivations."
            ) from None
    finally:
        with suppress(Exception):
            engine.close()
    click.echo(f"Completed {completed} task(s). Deferred {deferred} task(s).")


def _complete_episode_forget(
    store, episode_id: str
) -> tuple[dict[str, int], dict[str, int]]:
    """Idempotently remove one episode and all memory derived from it."""
    fact_counts = _purge_episode_fact_data([episode_id])
    graph_counts = _purge_episode_graph_data([episode_id])
    if store.get_by_id(episode_id) is not None and not store.delete(episode_id):
        raise RuntimeError(
            "Related memory cleanup completed, but the episode was no longer "
            "present at the final delete step. It may have been removed by "
            "another process; verify it with the episode show command."
        )
    store.finish_deletion_intent(episode_id)
    return fact_counts, graph_counts


def _archive_clear_snapshot(store: EpisodeStore) -> ArchiveClearIntent:
    from openjarvis.memory.episodes import ArchiveClearIntent

    graph_ids: set[str] = set()
    conversation_sources: set[str] = set()
    graph_path = Path(load_config().memory.graph_db_path).expanduser()
    if graph_path.is_file():
        from openjarvis.tools.storage.knowledge_graph import KnowledgeGraphMemory

        with KnowledgeGraphMemory(db_path=graph_path) as graph:
            graph_ids, conversation_sources = graph.episode_cleanup_targets()
    episode_ids = store.archive_episode_ids()
    graph_ids.update(episode_ids)
    return ArchiveClearIntent(
        token=uuid.uuid4().hex,
        episode_ids=episode_ids,
        graph_episode_ids=tuple(sorted(graph_ids)),
        conversation_sources=tuple(sorted(conversation_sources)),
    )


def _complete_episode_clear(
    store: EpisodeStore, intent: ArchiveClearIntent
) -> tuple[dict[str, int], dict[str, int], int]:
    """Remove the episodes and evidence selected when clear was confirmed."""
    fact_counts = _purge_episode_fact_data(intent.episode_ids)
    graph_counts = _purge_episode_graph_data(
        intent.graph_episode_ids, conversation_sources=intent.conversation_sources
    )
    removed = store.finish_archive_clear(intent)
    return fact_counts, graph_counts, removed


@episodes_group.command(name="recover")
def recover_episode_deletions() -> None:
    """Resume episode cleanup interrupted by a storage error or process exit."""
    console = Console()
    store = _get_episode_store()
    try:
        intent = store.pending_archive_clear()
        pending = store.pending_deletion_intents()
        legacy_clear_pending = store.has_legacy_archive_clear()
        if not (intent or pending or legacy_clear_pending):
            console.print("[green]No interrupted episode cleanup is pending.[/green]")
            return
        if intent is not None:
            try:
                _, _, removed = _complete_episode_clear(store, intent)
            except Exception as exc:
                raise click.ClickException(
                    "Could not resume the interrupted archive clear. Its recovery "
                    "intent remains saved; resolve the storage error and run "
                    "jarvis memory episodes recover again: " + str(exc)
                ) from exc
            console.print(
                f"[green]Recovered the interrupted archive clear "
                f"({removed} episode(s) removed).[/green]"
            )

        completed = 0
        failures: list[tuple[str, str]] = []
        for episode_id in pending:
            try:
                _complete_episode_forget(store, episode_id)
                completed += 1
            except Exception as exc:
                failures.append((episode_id, str(exc)))
        if completed:
            console.print(
                f"[green]Recovered cleanup for {completed} episode(s).[/green]"
            )
        if failures:
            details = "; ".join(
                f"{episode_id}: {error}" for episode_id, error in failures
            )
            raise click.ClickException(
                f"Recovered {completed} cleanup(s); {len(failures)} remain pending. "
                "Resolve the storage error and run jarvis memory episodes recover "
                f"again. Details: {details}"
            )
        if legacy_clear_pending:
            raise click.ClickException(
                "An older interrupted archive clear has no saved scope. Run "
                "jarvis memory episodes clear to review and confirm a new cleanup."
            )
    finally:
        store.close()


@episodes_group.command(name="forget")
@click.argument("episode_id")
def forget_episode(episode_id: str) -> None:
    """Delete one episode and any memory supported only by that episode."""
    episode_id = (episode_id or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{32}", episode_id):
        raise click.ClickException("Episode ID must be a full 32-character ID.")
    store = _get_episode_store()
    try:
        pending = store.pending_deletion_intents()
        if store.get_by_id(episode_id) is None and episode_id not in pending:
            raise click.ClickException(f"Episode not found: {episode_id}")
        if episode_id not in pending:
            try:
                store.record_deletion_intent(episode_id)
            except Exception as exc:
                raise click.ClickException(
                    "Could not save a recovery intent; no related memory was "
                    f"changed: {exc}"
                ) from exc
        try:
            fact_counts, graph_counts = _complete_episode_forget(store, episode_id)
        except Exception as exc:
            raise click.ClickException(
                "Could not finish cleanup of all related memory data. The "
                "related memory may already have been removed, including facts "
                "or graph "
                "evidence. The episode may still be present if its delete step "
                "had not "
                "completed. Its recovery intent is saved; resolve the error "
                "and retry jarvis memory episodes recover: " + str(exc)
            ) from exc
    finally:
        store.close()

    Console().print(f"[green]Forgot episode {escape(episode_id)}.[/green]")
    if any(graph_counts.values()):
        Console().print(
            "Removed related graph data: "
            f"{graph_counts['evidence_removed']} evidence item(s), "
            f"{graph_counts['relations_removed']} relation(s), "
            f"{graph_counts['relations_updated']} relation(s) updated, "
            f"{graph_counts['entities_removed']} entity/entities removed, "
            f"{graph_counts['entities_updated']} entity/entities updated."
        )
    if fact_counts["removed_facts"] or fact_counts["removed_links"]:
        Console().print(
            "Removed "
            f"{fact_counts['removed_facts']} fact(s) supported only by this episode; "
            f"cleared {fact_counts['removed_links']} episode source link(s)."
        )
    if fact_counts["retained_facts"]:
        Console().print(
            f"Kept {fact_counts['retained_facts']} fact(s) with another known origin."
        )
    if fact_counts["untracked_remaining"] or fact_counts["unsupported"]:
        Console().print(
            "[yellow]Some facts have no complete episode provenance and may need "
            "manual review with jarvis memory list.[/yellow]"
        )


@episodes_group.command(name="clear")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
def clear_episodes(yes: bool) -> None:
    """Remove episodes and memory supported only by those episodes."""
    console = Console()
    store = _get_episode_store()
    try:
        if store.pending_archive_clear() is not None:
            raise click.ClickException(
                "An archive clear is already pending. "
                "Run jarvis memory episodes recover."
            )
        try:
            intent = _archive_clear_snapshot(store)
        except Exception as exc:
            raise click.ClickException(
                f"Could not read the archive cleanup targets: {exc}"
            ) from exc
        count = len(intent.episode_ids)
        confirmation = (
            f"Remove the {count} selected conversation episode(s), linked facts, "
            "and existing conversation-derived graph evidence?"
            if count
            else "No episodes are stored. Remove existing conversation graph evidence?"
        )
        if not yes and not click.confirm(confirmation):
            console.print("[dim]Aborted.[/dim]")
            return
        try:
            store.record_archive_clear(intent)
        except Exception as exc:
            raise click.ClickException(
                "Could not save a recovery intent; no related memory was changed: "
                + str(exc)
            ) from exc
        try:
            fact_counts, graph_counts, removed = _complete_episode_clear(store, intent)
        except Exception as exc:
            raise click.ClickException(
                "Could not finish cleanup of all conversation-derived memory data. "
                "Some episodes may remain, and related memory may already "
                "have been removed. The recovery intent is saved; resolve the "
                "error and "
                "retry jarvis memory episodes recover: " + str(exc)
            ) from exc
    finally:
        store.close()
    if removed:
        console.print(f"[green]Cleared {removed} conversation episode(s).[/green]")
    else:
        console.print("[yellow]No conversation episodes to clear.[/yellow]")
    if any(graph_counts.values()):
        console.print(
            "Removed conversation-derived graph data: "
            f"{graph_counts['evidence_removed']} evidence item(s), "
            f"{graph_counts['relations_removed']} relation(s), "
            f"{graph_counts['relations_updated']} relation(s) updated, "
            f"{graph_counts['entities_removed']} entity/entities removed, "
            f"{graph_counts['entities_updated']} entity/entities updated."
        )
    if fact_counts["removed_facts"] or fact_counts["removed_links"]:
        console.print(
            "Removed "
            f"{fact_counts['removed_facts']} fact(s) supported only by cleared "
            f"episodes; cleared {fact_counts['removed_links']} episode source link(s)."
        )
    if fact_counts["retained_facts"]:
        console.print(
            f"Kept {fact_counts['retained_facts']} fact(s) with another known origin."
        )
    if fact_counts["untracked_remaining"] or fact_counts["unsupported"]:
        console.print(
            "[yellow]Some facts have no complete episode provenance and may need "
            "manual review with jarvis memory list.[/yellow]"
        )


@memory.group(name="procedures")
def procedures_group() -> None:
    """Review procedure proposals learned from successful interactions."""


@procedures_group.command(name="list")
@click.option(
    "--status",
    type=click.Choice(["proposed", "approved", "rejected", "all"]),
    default="proposed",
    show_default=True,
)
@click.option("--limit", type=click.IntRange(1, 500), default=50)
def list_procedure_proposals(status: str, limit: int) -> None:
    """List procedures waiting for review, or inspect a selected status."""
    console = Console()
    store = _get_episode_store()
    try:
        proposals = store.list_procedure_proposals(status=status, limit=limit)
    finally:
        store.close()
    if not proposals:
        console.print(f"[yellow]No {status} procedure proposals.[/yellow]")
        return
    table = Table(title=f"Procedure Proposals ({status})")
    table.add_column("ID", style="dim")
    table.add_column("Status", style="cyan")
    table.add_column("Title")
    table.add_column("Skill name")
    for proposal in proposals:
        table.add_row(
            proposal.proposal_id,
            proposal.status,
            escape(proposal.title),
            proposal.skill_name,
        )
    console.print(table)


@procedures_group.command(name="outcomes")
@click.option("--limit", type=click.IntRange(1, 500), default=50, show_default=True)
def list_procedure_outcomes(limit: int) -> None:
    """Review explicit failure reports linked to recent procedure episodes."""
    store = _get_episode_store()
    try:
        outcomes = store.list_procedure_outcomes(limit=limit)
    finally:
        store.close()
    if not outcomes:
        Console().print("[yellow]No procedure failure outcomes stored.[/yellow]")
        return
    console = Console()
    console.print(f"[bold]Procedure Outcomes ({len(outcomes)})[/bold]")
    for outcome in outcomes:
        console.print(Text(f"Episode: {outcome.evidence_episode_id}", style="dim"))
        console.print(Text(f"Task: {outcome.task_title}", style="bold"))
        console.print(
            Text(
                f"Result: {outcome.outcome} · "
                f"{time.strftime('%Y-%m-%d', time.gmtime(outcome.created_at))}",
                style="cyan",
            )
        )
        console.print(Text(f"User report: {outcome.signal[:500]}"))
        console.print()


def _load_procedure_proposal(proposal_id: str) -> tuple[object, ProcedureProposal]:
    store = _get_episode_store()
    proposal = store.get_procedure_proposal(proposal_id)
    if proposal is None:
        store.close()
        raise click.ClickException(
            "Proposal not found. Use the complete ID or an unambiguous prefix "
            "of at least 8 hexadecimal characters."
        )
    return store, proposal


@procedures_group.command(name="show")
@click.argument("proposal_id")
def show_procedure_proposal(proposal_id: str) -> None:
    """Show one draft and possible matches from prior conversations."""
    console = Console()
    store, proposal = _load_procedure_proposal(proposal_id)
    try:
        related = store.similar_procedure_proposals(proposal.proposal_id)
        failures = store.similar_procedure_outcomes(
            proposal.title,
            session_id=proposal.session_id,
        )
    finally:
        store.close()
    console.print(f"[bold]Proposal:[/bold] {proposal.proposal_id}")
    console.print(f"[bold]Status:[/bold] {proposal.status}")
    console.print(f"[bold]Title:[/bold] {escape(proposal.title)}")
    console.print(f"[bold]Source episode:[/bold] {proposal.evidence_episode_id}")
    console.print("[bold]User success signal:[/bold]")
    console.print(Text(proposal.success_signal))
    console.print(Panel(Text(proposal.draft_markdown), title="Unreviewed draft"))
    if related:
        console.print(
            "[bold]Possible matches from other conversations[/bold] "
            "[dim](lexical title similarity; compare manually)[/dim]"
        )
        for match in related:
            earlier = match.proposal
            console.print(
                f"[cyan]{match.similarity:.0%}[/cyan] · {earlier.status} · "
                f"{earlier.proposal_id[:12]} · {escape(earlier.title)}"
            )
            console.print(Text(f"Earlier success signal: {earlier.success_signal}"))
            console.print(
                Panel(
                    Text(earlier.draft_markdown[:1_200]),
                    title="Earlier draft excerpt",
                )
            )
    if failures:
        console.print(
            "[bold]Reported failures on similar tasks[/bold] "
            "[dim](other conversations; user-reported and unverified)[/dim]"
        )
        for match in failures:
            earlier = match.outcome
            console.print(
                f"[yellow]{match.similarity:.0%}[/yellow] · "
                f"{escape(earlier.task_title)} · "
                f"episode {earlier.evidence_episode_id[:12]} · "
                f"{time.strftime('%Y-%m-%d', time.gmtime(earlier.created_at))}"
            )
            console.print(Text(earlier.signal))
    if proposal.skill_path:
        console.print(f"[bold]Approved skill:[/bold] {escape(proposal.skill_path)}")


def _procedure_skill_file(skill_name: str, *, create: bool = False) -> Path:
    validate_skill_name(skill_name)
    skills_root = Path(load_config().skills.skills_dir).expanduser()
    if skills_root.is_symlink():
        raise click.ClickException("Configured skills directory is a symbolic link.")
    if create:
        skills_root.mkdir(parents=True, exist_ok=True)
    if not skills_root.exists():
        raise FileNotFoundError("Configured skills directory does not exist.")
    resolved_root = skills_root.resolve(strict=True)
    skill_dir = resolved_root / skill_name
    if skill_dir.is_symlink():
        raise click.ClickException("Procedure skill directory is a symbolic link.")
    if create and not skill_dir.exists():
        skill_dir.mkdir(mode=0o700)
    if not skill_dir.exists():
        raise FileNotFoundError(f"Skill does not exist: {skill_name}")
    if skill_dir.resolve(strict=True).parent != resolved_root:
        raise click.ClickException("Procedure skill path escaped the skills directory.")
    if not skill_dir.is_dir():
        raise click.ClickException("Procedure skill destination is not a directory.")
    skill_file = skill_dir / "SKILL.md"
    if skill_file.is_symlink():
        raise click.ClickException("Procedure skill file is a symbolic link.")
    if skill_file.exists() and not skill_file.is_file():
        raise click.ClickException("Procedure skill destination is not a regular file.")
    return skill_file


def _preview_procedure_update(
    proposal: ProcedureProposal,
    skill_name: str,
) -> tuple[Path, str, str, str, str]:
    skill_file = _procedure_skill_file(skill_name)
    if not skill_file.is_file():
        raise FileNotFoundError(f"Skill file does not exist: {skill_name}")
    current = skill_file.read_text(encoding="utf-8")
    if skill_name_from_markdown(current) != skill_name:
        raise click.ClickException("Active skill frontmatter has a mismatched name.")
    current_version = skill_version_from_markdown(current)
    version = next_patch_version(current)
    updated = skill_markdown(
        proposal,
        proposal.evidence_episode_id,
        version=version,
        skill_name=skill_name,
    )
    return skill_file, current_version, version, current, updated


def _write_approved_procedure(
    proposal: ProcedureProposal,
    *,
    update_skill_name: str | None = None,
    expected_current_content: str | None = None,
) -> tuple[Path, str, str | None, bool]:
    history_root = get_config_dir() / "skill_versions"
    if update_skill_name:
        skill_file, _old_version, version, previous, updated = (
            _preview_procedure_update(proposal, update_skill_name)
        )
        if (
            expected_current_content is not None
            and previous != expected_current_content
        ):
            raise click.ClickException(
                "The active skill changed after review; inspect it and retry."
            )
        skill_file = _procedure_skill_file(update_skill_name)
        if skill_file.read_text(encoding="utf-8") != previous:
            raise click.ClickException(
                "The active skill changed while approval was being saved."
            )
        record_skill_version(history_root, update_skill_name, previous)
        record_skill_version(history_root, update_skill_name, updated)
        secure_write_text(skill_file, updated)
        return skill_file, version, previous, False

    skill_name = proposal.skill_name
    if not re.fullmatch(r"ace-procedure-[a-f0-9]{8,16}", skill_name):
        raise click.ClickException("The proposal has an invalid skill name.")
    skill_file = _procedure_skill_file(skill_name, create=True)
    skill_dir = skill_file.parent
    expected = skill_markdown(proposal, proposal.evidence_episode_id)
    if skill_file.exists():
        if skill_file.read_text(encoding="utf-8") == expected:
            record_skill_version(history_root, skill_name, expected)
            return skill_file, "0.1.0", None, False
        raise FileExistsError(skill_file)
    if any(skill_dir.iterdir()):
        raise FileExistsError(skill_dir)
    secure_write_text(skill_file, expected)
    try:
        record_skill_version(history_root, skill_name, expected)
    except Exception:
        skill_file.unlink(missing_ok=True)
        try:
            skill_dir.rmdir()
        except OSError:
            pass
        raise
    return skill_file, "0.1.0", None, True


def _restore_after_failed_approval(
    skill_file: Path,
    previous_content: str | None,
    created: bool,
) -> None:
    if skill_file.is_symlink():
        raise click.ClickException(
            "Approval state changed and the skill path became a symbolic link."
        )
    if previous_content is not None:
        secure_write_text(skill_file, previous_content)
    elif created:
        skill_file.unlink(missing_ok=True)


@procedures_group.command(name="approve")
@click.argument("proposal_id")
@click.option(
    "--update-skill",
    help="Create a new version of this existing ACE procedure skill.",
)
def approve_procedure_proposal(
    proposal_id: str,
    update_skill: str | None,
) -> None:
    """Approve a new skill or explicitly update an existing one."""
    console = Console()
    store, proposal = _load_procedure_proposal(proposal_id)
    try:
        if proposal.status == "approved":
            console.print(f"[green]Already approved:[/green] {proposal.skill_path}")
            return
        if proposal.status != "proposed":
            raise click.ClickException(f"Proposal is already {proposal.status}.")
        previous_content = None
        if update_skill:
            try:
                (
                    _skill_file,
                    current_version,
                    next_version,
                    previous_content,
                    updated_content,
                ) = _preview_procedure_update(proposal, update_skill)
            except (OSError, ValueError, RuntimeError) as exc:
                raise click.ClickException(
                    f"Could not prepare skill update: {exc}"
                ) from exc
            console.print(f"[bold]Skill:[/bold] {update_skill}")
            console.print(f"[bold]Version:[/bold] {current_version} → {next_version}")
            console.print(
                Panel(Text(updated_content), title="Proposed new skill version")
            )
            confirmation = (
                f"Save version {next_version} as the active skill? "
                "A copy of the previous version will be kept."
            )
        else:
            console.print(f"[bold]Proposed skill:[/bold] {proposal.skill_name}")
            console.print(f"[bold]Based on:[/bold] {escape(proposal.title)}")
            console.print(
                Panel(Text(proposal.draft_markdown), title="Review before approval")
            )
            confirmation = (
                "Save this draft as a user-invoked skill? "
                "It will not run automatically."
            )
        if not click.confirm(confirmation, default=False):
            console.print("[dim]Approval cancelled.[/dim]")
            return
        try:
            skill_file, version, previous, created = _write_approved_procedure(
                proposal,
                update_skill_name=update_skill,
                expected_current_content=previous_content if update_skill else None,
            )
        except click.ClickException:
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            raise click.ClickException(
                f"Could not save the approved skill: {exc}"
            ) from exc

        try:
            updated = store.set_procedure_proposal_status(
                proposal.proposal_id,
                status="approved",
                skill_path=str(skill_file),
            )
        except Exception:
            _restore_after_failed_approval(skill_file, previous, created)
            raise
        if not updated:
            _restore_after_failed_approval(skill_file, previous, created)
            raise click.ClickException(
                "The skill was written, but the proposal state changed. "
                "The previous active skill was restored."
            )
        console.print(
            f"[green]Saved skill version {version}:[/green] {escape(str(skill_file))}"
        )
    finally:
        store.close()


@procedures_group.command(name="versions")
@click.argument("skill_name")
def list_approved_skill_versions(skill_name: str) -> None:
    """List immutable snapshots for one approved ACE procedure skill."""
    try:
        validate_skill_name(skill_name)
        snapshots = list_skill_versions(
            get_config_dir() / "skill_versions",
            skill_name,
        )
        active_version = ""
        try:
            active = _procedure_skill_file(skill_name)
            if active.is_file():
                active_version = skill_version_from_markdown(
                    active.read_text(encoding="utf-8")
                )
        except FileNotFoundError:
            pass
    except (OSError, ValueError, RuntimeError) as exc:
        raise click.ClickException(f"Could not read skill history: {exc}") from exc
    if not snapshots:
        Console().print(f"[yellow]No saved versions for {skill_name}.[/yellow]")
        return
    table = Table(title=f"Skill Versions — {skill_name}")
    table.add_column("Version", style="cyan")
    table.add_column("State")
    table.add_column("Modified")
    table.add_column("SHA-256")
    for snapshot in snapshots:
        table.add_row(
            snapshot.version,
            "active" if snapshot.version == active_version else "archived",
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snapshot.modified_at)),
            snapshot.sha256[:12],
        )
    Console().print(table)


@procedures_group.command(name="rollback")
@click.argument("skill_name")
@click.argument("version")
def rollback_approved_skill(skill_name: str, version: str) -> None:
    """Restore an archived skill body as a new, explicitly confirmed version."""
    try:
        skill_file = _procedure_skill_file(skill_name)
        if not skill_file.is_file():
            raise FileNotFoundError(f"Skill file does not exist: {skill_name}")
        current = skill_file.read_text(encoding="utf-8")
        if skill_name_from_markdown(current) != skill_name:
            raise ValueError("Active skill frontmatter has a mismatched name.")
        current_version = skill_version_from_markdown(current)
        target = load_skill_version(
            get_config_dir() / "skill_versions",
            skill_name,
            version,
        )
        next_version = next_patch_version(current)
        restored = set_skill_version(target, next_version)
    except (OSError, ValueError, RuntimeError) as exc:
        raise click.ClickException(f"Could not prepare skill rollback: {exc}") from exc

    console = Console()
    console.print(f"[bold]Rollback:[/bold] {skill_name} {version} → {next_version}")
    console.print(
        f"[dim]Current version {current_version} will remain in the version "
        "history.[/dim]"
    )
    console.print(Panel(Text(restored), title="Restored content"))
    if not click.confirm(
        f"Make this restored content active as version {next_version}?",
        default=False,
    ):
        console.print("[dim]Rollback cancelled.[/dim]")
        return

    try:
        active_path = _procedure_skill_file(skill_name)
        if active_path.read_text(encoding="utf-8") != current:
            raise click.ClickException(
                "The active skill changed after review; inspect it and retry."
            )
        record_skill_version(
            get_config_dir() / "skill_versions",
            skill_name,
            current,
        )
        record_skill_version(
            get_config_dir() / "skill_versions",
            skill_name,
            restored,
        )
        secure_write_text(active_path, restored)
    except click.ClickException:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise click.ClickException(f"Could not save skill rollback: {exc}") from exc
    console.print(
        f"[green]Restored skill as version {next_version}:[/green] "
        f"{escape(str(active_path))}"
    )


@procedures_group.command(name="reject")
@click.argument("proposal_id")
def reject_procedure_proposal(proposal_id: str) -> None:
    """Reject a pending proposal without creating a skill."""
    console = Console()
    store, proposal = _load_procedure_proposal(proposal_id)
    try:
        if not store.set_procedure_proposal_status(
            proposal.proposal_id,
            status="rejected",
        ):
            raise click.ClickException(f"Proposal is already {proposal.status}.")
        console.print(f"[green]Rejected proposal {proposal.proposal_id}.[/green]")
    finally:
        store.close()


@memory.command(name="list")
def list_facts() -> None:
    """List durable facts captured by the automatic memory service."""
    console = Console()

    store = _get_fact_store()
    facts = store.list()
    if not facts:
        console.print("[yellow]No memory facts stored yet.[/yellow]")
        return

    table = Table(title=f"Memory Facts ({len(facts)})")
    table.add_column("#", style="dim", width=4)
    table.add_column("Fact")
    table.add_column("Source", style="cyan")
    table.add_column("Trust")
    any_quarantined = False
    for i, fact in enumerate(facts, 1):
        superseded = bool(getattr(fact, "superseded", False))
        if superseded:
            trust_disp = "[dim]superseded[/dim]"
        elif not fact.trusted_for_recall:
            any_quarantined = True
            trust_disp = "[red]⚠ quarantined[/red]"
        else:
            trust_disp = f"[green]{escape(fact.trust or 'legacy')}[/green]"
        table.add_row(str(i), escape(fact.text), escape(fact.source or "-"), trust_disp)
    console.print(table)
    if any_quarantined:
        console.print(
            "[red]⚠ quarantined[/red] facts are retained for audit but excluded "
            "from model recall because they may contain hostile input. Review "
            "one and run [bold]jarvis memory trust <#>[/bold] to allow recall."
        )


@memory.command(name="trust")
@click.argument("index", type=int)
def trust_fact(index: int) -> None:
    """Promote quarantined fact INDEX (as shown by `memory list`) to trusted.

    Only do this after reading the fact: a quarantined fact matched an
    injection pattern, and promoting it puts its text into future prompts.
    """
    console = Console()

    store = _get_fact_store()
    facts = store.list()
    if not 1 <= index <= len(facts):
        console.print(
            f"[red]No fact #{index}.[/red] "
            f"{len(facts)} fact(s) stored — see [bold]jarvis memory list[/bold]."
        )
        raise SystemExit(1)

    fact = facts[index - 1]
    if getattr(fact, "superseded", False):
        raise click.ClickException(
            "This fact is superseded. To restore that wording, revise its "
            "current replacement with jarvis memory facts revise."
        )
    if fact.trusted_for_recall:
        console.print(f"[yellow]Fact #{index} is already recallable.[/yellow]")
        return
    if not store.promote_reviewed(index - 1, fact.text):
        console.print(f"[red]Could not update fact #{index}.[/red]")
        raise SystemExit(1)
    _prune_fact_embeddings(store.list())
    console.print(f"[green]Fact #{index} is now trusted for recall:[/green]")
    console.print(f"  {fact.text}")


@memory.command(name="forget")
@click.argument("index", type=int)
def forget_fact(index: int) -> None:
    """Delete one durable fact by its 1-based index from `memory list`."""
    console = Console()
    store = _get_fact_store()
    facts = store.list()
    if not 1 <= index <= len(facts):
        console.print(
            f"[red]No fact #{index}.[/red] "
            f"{len(facts)} fact(s) stored — see [bold]jarvis memory list[/bold]."
        )
        raise SystemExit(1)

    fact = facts[index - 1]
    if not store.remove_reviewed(index - 1, expected_text=fact.text):
        console.print(
            f"[red]Could not remove fact #{index}.[/red] "
            "The memory list may have changed; run `jarvis memory list` again."
        )
        raise SystemExit(1)
    _prune_fact_embeddings(store.list())
    console.print(f"[green]Forgot fact #{index}:[/green] {fact.text}")


@memory.command()
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip the confirmation prompt.",
)
def clear(yes: bool) -> None:
    """Remove all durable facts captured by the automatic memory service."""
    console = Console()

    store = _get_fact_store()
    count = store.count()
    if count == 0:
        _prune_fact_embeddings(clear=True)
        console.print("[yellow]No memory facts to clear.[/yellow]")
        return

    if not yes:
        if not click.confirm(f"Remove all {count} stored memory fact(s)?"):
            console.print("[dim]Aborted.[/dim]")
            return

    removed = store.clear()
    _prune_fact_embeddings(clear=True)
    console.print(f"[green]Cleared {removed} memory fact(s).[/green]")


@memory.command()
@click.option(
    "--backend",
    "-b",
    default=None,
    help="Override the default memory backend.",
)
def stats(backend: str | None) -> None:
    """Show memory store statistics."""
    console = Console()

    mem = _get_backend(backend)
    try:
        count = 0
        if hasattr(mem, "count"):
            count = mem.count()

        table = Table(title="Memory Statistics")
        table.add_column("Property", style="cyan")
        table.add_column("Value")
        table.add_row("Backend", mem.backend_id)
        table.add_row("Documents", str(count))

        if hasattr(mem, "_db_path"):
            db_path = Path(mem._db_path)
            if db_path.exists():
                size_kb = db_path.stat().st_size / 1024
                table.add_row(
                    "Database size",
                    f"{size_kb:.1f} KB",
                )
            table.add_row("Database path", str(db_path))

        console.print(table)
    finally:
        if hasattr(mem, "close"):
            mem.close()
