"""CLI for local-first, reviewable self-improvement proposals."""

from __future__ import annotations

from pathlib import Path

import click

from openjarvis.learning.code_improvement import (
    CodeImprovementManager,
    discover_related_python_files,
    generate_local_patch,
)


def _local_patch(
    task: str, files: tuple[str, ...], repository: Path
) -> tuple[str, dict[str, str]]:
    """Ask the local engine for a diff and return hashes of its exact context."""
    try:
        return generate_local_patch(task, files, repository)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


def _open_manager() -> CodeImprovementManager:
    try:
        return CodeImprovementManager()
    except (OSError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc


@click.group("improve")
def improve() -> None:
    """Review, apply, or roll back code improvements."""


@improve.command("propose")
@click.option("--task", required=True, help="Describe the code improvement.")
@click.option(
    "--file",
    "files",
    multiple=True,
    required=True,
    help="Source file the local model may inspect. Repeat for multiple files.",
)
@click.option(
    "--discover-related",
    is_flag=True,
    help="Add up to 8 nearby Python modules found from local imports.",
)
@click.option("--repo", "repository", type=click.Path(file_okay=False), default=".")
def propose(
    task: str,
    files: tuple[str, ...],
    discover_related: bool,
    repository: str,
) -> None:
    """Generate a local-model proposal without editing the active checkout."""
    root = Path(repository).expanduser().resolve()
    files = tuple(dict.fromkeys(files))
    if discover_related:
        try:
            related = discover_related_python_files(root, files)
        except (OSError, ValueError, RuntimeError) as exc:
            raise click.ClickException(f"Related-file discovery failed: {exc}") from exc
        files = tuple(dict.fromkeys((*files, *related)))
        click.echo(
            "Related files added: " + (", ".join(related) if related else "none found")
        )
    patch, source_hashes = _local_patch(task, files, root)
    try:
        proposal = CodeImprovementManager().propose(
            root,
            task,
            patch,
            expected_source_hashes=source_hashes,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise click.ClickException(f"Proposal rejected: {exc}") from exc
    click.echo(f"Proposal {proposal.proposal_id} saved ({proposal.status}).")
    click.echo(f"Files: {', '.join(proposal.files)}")
    click.echo(f"Review with: jarvis improve show {proposal.proposal_id}")
    click.echo(f"Validate with: jarvis improve validate {proposal.proposal_id}")


@improve.command("list")
def list_proposals() -> None:
    """List saved code improvement proposals."""
    for item in _open_manager().list():
        click.echo(
            f"{item.proposal_id}  {item.status:10}  {item.title} "
            f"[{', '.join(item.files)}]"
        )


@improve.command("show")
@click.argument("proposal_id")
def show(proposal_id: str) -> None:
    """Print the proposal diff and its current state."""
    item = _open_manager().get(proposal_id)
    if item is None:
        raise click.ClickException(f"Unknown proposal: {proposal_id}")
    click.echo(f"{item.title} [{item.status}] — {item.proposal_id}")
    click.echo(f"Repository: {item.repository}")
    click.echo(f"Files: {', '.join(item.files)}")
    if item.validation:
        click.echo(f"Validation: {item.validation}")
    if item.validation_receipt:
        click.echo(f"Validation receipt: {item.validation_receipt}")
    click.echo()
    click.echo(item.patch, nl=not item.patch.endswith("\n"))


@improve.command("validate")
@click.argument("proposal_id")
@click.option(
    "--run-tests",
    is_flag=True,
    help="Run the full Python pytest suite in the Docker sandbox.",
)
@click.option(
    "--test-path",
    "test_paths",
    multiple=True,
    help="Run selected test paths in Docker; repeat to select multiple paths.",
)
@click.option(
    "--frontend",
    "run_frontend",
    is_flag=True,
    help="Run frontend npm test and production build in the Docker sandbox.",
)
@click.option(
    "--rust",
    "run_rust",
    is_flag=True,
    help="Run cargo fmt check and Rust workspace crate tests in Docker.",
)
@click.option(
    "--runtime",
    type=click.Choice(["docker", "podman"]),
    default="docker",
    show_default=True,
    help="Container runtime used for isolated tests.",
)
@click.option(
    "--image",
    default="openjarvis-sandbox:latest",
    show_default=True,
    help="Existing local image used for isolated tests.",
)
@click.option("--timeout", type=click.IntRange(1, 3600), default=900, show_default=True)
def validate(
    proposal_id: str,
    run_tests: bool,
    test_paths: tuple[str, ...],
    run_frontend: bool,
    run_rust: bool,
    runtime: str,
    image: str,
    timeout: int,
) -> None:
    """Check patch application and Python syntax in a disposable copy."""
    selected_tests = test_paths or (("tests",) if run_tests else ())
    try:
        item = CodeImprovementManager().validate(
            proposal_id,
            test_paths=selected_tests,
            run_frontend=run_frontend,
            run_rust=run_rust,
            runtime=runtime,
            image=image,
            timeout=timeout,
        )
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        raise click.ClickException(f"Validation failed: {exc}") from exc
    click.echo(f"Proposal {item.proposal_id} validated.")
    click.echo(item.validation)
    click.echo(f"Validation receipt: {item.validation_receipt}")
    click.echo("Review it, then run `jarvis improve apply <id>` to approve and apply.")


@improve.command("apply")
@click.argument("proposal_id")
def apply(proposal_id: str) -> None:
    """Confirm and apply a validated or recovered proposal."""
    manager = _open_manager()
    item = manager.get(proposal_id)
    if item is None:
        raise click.ClickException(f"Unknown proposal: {proposal_id}")
    if item.status not in ("validated", "approved"):
        raise click.ClickException(
            "Validate and review the proposal before applying it."
        )
    if not item.validation_receipt:
        raise click.ClickException(
            "Run jarvis improve validate to bind this proposal to its validation."
        )
    click.echo(f"Title: {item.title}")
    click.echo(f"Repository: {item.repository}")
    click.echo(f"Files: {', '.join(item.files)}")
    if item.validation:
        click.echo(f"Validation: {item.validation}")
    click.echo(f"Validation receipt: {item.validation_receipt}")
    click.echo("Proposed diff:")
    click.echo(item.patch, nl=not item.patch.endswith("\n"))
    if item.status == "approved":
        confirmation = "Confirm retry of this previously approved diff?"
    else:
        confirmation = "Approve and apply this exact diff?"
    if not click.confirm(confirmation):
        click.echo("Proposal left unchanged.")
        return
    try:
        if item.status == "validated":
            manager.approve(proposal_id, expected_receipt=item.validation_receipt)
        updated = manager.apply(proposal_id, expected_receipt=item.validation_receipt)
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        raise click.ClickException(f"Could not apply proposal: {exc}") from exc
    click.echo(f"Proposal {updated.proposal_id} applied.")


@improve.command("rollback")
@click.argument("proposal_id")
def rollback(proposal_id: str) -> None:
    """Confirm restoration of the files saved before an applied proposal."""
    manager = _open_manager()
    item = manager.get(proposal_id)
    if item is None:
        raise click.ClickException(f"Unknown proposal: {proposal_id}")
    if item.status == "rolled_back":
        click.echo(f"Proposal {proposal_id} has already been rolled back.")
        return
    if item.status not in ("applied", "rolling_back"):
        raise click.ClickException("Only an applied proposal can be rolled back.")

    if item.status == "applied":
        click.echo(f"Title: {item.title}")
        click.echo(f"Repository: {item.repository}")
        click.echo(f"Files: {', '.join(item.files)}")
        if item.validation_receipt:
            click.echo(f"Validation receipt: {item.validation_receipt}")
        if not click.confirm("Restore the saved originals for this proposal?"):
            click.echo("Proposal left unchanged.")
            return

    try:
        updated = manager.rollback(
            proposal_id, expected_receipt=item.validation_receipt
        )
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        raise click.ClickException(f"Could not roll back proposal: {exc}") from exc
    click.echo(f"Proposal {updated.proposal_id} rolled back.")


@improve.command("reject")
@click.argument("proposal_id")
def reject(proposal_id: str) -> None:
    """Reject and archive a proposal."""
    try:
        item = _open_manager().reject(proposal_id)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Proposal {item.proposal_id} rejected.")


__all__ = ["improve"]
