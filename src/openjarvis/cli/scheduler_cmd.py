"""``jarvis scheduler`` -- task scheduling commands."""

from __future__ import annotations

import signal
import threading
from typing import Optional

import click
from rich.console import Console
from rich.markup import escape
from rich.table import Table


def _get_store() -> "SchedulerStore":  # noqa: F821
    """Build a SchedulerStore from user config."""
    from openjarvis.core.config import DEFAULT_CONFIG_DIR, load_config
    from openjarvis.scheduler.store import SchedulerStore

    config = load_config()
    db_path = getattr(getattr(config, "scheduler", None), "db_path", None) or str(
        DEFAULT_CONFIG_DIR / "scheduler.db"
    )
    return SchedulerStore(db_path)


def _get_scheduler(store: "SchedulerStore") -> "TaskScheduler":  # noqa: F821
    """Build a TaskScheduler from a store."""
    from openjarvis.scheduler.scheduler import TaskScheduler

    return TaskScheduler(store)


@click.group()
def scheduler() -> None:
    """Manage scheduled tasks."""


@scheduler.command("install-startup")
def scheduler_install_startup() -> None:
    """Start the scheduler automatically at this Windows user's login."""
    from openjarvis.scheduler.startup import install_scheduler_startup

    try:
        result = install_scheduler_startup()
    except Exception as exc:
        raise click.ClickException(
            f"Could not install scheduler login startup: {exc}"
        ) from exc
    console = Console()
    console.print(
        f"[green]{result['output'] or 'ACE scheduler login startup installed.'}[/green]"
    )
    console.print(
        "It runs as this user at the next login. Remove it with "
        "`jarvis scheduler remove-startup`."
    )


@scheduler.command("remove-startup")
def scheduler_remove_startup() -> None:
    """Remove automatic scheduler startup for this user."""
    from openjarvis.scheduler.startup import remove_scheduler_startup

    try:
        result = remove_scheduler_startup()
    except Exception as exc:
        raise click.ClickException(
            f"Could not remove scheduler login startup: {exc}"
        ) from exc
    Console().print(
        f"[green]{result['output'] or 'ACE scheduler login startup removed.'}[/green]"
    )
    Console().print(
        "This affects future logins. Use `jarvis scheduler stop` to stop a "
        "currently running scheduler after its active task finishes."
    )


@scheduler.command("startup-status")
def scheduler_startup_status() -> None:
    """Show this user's automatic scheduler startup status."""
    from openjarvis.scheduler.startup import get_scheduler_startup_status

    try:
        status = get_scheduler_startup_status()
    except Exception as exc:
        raise click.ClickException(
            f"Could not read scheduler login startup status: {exc}"
        ) from exc
    if not status.get("installed"):
        Console().print(
            "[yellow]ACE scheduler login startup is not installed.[/yellow]"
        )
        return
    Console().print(
        "[green]ACE scheduler login startup is installed.[/green] "
        f"Task state: {status.get('state', 'unknown')}; "
        f"last run: {status.get('last_run') or 'unknown'}; "
        f"last result: {status.get('last_result', 'unknown')}"
    )


@scheduler.command("stop")
def scheduler_stop() -> None:
    """Request a persistent scheduler to stop after its current task."""
    from openjarvis.core.config import DEFAULT_CONFIG_DIR, load_config
    from openjarvis.scheduler.instance_lock import SchedulerInstanceLock

    config = load_config()
    database_path = config.scheduler.db_path or str(DEFAULT_CONFIG_DIR / "scheduler.db")
    if database_path == ":memory:":
        Console().print("[yellow]No persistent scheduler is running.[/yellow]")
        return
    lock = SchedulerInstanceLock(database_path)
    try:
        if lock.acquire():
            lock.release()
            lock.clear_stop_request()
            Console().print("[yellow]No persistent scheduler is running.[/yellow]")
            return
        lock.request_stop()
    except Exception as exc:
        raise click.ClickException(f"Could not stop the scheduler: {exc}") from exc
    Console().print(
        "[yellow]Scheduler stop requested. It will finish its current task "
        "before exiting.[/yellow]"
    )


@scheduler.command("create")
@click.argument("prompt")
@click.option(
    "--type",
    "schedule_type",
    required=True,
    type=click.Choice(["cron", "interval", "once"]),
    help="Schedule type.",
)
@click.option(
    "--value",
    "schedule_value",
    required=True,
    help="Schedule value (cron expr, seconds, or ISO datetime).",
)
@click.option("--agent", default="simple", help="Agent to use for execution.")
@click.option("--tools", default="", help="Comma-separated tool names.")
def scheduler_create(
    prompt: str,
    schedule_type: str,
    schedule_value: str,
    agent: str,
    tools: str,
) -> None:
    """Create a new scheduled task."""
    console = Console()
    store = _get_store()
    try:
        sched = _get_scheduler(store)
        task = sched.create_task(
            prompt=prompt,
            schedule_type=schedule_type,
            schedule_value=schedule_value,
            agent=agent,
            tools=tools,
        )
        console.print(f"[green]Created task {task.id}[/green]")
        console.print(f"  Type: {task.schedule_type}")
        console.print(f"  Value: {task.schedule_value}")
        console.print(f"  Next run: {task.next_run or 'N/A'}")
        console.print(f"  Agent: {task.agent}")
        if task.tools:
            console.print(f"  Tools: {task.tools}")
    finally:
        store.close()


@scheduler.command("list")
@click.option(
    "--status",
    default=None,
    type=click.Choice(
        [
            "active",
            "pending_approval",
            "paused",
            "review_required",
            "completed",
            "cancelled",
        ]
    ),
    help="Filter by status.",
)
def scheduler_list(status: Optional[str]) -> None:
    """List scheduled tasks."""
    console = Console()
    store = _get_store()
    try:
        sched = _get_scheduler(store)
        tasks = sched.list_tasks(status=status)

        if not tasks:
            console.print("[dim]No scheduled tasks found.[/dim]")
            return

        table = Table(title="Scheduled Tasks")
        table.add_column("ID", style="cyan")
        table.add_column("Prompt", max_width=40)
        table.add_column("Type", style="blue")
        table.add_column("Status", style="green")
        table.add_column("Next Run")
        table.add_column("Agent")

        for t in tasks:
            prompt_short = t.prompt[:37] + "..." if len(t.prompt) > 40 else t.prompt
            status_style = {
                "active": "green",
                "pending_approval": "yellow",
                "paused": "yellow",
                "review_required": "yellow",
                "completed": "dim",
                "cancelled": "red",
            }.get(t.status, "white")
            table.add_row(
                t.id,
                prompt_short,
                t.schedule_type,
                f"[{status_style}]{t.status}[/{status_style}]",
                t.next_run or "N/A",
                t.agent,
            )
        console.print(table)
    finally:
        store.close()


@scheduler.command("pause")
@click.argument("task_id")
def scheduler_pause(task_id: str) -> None:
    """Pause a scheduled task."""
    console = Console()
    store = _get_store()
    try:
        sched = _get_scheduler(store)
        sched.pause_task(task_id)
        console.print(f"[yellow]Task {task_id} paused[/yellow]")
    except KeyError:
        console.print(f"[red]Task not found: {task_id}[/red]")
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        store.close()


@scheduler.command("resume")
@click.argument("task_id")
def scheduler_resume(task_id: str) -> None:
    """Resume a paused task or review an interrupted recurring run."""
    console = Console()
    store = _get_store()
    try:
        sched = _get_scheduler(store)
        task = store.get_task(task_id)
        if task is None:
            raise KeyError(task_id)
        if task["status"] == "review_required":
            console.print(
                "The previous run may have taken effect. "
                f"Inspect its log with jarvis scheduler logs {escape(task_id)}."
            )
            if task["schedule_type"] == "once":
                raise RuntimeError(
                    "Review its outcome and create a new task if it should run again."
                )
            if not click.confirm(
                "Resume at the next scheduled time, skipping the interrupted run?",
                default=False,
            ):
                console.print("[dim]Task still needs review.[/dim]")
                return
            sched.resume_task(task_id, reviewed_interruption=True)
        else:
            sched.resume_task(task_id)
        console.print(f"[green]Task {escape(task_id)} resumed[/green]")
    except KeyError:
        console.print(f"[red]Task not found: {task_id}[/red]")
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        store.close()


@scheduler.command("approve")
@click.argument("task_id")
def scheduler_approve(task_id: str) -> None:
    """Review and approve an agent-created scheduled task."""
    console = Console()
    store = _get_store()
    try:
        sched = _get_scheduler(store)
        task = next((item for item in sched.list_tasks() if item.id == task_id), None)
        if task is None:
            console.print(f"[red]Task not found: {task_id}[/red]")
            return
        if task.status != "pending_approval":
            console.print(
                f"[yellow]Task {task_id} is not waiting for approval.[/yellow]"
            )
            return

        console.print("[bold]Review scheduled task[/bold]")
        console.print(f"  Prompt: {task.prompt}")
        console.print(f"  Schedule: {task.schedule_type} {task.schedule_value}")
        console.print(f"  Next run: {task.next_run or 'N/A'}")
        console.print(f"  Agent: {task.agent}")
        if task.metadata.get("_openjarvis_explicit_tool_scope"):
            console.print(f"  Tools: {task.tools or '(none)'}")
        else:
            console.print(f"  Tools: {task.tools or '(agent defaults)'}")
        if not click.confirm(
            "Approve and activate this task for scheduled execution?", default=False
        ):
            console.print("[dim]Approval cancelled; task remains pending.[/dim]")
            return

        sched.approve_task(task_id)
        console.print(f"[green]Task {task_id} approved and activated[/green]")
    except KeyError:
        console.print(f"[red]Task not found: {task_id}[/red]")
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        store.close()


@scheduler.command("cancel")
@click.argument("task_id")
def scheduler_cancel(task_id: str) -> None:
    """Cancel a scheduled task."""
    console = Console()
    store = _get_store()
    try:
        sched = _get_scheduler(store)
        sched.cancel_task(task_id)
        console.print(f"[red]Task {task_id} cancelled[/red]")
    except KeyError:
        console.print(f"[red]Task not found: {task_id}[/red]")
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        store.close()


@scheduler.command("logs")
@click.argument("task_id")
@click.option("-n", "--limit", default=10, type=int, help="Number of logs to show.")
def scheduler_logs(task_id: str, limit: int) -> None:
    """Show run logs for a scheduled task."""
    console = Console()
    store = _get_store()
    try:
        logs = store.get_run_logs(task_id, limit=limit)
        if not logs:
            console.print(f"[dim]No run logs for task {task_id}[/dim]")
            return

        table = Table(title=f"Run Logs for {task_id}")
        table.add_column("ID", justify="right")
        table.add_column("Started")
        table.add_column("Finished")
        table.add_column("Outcome")
        table.add_column("Result", max_width=40)
        table.add_column("Error", max_width=30)

        for log in logs:
            outcome = log.get("status") or ("succeeded" if log["success"] else "failed")
            color = {
                "succeeded": "green",
                "failed": "red",
                "running": "cyan",
                "interrupted": "yellow",
            }.get(outcome, "white")
            result_text = log["result"]
            result_short = (
                (result_text[:37] + "...") if len(result_text) > 40 else result_text
            )
            table.add_row(
                str(log["id"]),
                log["started_at"],
                log.get("finished_at") or "N/A",
                f"[{color}]{escape(outcome)}[/{color}]",
                escape(result_short),
                escape(log.get("error", "")),
            )
        console.print(table)
    finally:
        store.close()


@scheduler.command("run-task")
@click.argument("agent_name")
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print what would run without executing.",
)
def scheduler_run_task(agent_name: str, dry_run: bool) -> None:
    """Immediately execute the active task for AGENT_NAME.

    Finds the first active scheduled task whose agent matches AGENT_NAME
    and runs it right now — useful for testing and for launchd invocation
    when OpenJarvis is not running as a persistent daemon.

    Example (launchd plist ProgramArguments):
        jarvis scheduler run-task proactive
    """
    console = Console()
    store = _get_store()
    try:
        sched = _get_scheduler(store)
        tasks = sched.list_tasks(status="active")
        match = next((t for t in tasks if t.agent == agent_name), None)

        if match is None:
            console.print(
                f"[yellow]No active task found for agent '{agent_name}'. "
                "Register it first with 'jarvis scheduler create'.[/yellow]"
            )
            return

        if dry_run:
            console.print("[dim]Dry run — would execute:[/dim]")
            console.print(f"  Task : {match.id}")
            console.print(f"  Agent: {match.agent}")
            console.print(f"  Prompt: {match.prompt[:80]}")
            return

        console.print(f"Running task [cyan]{match.id}[/cyan] (agent: {match.agent})…")

        from openjarvis.core.config import load_config
        from openjarvis.system import SystemBuilder

        with SystemBuilder(load_config()).build() as system:
            sched.set_system(system)
            outcome = sched.run_task_now(match.id)
        if outcome is None:
            raise RuntimeError(
                "Task changed before dispatch. Review its current state."
            )
        success, result = outcome
        if not success:
            raise RuntimeError(result or "The scheduled task reported a failure.")
        console.print("[green]Done.[/green]")
        if result:
            console.print(escape(result))
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise SystemExit(1)
    finally:
        store.close()


def _build_scheduler_runtime(poll_interval: int, *, dry_run: bool):
    """Create a scheduler with a live JarvisSystem unless dry-run was requested."""
    if dry_run:
        from openjarvis.scheduler.scheduler import TaskScheduler

        store = _get_store()
        return TaskScheduler(store, poll_interval=poll_interval), None, store

    from openjarvis.core.config import load_config
    from openjarvis.system.builder import SystemBuilder

    config = load_config()
    config.scheduler.enabled = True
    config.scheduler.poll_interval = poll_interval
    system = SystemBuilder(config).scheduler(True).build()
    if system.scheduler is None:
        system.close()
        raise RuntimeError("The configured task scheduler could not be created")
    return system.scheduler, system, None


def _stop_scheduler_runtime(sched, system, store) -> None:
    if system is not None:
        system.close()
    else:
        sched.stop()
        store.close()


@scheduler.command("start")
@click.option(
    "--poll-interval",
    default=60,
    type=int,
    help="Seconds between poll cycles.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Log due tasks without invoking an agent or its tools.",
)
def scheduler_start(poll_interval: int, dry_run: bool) -> None:
    """Run scheduled tasks against the configured JarvisSystem."""
    console = Console()
    try:
        sched, system, store = _build_scheduler_runtime(poll_interval, dry_run=dry_run)
    except Exception as exc:
        raise click.ClickException(f"Could not initialize scheduler: {exc}") from exc

    sched.start()
    mode = "dry-run" if dry_run else "live"
    console.print(
        f"[green]Scheduler running in {mode} mode (poll every {poll_interval}s). "
        "Press Ctrl+C to stop.[/green]"
    )

    shutdown_requested = threading.Event()

    def _handle_signal(signum: int, frame: object) -> None:
        shutdown_requested.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not shutdown_requested.wait(timeout=0.5):
            if getattr(sched, "external_stop_requested", False):
                break
    finally:
        _stop_scheduler_runtime(sched, system, store)
        try:
            clear_stop_request = getattr(sched, "clear_external_stop_request", None)
            if clear_stop_request is not None:
                clear_stop_request()
        finally:
            console.print("\n[yellow]Scheduler stopped.[/yellow]")
