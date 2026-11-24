"""Run ACE's local resource monitor without an interactive chat session."""

from __future__ import annotations

import time

import click
from rich.console import Console
from rich.text import Text


@click.command()
@click.option(
    "--once",
    "once_only",
    is_flag=True,
    help="Take one local sample, record threshold changes, and exit.",
)
@click.option(
    "--background",
    "background",
    is_flag=True,
    help="Start the local resource monitor detached from this terminal.",
)
@click.option("--status", "show_status", is_flag=True, help="Show monitor status.")
@click.option(
    "--history",
    "show_history",
    is_flag=True,
    help="Review recent local incidents and saved assessments.",
)
@click.option(
    "--history-limit",
    type=click.IntRange(1, 50),
    default=10,
    show_default=True,
    help="Number of incidents to show with --history.",
)
@click.option(
    "--unacknowledged",
    is_flag=True,
    help="With --history, show incidents awaiting review.",
)
@click.option(
    "--acknowledge",
    "incident_to_acknowledge",
    metavar="ID",
    type=click.IntRange(1, 2**63 - 1),
    help="Mark a local incident as reviewed.",
)
@click.option(
    "--install-startup",
    "install_startup",
    is_flag=True,
    help="Start ACE monitoring automatically at this user's next login.",
)
@click.option(
    "--remove-startup",
    "remove_startup",
    is_flag=True,
    help="Remove ACE's automatic login startup for this user.",
)
@click.option(
    "--startup-status",
    "show_startup_status",
    is_flag=True,
    help="Show whether automatic ACE monitor startup is installed.",
)
@click.option(
    "--stop",
    "stop_background",
    is_flag=True,
    help="Stop the background monitor.",
)
@click.option(
    "--monitor-screen",
    "monitor_screen",
    is_flag=True,
    help=(
        "With --background, opt in to local screen analysis. Text assessments "
        "are written to the local monitor log; images are not saved."
    ),
)
@click.option("--daemon-worker", is_flag=True, hidden=True)
def monitor(
    once_only: bool,
    background: bool,
    show_status: bool,
    show_history: bool,
    history_limit: int,
    unacknowledged: bool,
    incident_to_acknowledge: int | None,
    install_startup: bool,
    remove_startup: bool,
    show_startup_status: bool,
    stop_background: bool,
    monitor_screen: bool,
    daemon_worker: bool,
) -> None:
    """Monitor local resources and optionally observe the screen with a local model."""
    selected_modes = sum(
        (
            once_only,
            background,
            show_status,
            show_history,
            incident_to_acknowledge is not None,
            install_startup,
            remove_startup,
            show_startup_status,
            stop_background,
            daemon_worker,
        )
    )
    if selected_modes > 1:
        raise click.UsageError(
            "Choose only one monitor mode: --once, --background, --status, "
            "--history, --acknowledge, --stop, --install-startup, "
            "--remove-startup, or --startup-status."
        )

    if unacknowledged and not show_history:
        raise click.UsageError("--unacknowledged requires --history.")

    if monitor_screen and not (background or daemon_worker):
        raise click.UsageError("--monitor-screen requires --background.")

    if daemon_worker:
        import os

        from openjarvis.proactive.monitor_daemon import run_monitor_worker

        token = os.environ.get("OPENJARVIS_MONITOR_LAUNCH_TOKEN", "")
        raise SystemExit(run_monitor_worker(token, monitor_screen))

    if install_startup:
        from openjarvis.core.config import load_config
        from openjarvis.proactive.startup import install_monitor_startup

        try:
            config = load_config()
            if not config.proactive.local_monitor_enabled:
                raise click.ClickException(
                    "Enable [proactive].local_monitor_enabled in config.toml "
                    "before installing automatic startup."
                )
            result = install_monitor_startup()
        except click.ClickException:
            raise
        except Exception as exc:
            raise click.ClickException(
                f"Could not install ACE login startup: {exc}"
            ) from exc
        Console().print(
            f"[green]{result['output'] or 'ACE login startup installed.'}[/green]"
        )
        Console().print("Start it now with `jarvis monitor --background`.")
        return

    if remove_startup:
        from openjarvis.proactive.startup import remove_monitor_startup

        try:
            result = remove_monitor_startup()
        except Exception as exc:
            raise click.ClickException(
                f"Could not remove ACE login startup: {exc}"
            ) from exc
        Console().print(
            f"[green]{result['output'] or 'ACE login startup removed.'}[/green]"
        )
        return

    if show_startup_status:
        from openjarvis.proactive.startup import get_monitor_startup_status

        try:
            status = get_monitor_startup_status()
        except Exception as exc:
            raise click.ClickException(
                f"Could not read ACE login startup status: {exc}"
            ) from exc
        if status.get("installed"):
            details = (
                f"Task state: {status.get('state', 'unknown')}; "
                f"last run: {status.get('last_run', 'unknown')}"
                if "last_run" in status
                else f"Startup manager state: {status.get('state', 'unknown')}"
            )
            Console().print(
                "[green]ACE automatic startup is installed.[/green] " + details
            )
        else:
            Console().print("[yellow]ACE automatic startup is not installed.[/yellow]")
        return

    if show_status:
        from openjarvis.proactive.monitor_daemon import get_monitor_daemon_status

        state = get_monitor_daemon_status()
        if state is None:
            Console().print("[yellow]ACE background monitor is not running.[/yellow]")
        else:
            screen_status = (
                "enabled"
                if state.get("screen_monitor_enabled", False)
                else "off"
            )
            Console().print(
                f"[green]ACE background monitor {state['status']}[/green] "
                f"(PID {state['pid']}, every {state['interval_seconds']}s).\n"
                f"  Screen monitoring: {screen_status}\n"
                f"  Started: {state['started_at']}\n"
                f"  Log: {state['log_file']}"
            )
        return

    if incident_to_acknowledge is not None:
        from openjarvis.proactive.local_monitor import LocalMonitorStore

        incident = f"Local incident #{incident_to_acknowledge}"
        if not LocalMonitorStore().acknowledge(incident_to_acknowledge):
            raise click.ClickException(f"{incident} not found.")
        Console().print(Text(f"{incident} acknowledged.", style="green"))
        return

    if show_history:
        from openjarvis.proactive.local_monitor import LocalMonitorStore

        incidents = LocalMonitorStore().recent(
            history_limit, unacknowledged_only=unacknowledged
        )
        if not incidents:
            message = (
                "No local incidents are awaiting review."
                if unacknowledged
                else "No local monitor history is available."
            )
            Console().print(Text(message, style="yellow"))
            return
        console = Console()
        console.print("[bold]A.C.E. local monitor history[/bold]")
        for incident in incidents:
            assessments = incident.get("assessments", [])
            latest = (
                str(assessments[0].get("analysis", ""))
                if assessments
                else "No local assessment saved"
            )
            console.print(
                Text(
                    f"#{incident.get('id', '')} {incident.get('metric', '')} — "
                    f"{incident.get('severity', '')} / {incident.get('status', '')}",
                    style="bold",
                )
            )
            console.print(Text(str(incident.get("summary", ""))))
            acknowledged_at = incident.get("acknowledged_at")
            review_state = (
                f"Review: acknowledged at {acknowledged_at}"
                if acknowledged_at
                else "Review: awaiting acknowledgement"
            )
            console.print(Text(review_state, style="dim"))
            console.print(Text(f"Latest assessment: {latest}", style="dim"))
            captured_at = incident.get("memory_captured_at")
            memory_state = (
                f"Episodic memory: saved at {captured_at}"
                if captured_at
                else "Episodic memory: not recorded"
            )
            console.print(Text(memory_state, style="dim"))
            console.print()
        return

    if stop_background:
        from openjarvis.proactive.monitor_daemon import stop_monitor_daemon

        try:
            state = stop_monitor_daemon()
        except Exception as exc:
            raise click.ClickException(f"Could not stop ACE monitor: {exc}") from exc
        if state is None:
            Console().print("[yellow]ACE background monitor is not running.[/yellow]")
        else:
            Console().print(
                f"[green]ACE background monitor stopped[/green] (PID {state['pid']})."
            )
        return

    from openjarvis.core.config import load_config
    from openjarvis.proactive.local_monitor import LocalProactiveMonitor

    try:
        config = load_config()
    except Exception as exc:
        raise click.ClickException(f"Could not load configuration: {exc}") from exc

    if not once_only and not config.proactive.local_monitor_enabled:
        raise click.ClickException(
            "Continuous monitoring is disabled. Enable "
            "[proactive].local_monitor_enabled in config.toml or run `jarvis "
            "monitor --once` for a single local sample."
        )

    console = Console()
    interval = min(max(int(config.proactive.local_monitor_interval_seconds), 15), 3600)

    if background:
        from openjarvis.proactive.monitor_daemon import start_monitor_daemon

        try:
            state = start_monitor_daemon(
                interval,
                screen_monitor_enabled=monitor_screen,
            )
        except Exception as exc:
            raise click.ClickException(f"Could not start ACE monitor: {exc}") from exc
        console.print(
            f"[green]ACE background monitor {state['status']}[/green] "
            f"(PID {state['pid']}, every {interval}s).\n"
            f"  Screen monitoring: {'enabled' if monitor_screen else 'off'}\n"
            f"  Log: {state['log_file']}\n"
            "Use `jarvis monitor --status` to inspect it and "
            "`jarvis monitor --stop` to stop it."
        )
        return

    def show_notice(message: str) -> None:
        console.print(f"[yellow]ACE monitor:[/yellow] {message}")

    try:
        service = LocalProactiveMonitor(config.proactive, on_notice=show_notice)
    except Exception as exc:
        raise click.ClickException(f"Could not start local monitor: {exc}") from exc

    if once_only:
        try:
            notices = service.poll_once()
        except Exception as exc:
            raise click.ClickException(f"Local sample failed: {exc}") from exc
        if not notices:
            console.print("[green]No new resource alerts or recoveries.[/green]")
        return

    if not service.start():
        raise click.ClickException(
            "A local ACE resource monitor is already running in another session."
        )

    monitor_scope = "CPU, memory, and disk"
    if getattr(config.proactive, "local_monitor_process_context_enabled", False):
        monitor_scope += "; top processes on new CPU/memory alerts"
    if getattr(config.proactive, "local_monitor_reasoning_enabled", False):
        monitor_scope += "; local analysis runs in chat or detached mode"
    console.print(
        f"[green]ACE local monitor active[/green] — sampling {monitor_scope} "
        f"every {interval} seconds. Press Ctrl+C to stop."
    )
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopping local monitor…[/dim]")
    finally:
        service.stop()


__all__ = ["monitor"]
