# Task Scheduler

The task scheduler lets you run agent queries automatically on a schedule -- once at a future time, on a recurring interval, or via a cron expression. Scheduled tasks are persisted in SQLite so they survive process restarts, and execution is handled by a background daemon thread that polls for due tasks every 60 seconds.

!!! note "Optional component"
    `SystemBuilder` attaches scheduled tasks to the live `JarvisSystem` when the scheduler is enabled. The CLI daemon `jarvis scheduler start` runs tasks in live mode by default; pass `--dry-run` to record due tasks without invoking a model or tools. The daemon stays in the foreground until interrupted.

---

## Schedule Types

| `schedule_type` | `schedule_value` format            | Example                     | Meaning                             |
|-----------------|------------------------------------|-----------------------------|-------------------------------------|
| `once`          | ISO 8601 UTC datetime              | `"2026-03-01T09:00:00Z"`    | Run once at that timestamp          |
| `interval`      | Seconds as a string                | `"3600"`                    | Run every hour, starting immediately |
| `cron`          | Standard 5-field cron expression   | `"0 9 * * 1-5"`             | 09:00 UTC, Monday–Friday            |

!!! tip "Cron support"
    Full cron expression support requires `croniter` (`uv pip install croniter`). Without it, the scheduler uses a minimal built-in parser that handles simple `minute hour * * *` patterns only.

---

## CLI Commands

The `jarvis scheduler` subcommand group manages tasks and the daemon from the terminal.

### Start the daemon

```bash
jarvis scheduler start
```

Starts the live background polling daemon. Due tasks run through the configured `JarvisSystem`, and task responses are saved to run logs. Use `jarvis scheduler start --dry-run` to check schedules without calling an agent. The daemon runs in the foreground until interrupted (++ctrl+c++). In production, run it under systemd or launchd (see [Deployment](../deployment/systemd.md)).

On Windows, login startup is opt-in and per-user:

```powershell
jarvis scheduler install-startup
jarvis scheduler startup-status
jarvis scheduler remove-startup
```

The login task runs the scheduler without opening a console, with the current
user's limited permissions. It starts at the next login; removing the task
affects future logins and does not stop a scheduler that is already running.
Only one scheduler process can run against a persistent scheduler database at a
time. A second process exits with an error instead of running due tasks twice.
Use `jarvis scheduler stop` to request a graceful stop. The scheduler finishes
its active task and then exits; remaining due tasks wait for the next run.

Tasks created directly with `jarvis scheduler create` are active because the
user explicitly invoked the command. Tasks created by an agent through
`schedule_task` are saved as `pending_approval`; the scheduler never runs them
until a user reviews and approves them.

### Create a task

```bash
# Run once at a specific time
jarvis scheduler create \
  --prompt "Generate the weekly summary report" \
  --type once \
  --value "2026-03-01T09:00:00Z"

# Run every hour
jarvis scheduler create \
  --prompt "Check for new emails and summarize" \
  --type interval \
  --value "3600" \
  --agent orchestrator \
  --tools retrieval,think

# Run on a cron schedule
jarvis scheduler create \
  --prompt "Summarize overnight logs" \
  --type cron \
  --value "0 8 * * 1-5"
```

### List tasks

```bash
# All tasks
jarvis scheduler list

# Active tasks only
jarvis scheduler list --status active

# Paused tasks
jarvis scheduler list --status paused

# Agent-created tasks waiting for review
jarvis scheduler list --status pending_approval
```

Example output:

```
ID               AGENT     TYPE       VALUE          STATUS   NEXT RUN
a3f9b12c4d8e    simple    cron       0 8 * * 1-5    active   2026-02-26T08:00:00+00:00
b7c2e56f1a3d    orchestr  interval   3600           active   2026-02-25T14:05:00+00:00
```

### Pause and resume tasks

```bash
# Pause a running task
jarvis scheduler pause a3f9b12c4d8e

# Resume -- next_run is recomputed from the current time
jarvis scheduler resume a3f9b12c4d8e
```

`resume` can reactivate tasks the user previously paused. If a scheduler
stops before it saves an outcome, the next scheduler startup marks that run
`interrupted` and puts the task in `review_required`. Check
`jarvis scheduler logs <task-id>` for the uncertain run. For a recurring task,
`jarvis scheduler resume <task-id>` asks for confirmation and schedules the next
future occurrence. For a one-time task, review whether it took effect and
create a fresh task if you want to run it again.

Reviewing an interrupted run is separate from approving a new agent-created
task. To approve that task, inspect its exact prompt and tools first:

```bash
jarvis scheduler approve <task-id>
```

The command displays the prompt, schedule, agent, and tool list, then asks for
confirmation. Declining leaves the task pending. The model has no scheduler
approval tool, so it cannot approve its own task. For agent-created tasks, the
displayed tool list is the complete allowed scope; an empty list means no tools
and does not inherit the configured defaults. User-created CLI schedules keep
their existing default-tool behavior.

Tasks already active before this approval gate was added keep their state,
because older records do not identify how they were created. Review active
entries with `jarvis scheduler list --status active` and pause or cancel any
schedule you do not recognize.

### Cancel a task

```bash
# Permanently cancel (status -> "cancelled", next_run cleared)
jarvis scheduler cancel a3f9b12c4d8e
```

### View run logs

```bash
# Last 10 executions for a task
jarvis scheduler logs a3f9b12c4d8e
```

Example output:

```
Run 1: started=2026-02-25T08:00:01Z finished=2026-02-25T08:00:04Z success=True
  Result: Overnight logs contain 3 warnings and no errors.
Run 2: started=2026-02-24T08:00:00Z finished=2026-02-24T08:00:05Z success=True
  Result: Logs are clean.
```

---

## Python API

```python title="scheduler_example.py"
from openjarvis.scheduler.store import SchedulerStore
from openjarvis.scheduler.scheduler import TaskScheduler

# Set up storage
store = SchedulerStore(db_path="~/.openjarvis/scheduler.db")  # (1)!

# Wire in a JarvisSystem for task execution
from openjarvis import Jarvis
jarvis = Jarvis()

scheduler = TaskScheduler(
    store=store,
    system=jarvis,         # (2)!
    poll_interval=60,      # (3)!
)

# Create tasks
daily_summary = scheduler.create_task(
    prompt="Summarize the latest news headlines",
    schedule_type="cron",
    schedule_value="0 8 * * *",
    agent="simple",
)
print(f"Created task {daily_summary.id}, next run: {daily_summary.next_run}")

# List active tasks
for task in scheduler.list_tasks(status="active"):
    print(f"  {task.id}: {task.prompt} @ {task.next_run}")

# Manage task state
scheduler.pause_task(daily_summary.id)
scheduler.resume_task(daily_summary.id)   # next_run recomputed from now
scheduler.cancel_task(daily_summary.id)  # permanent

# Start the background thread
scheduler.start()   # (4)!

# ... application runs ...

scheduler.stop()
jarvis.close()
```

1. SQLite database storing all task state and run logs.
2. The scheduler calls `system.ask(task.prompt, agent=task.agent, tools=...)` when a task is due. Pass `system=None` for a dry-run mode that logs what it would execute without calling the agent.
3. Seconds between polling cycles. Lower values increase responsiveness at the cost of more SQLite reads.
4. Starts a daemon thread named `"jarvis-scheduler"`. Daemon threads exit automatically when the main process exits.

---

## ScheduledTask Fields

Every task is represented as a `ScheduledTask` dataclass.

| Field            | Type              | Default       | Description                                       |
|------------------|-------------------|---------------|---------------------------------------------------|
| `id`             | `str`             | auto (16 hex) | Unique task identifier                            |
| `prompt`         | `str`             | --            | Query sent to the agent on execution              |
| `schedule_type`  | `str`             | --            | `"cron"`, `"interval"`, or `"once"`              |
| `schedule_value` | `str`             | --            | Cron expression, interval seconds, or ISO datetime|
| `context_mode`   | `str`             | `"isolated"`  | Execution context mode                            |
| `status`         | `str`             | `"active"`    | `"active"`, `"pending_approval"`, `"paused"`, `"completed"`, `"cancelled"` |
| `next_run`       | `str` or `None`   | computed      | ISO 8601 UTC datetime of the next execution       |
| `last_run`       | `str` or `None`   | `None`        | ISO 8601 UTC datetime of the last execution       |
| `agent`          | `str`             | `"simple"`    | Agent registry key to use for execution           |
| `tools`          | `str`             | `""`          | Comma-separated tool names for the agent          |
| `metadata`       | `dict`            | `{}`          | Arbitrary metadata for the task                   |

---

## Using Scheduler Tools with Agents

The five scheduler MCP tools (`schedule_task`, `list_scheduled_tasks`, `pause_scheduled_task`, `resume_scheduled_task`, `cancel_scheduled_task`) can be passed to any `ToolUsingAgent`, allowing an agent to propose follow-up tasks. Tasks it creates remain pending until a user reviews and approves them in the CLI.

```bash
# Let the orchestrator schedule its own follow-up
jarvis ask --agent orchestrator \
  --tools schedule_task,list_scheduled_tasks \
  "Research transformer architectures and schedule a daily summary at 8am"
```

```python
from openjarvis import Jarvis

j = Jarvis()
response = j.ask(
    "Schedule a weekly digest of research papers every Monday at 9am",
    agent="orchestrator",
    tools=["schedule_task"],
)
print(response)
```

See [Scheduler Tools](tools.md#scheduler-tools) for full parameter reference.

---

## Configuration

Scheduler settings live in the `[scheduler]` section of `~/.openjarvis/config.toml`.

```toml title="~/.openjarvis/config.toml"
[scheduler]
enabled = false
db_path = "~/.openjarvis/scheduler.db"
poll_interval = 60
default_agent = "simple"
```

| Key              | Type   | Default                          | Description                                |
|------------------|--------|----------------------------------|--------------------------------------------|
| `enabled`        | `bool` | `false`                          | Construct the scheduler in `SystemBuilder`; start it with `jarvis scheduler start` |
| `db_path`        | `str`  | `~/.openjarvis/scheduler.db`     | SQLite database path                       |
| `poll_interval`  | `int`  | `60`                             | Seconds between polling cycles             |
| `default_agent`  | `str`  | `"simple"`                       | Default agent for tasks that omit `agent`  |

---

## See Also

- [Scheduler Tools reference](tools.md#scheduler-tools) -- MCP tool parameter details
- [Architecture: Agentic Logic](../architecture/agents.md) -- how the scheduler integrates with agents
- [Deployment: systemd](../deployment/systemd.md) -- running the scheduler as a system service
