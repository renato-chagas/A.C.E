# A.C.E. — Persistent Local Proactivity

A.C.E.'s resource monitor can run outside `chat` and `serve` as a detached,
per-user process. It samples CPU, available memory, and free disk space and
stores threshold incidents in the local `proactive_events.db`. When local
reasoning is enabled, it also saves bounded model assessments there and writes
notices to `proactive_monitor.log`.
Monitoring itself does not need an inference engine, network access, or an API
key. The detached worker can optionally show a local, category-only desktop
notice for each new incident and recovery; if no operating-system notification
channel is available, it still records the notice in its local log.

```bash
jarvis monitor --background
jarvis monitor --status
jarvis monitor --history --history-limit 20
jarvis monitor --history --unacknowledged
jarvis monitor --acknowledge 42
jarvis monitor --stop
jarvis monitor --install-startup
jarvis monitor --startup-status
jarvis monitor --remove-startup
```

Use the incident ID displayed in history to record that you reviewed it.
The review timestamp is stored in SQLite and repeated acknowledgements preserve
the original timestamp. The incident remains active until the monitor measures
recovery. An incident reviewed at warning severity returns to the review queue
if it becomes critical.

The process is detached from the launching terminal. Its state, control lock,
singleton lock, event database, and rotated logs are stored in the OpenJarvis
config directory with owner-only file permissions. PID state includes the
process creation time and a launch token. Status and stop verify both the
process identity and its command line before acting, so a recycled PID cannot
cause an unrelated process to be terminated. A cross-process file lease keeps
the background monitor from sampling at the same time as a monitor started by
chat or another foreground command.

The worker notices a stop request within about a second and releases its file
lease as it exits. Stop waits for a graceful exit, then terminates only the
verified worker if necessary. Stale state left by a crash or system shutdown is
discarded the next time the monitor is managed. On Windows, the user can
explicitly register a least-privilege Task Scheduler entry to start the
background monitor at the next login. This is opt-in and can be inspected or
removed with the commands above. Removing the login entry does not stop a
monitor that is already running. On Linux, the same commands manage a per-user
systemd unit in the XDG configuration directory. It starts with the user's
systemd session; the integration does not enable systemd lingering or keep the
monitor running after the final user session ends. This requires a working
systemd user manager, so Linux environments without one (including some WSL
setups) cannot install this startup entry. On macOS, the commands manage a
per-user LaunchAgent in `~/Library/LaunchAgents`; it launches the detached
monitor at login. Both integrations run as the current user and need no
administrator privileges.

When `[proactive].local_monitor_reasoning_enabled` is enabled, the worker may
build a local inference system to assess new incidents. It uses the shared
local-only engine resolver, including retries across other healthy local
backends. If the configured provider is remote, no local engine is available,
or all local retries fail, inference is skipped and monitoring continues.
If the worker starts while no local engine is available, it retries reasoner
setup every 60 seconds. For a known CPU, memory, or disk incident with local
reasoning opted in, ACE records a deterministic fallback containing only
whitelisted measurements and safe, read-only checks. It does not infer a cause,
inspect process names in this fallback, or change the host. That incident
remains pending for local-model analysis when inference returns. The fallback
is recorded at most once per incident. An incident whose model analysis fails
remains pending and is retried after a 60-second cooldown. Both the chat and
detached monitor save a successful model assessment and mark the incident
processed in one SQLite transaction. A failed write leaves the incident
available for retry. A model response for a resolved, already consumed, or
changed-severity incident leaves the current state intact. Recovery of a local
model requeues pending incidents
immediately. These retries do not start or install an inference runtime.
The reasoner only receives measured incident data plus bounded local memory and
graph context plus up to three earlier local assessments. Earlier assessments
are treated as unverified history. It has no tools and cannot change the host.
Successful advice is stored with the incident and written to the local monitor
log; `jarvis monitor --history` and `local_monitor_history` can retrieve it
later. Earlier assessments are bounded to three per incident in the history
view and are treated as hypotheses. When evidence supports one, an assessment
may describe a reversible next step, its expected effect, and how to undo it.
No action is executed; the user must request and confirm any change through the
normal interactive tool workflow. When enabled, the detached worker displays
generic local desktop notifications for new and resolved incidents; these do
not include measurements or process data. Automatic repair is not part of this
service.

The foreground `jarvis monitor` mode remains available and reports to its
terminal. `jarvis monitor --once` takes one sample without starting a worker.
Both share the same single-monitor lease, preventing duplicate samplers.

## Opt-in background screen observation

`jarvis monitor --background --monitor-screen` adds the existing
`LocalScreenMonitor` to the detached worker for that run. The screen path
requires `provider = "local"`, resolves the configured model through the shared
engine discovery path, and rejects non-local or unknown routes before capturing
the screen. Each captured image stays in process memory and is passed only to
that local model. The worker persists no image and exposes no tools to the
screen reasoner. Short assessments are written to the monitor's existing
rotating local log; the log may therefore contain text visible on screen. The
feature caps screen-assessment text at 256 KiB per worker run and is disabled
unless the command flag is provided. Login startup does not enable screen
observation, and stopping the shared worker stops both monitors. The existing
interval settings bound capture frequency and analysis frequency.


## Resolved incident memory

The separate local-monitor capture option consolidates resolved resource
incidents into the existing episodic SQLite store. Both
local_monitor_memory_capture_enabled and memory.episodes_enabled must be true.
The capture record contains only the metric category, severity, observation
count, whitelisted aggregate measurements, and timestamps. Process snapshots,
PIDs, paths, commands, and local-model assessments are not copied. The
synthetic record must pass both secret and injection scanners; missing or
flagged scanner results leave it pending for a later retry.

Each monitor event uses a stable episode key, so a crash between episode
persistence and marking the monitor row is recoverable without duplicate
episodes. Synchronous capture skips embeddings to keep host sampling local and
responsive; the episode remains available through FTS5 and can be recalled as
historical, unverified sensor evidence. Turning the option off leaves monitor
history intact and prevents new episode capture. No repair or other host action
is performed by consolidation.

## Approved connector action outcomes

When `proactive_action_memory_capture_enabled` and episodic memory are both
enabled, the connector execution path stores a local episode after a built-in
proactive action finishes. It records only the allowlisted action category and
whether it succeeded or returned for review. Connector payloads, descriptions,
targets, message bodies, and connector error text are excluded. Both local
secret and injection scanners must be available and accept the synthetic
record; capture failure does not change the already completed connector action.
The episode key is stable per action and outcome, so retries of the same failed
action do not duplicate that failure record, while a later success remains a
separate outcome. The record is historical evidence, not proof that an external
side effect occurred exactly as intended.

When both opt-ins remain enabled, the daily proactive agent searches this
source locally with FTS5 before proposing new actions. It retrieves at most
three entries and 1,800 characters, validates their exact generated format, and
rebuilds the prompt context from the allowlisted action type and reported
result. The prompt labels them as unverified history and says they cannot grant
permission or bypass approval. It does not recall connector messages or targets.

On 2026-11-18, the focused memory, proactive-action, monitor-consolidation,
and proactive-agent suite passed 49 tests. It verifies opt-in and scanner
gates, payload exclusion, stable deduplication across retries, and searchable
FTS5 retrieval. Connector execution is stubbed in these tests; they do not
verify that a live external service applied an action.

After adding proactive recall of those episodes, a focused run covering
episode retrieval, chat memory context, proactive prompts and approvals passed
92 tests on 2026-11-18. It includes source filtering and rejects stored entries
that do not match the generated outcome format. Agent and connector behavior
remain simulated; this does not evaluate decisions from a live model or execute
actions against connected services.

## Connector approval execution

The web approval panel now runs the selected built-in connector action only
after the user presses **Approve & execute**. Its payload is expanded before
the button so the user can review the exact target and arguments. Approval
uses a compare-and-swap status transition, then dispatches only the selected
action through `ExecutePendingActionsTool`; the same permission and executor
implementation is used by proactive runs. Denial never dispatches an action.

Successful actions move to `executed`. Connector failures return to `pending`
and the API omits the connector's raw error text. If ACE cannot determine
whether a connector completed, the action remains `approved` or `executing`
and is listed in the panel's status-check section. ACE does not retry this
state automatically, since a retry could repeat an external side effect. The
user should check the connected service before retrying. Expired pending and
approved actions are marked `expired`, and atomic approval/execution
transitions reject actions past their TTL. Channel replies are processed by
`JarvisSystem.wire_channel` only when both `[proactive].notification_channel`
and `[proactive].approval_senders` are configured. The sender must match the
allowlist, and the message must arrive from the exact configured destination.
The adapter must provide provider-authenticated sender IDs; an empty allowlist
disables channel approvals. Positive replies execute only selected pending
actions through the same connector dispatcher as the web panel. Denials only
change state. The parser expires stale items and uses atomic
pending-to-approved/denied transitions, so delayed or repeated replies cannot
revive expired or already decided actions. It requires the whole message to
contain only approval commands, separated by whitespace, commas, semicolons,
or newlines; an approval token embedded in ordinary text is ignored.

Tasks created by the general `schedule_task` agent tool are persisted as
`pending_approval` and excluded from the scheduler's due-task query. They can
only be activated by the CLI's explicit `approve` command after it displays
the prompt, schedule, agent, and complete tool scope. Agent-created schedules
carry an explicit tool scope: an empty list disables tools rather than
inheriting the system's configured set. Direct CLI schedules retain their
existing behavior. Older active tasks have no creator provenance and are not
automatically changed; users should review active schedules after upgrading.

The approval-tool and channel-wiring suite passed 38 tests after adding expiry,
repeat-decision, executor dispatch, and sender/destination checks. The approval
REST module's 34 tests were skipped in the current environment because FastAPI
is not installed; dependencies were not installed for this validation run.

The approval panel can now save a user's decision for supported built-in
email, SMS, and calendar action patterns. It displays the exact stored pattern
before showing the remember buttons. Remembered approval and pending-to-approved
status are committed together; the selected action then goes through the same
executor as one-time approval. Remembered denial and pending-to-denied status
are also atomic. The permission key comes from the queued action row, not the
request, and unsupported or mismatched patterns fail closed. High-tier actions
cannot save an allow rule, while remembering a denial remains available. The
focused approval-store and API tests cover exact-key persistence, high-tier
limits, and denial; REST coverage requires FastAPI to be installed.

On 2026-11-18, the proactive action suite passed 26 tests; the approval REST
module's 37 cases were skipped because FastAPI is absent from this environment.
Three frontend API tests passed for remembered approval, remembered denial, and
unchanged one-time approval requests. Frontend TypeScript checking and Ruff
passed. A follow-up ran the REST module with the existing global FastAPI and
the project's pytest under Python 3.14: all 37 route tests passed, with no
packages installed. The project's Python 3.13 virtual environment still lacks
FastAPI, so its normal test command continues to skip those cases.

The ACE capability regression batch on 2026-11-18 passed 527 tests and skipped
65 cases (plus 4 modules skipped during collection). It covered memory and
graph persistence, web research, local inference failover, proactive
monitoring, screen capture, desktop controls, code-improvement proposals,
conversational code-proposal generation, channel approval routing, and
configuration loading. Ruff passed for the
affected Python modules and tests, `git diff --check` passed, and TypeScript
type-checking passed for the approval UI. The batch uses mocked local engines,
channel connectors, and desktop APIs; it does not establish behavior against a
live model, a connected external account, or a real interactive desktop. The
approval API was verified separately in the focused 37-test run above.

## Focused validation and limits

Focused validation passed 22 tests covering local-only inference failover,
screen capture routing and cleanup, threshold-history retrieval, episode
consolidation, and confirmation-gated desktop controls. These tests use fake
engines and mocked desktop APIs; they do not verify an actual model outage,
Windows input delivery to a real application, or startup behavior across a
machine reboot. The background service still needs a deployment-level check on
each supported operating system.

After wiring background reasoning through the shared local-only resolver, a
focused run of the monitor-reasoning, local-failover, and engine-discovery
modules passed 28 tests. It verifies that the background reasoner delegates to
the resolver that owns runtime failover; it still uses mocked engines and does
not simulate a real local runtime outage.

Verified local HTTP inference clients now ignore proxy environment variables
in local-only mode for synchronous and asynchronous requests. Local episode
embeddings also disable environment proxies and redirects; failure still falls
back to lexical FTS5 retrieval. The focused engine-discovery, failover, Ollama,
embedding-connector, and CLI research-policy suite passed 86 tests with 1 skipped
checks that non-local profiles retain proxy behavior for remote Ollama hosts.
Local-only agent ticks now fail closed if the resolver finds no verified local
engine instead of falling back to the server's wrapped engine. The
agent-manager route tests could not run because FastAPI is unavailable in this
environment; its changed modules passed syntax parsing. These checks do not
verify every third-party local runtime or network stack on a deployed machine.

Category-only notification delivery is now an opt-in background-monitor feature;
the ACE preset enables it, while the generic configuration default remains off.
A focused run of notification and monitor-memory tests passed 9 tests covering
private payloads, local backend dispatch, config gating, and graceful fallback
when the OS notification utility is missing. The Windows API and native desktop
popups were mocked; a visible notification in an interactive Windows session
still needs deployment validation.

The detached monitor now retries local reasoner setup after runtime outages
and rate-limits incident retries after failed analysis. A focused 27-test run
covered reasoner recovery state, per-incident cooldowns, monitor history and
memory, notifications, startup management, and monitor CLI behavior. It uses
mocked engine discovery and does not exercise a live detached-worker outage.

A later Windows regression run passed 80 tests across the proactive modules,
monitor CLI, and chat CLI. It covered a simulated assessment write failure,
reopening the SQLite store, and retrying the same incident once. It also
covered rejecting a stale assessment after severity changes or recovery.

On 2026-11-18, a Windows smoke check ran `jarvis monitor --once` with an
isolated `OPENJARVIS_HOME` and memory/disk thresholds chosen to trigger alerts.
The command recorded both incidents, and a separate `jarvis monitor --history`
read them back from SQLite. This verified real host sampling and local
persistence without inference or network access; it did not exercise CPU's
two-sample threshold, background startup, or behavior across a reboot.

A detached Windows smoke check then exercised `jarvis monitor --background`,
`--status`, and `--stop` with an isolated config, reasoning, process context,
screen capture, and notifications disabled. It exposed the virtual-environment
Python launcher PID differing from the actual worker PID. After the worker
authenticates its launcher PID and private token, it now records its own PID
and creation time; status and stop still verify process identity before acting.
The start/status/stop cycle passed and the isolated monitor was confirmed
stopped. This verifies the detached lifecycle in the current Windows session,
not login startup or behavior across a reboot.

## Scheduled task login startup on Windows

The daily proactive scheduler can be registered as a per-user Windows Task
Scheduler task with `jarvis scheduler install-startup`. It starts the installed
environment's `pythonw.exe` at user logon with limited interactive-user
permissions and no console window. `startup-status` reports task state and its
last result; `remove-startup` removes future logon launches but leaves a
currently running scheduler alone. `jarvis scheduler stop` requests a graceful
shutdown after the active task completes. Installation is explicit and does not
activate pending agent-created tasks. A per-database OS process lock prevents a
manual scheduler or a second service from executing the same due task at the
same time. The login entry and lock behavior are covered by mocked Task
Scheduler tests and a cross-process lock test; behavior across a real logoff,
reboot, and Task Scheduler policy remains a deployment check.

## Scheduled run recovery

Before dispatch, the scheduler records the attempt in SQLite. After a task
returns, it updates the run outcome and advances the schedule in one
transaction. The polling daemon and the manual scheduler command use the same
database lock. Startup acquires that lock before marking unfinished attempts
as interrupted. Their results are unknown, so the tasks require review and no
new attempt is scheduled by default. Proactive cron registration preserves
these tasks even if its configuration changed.

Use the scheduler log to inspect the interruption. A confirmed resume of a
recurring task schedules its next future occurrence. If an interrupted
one-time task should run again, review any changes it may already have made
and create a new task. Synthetic tests cover interrupted dispatch, failed
finalization, concurrent claims, and schedule registration on restart.

## Local Ollama Runtime Recovery

The ACE preset enables `engine.auto_start_local`; generic profiles keep it off.
When the selected local-only Ollama endpoint is unhealthy, the resolver starts
the installed `ollama serve` executable with a loopback-only `OLLAMA_HOST`,
without shell execution or model downloads. During generation, the local
failover wrapper checks primary health after an error, waits for the configured
Ollama endpoint to recover, and retries the request once. Streaming recovery
runs startup checks outside the async event loop and is skipped after output
has been emitted; a later stream failure after partial output is surfaced
rather than replayed. Failed startup attempts are rate-limited before another
recovery attempt.

The focused local runtime, discovery, failover, Ollama, configuration, doctor,
and ACE preset suite passed 159 tests; Ruff and `git diff --check` passed. A
Windows smoke check started Ollama on a temporary loopback port, verified
`/api/tags`, stopped the service, and exercised a mid-session restart through
the real runtime health endpoint with generation mocked. Model inference,
model loading, and a natural local model outage remain unverified.

## Model-aware screen privacy

The foreground screen monitor now checks the selected model's actual route
through routing and security wrappers before starting. It accepts a known local
model in a mixed-engine pool and rejects cloud or unknown routes before screen
content is analyzed. The `ask` image and desktop gates and the `chat` screen,
microphone, and desktop gates check the selected model route. Chat performs its
check after the model picker. The focused screen-monitor, ask-vision, and chat
CLI suite passed 37 tests; its engines are simulated and it does not validate
live screen capture or real model inference.
