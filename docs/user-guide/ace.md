# A.C.E. — Adaptive Cognitive Entity

A.C.E. is the local-first personal assistant being developed on top of
OpenJarvis. Its profile brings together a user-selected local model, persistent
memory, internet research with local storage, host health monitoring, and scoped tools.
It can use screen context on demand, perform confirmed desktop actions in an
interactive session, and prepare reviewed code-improvement proposals. Screen
observation runs only when explicitly enabled for a foreground session.
Desktop actions require confirmation, and code changes remain proposals until
the user reviews and applies them.

This A.C.E. project is separate from OpenJarvis's optional Agentic Context
Engineering optimizer, documented in [Learning: ACE](../learning/ace.md).

## Start the profile

The ACE preset does not choose a model for the user. `jarvis model setup`
offers a small local model catalog, accepts other Ollama model tags, and asks
whether to download the selected model. On Windows, it can also install Ollama
through WinGet after confirmation if the runtime is missing. Pass
`--install-runtime` to opt in directly or `--no-install-runtime` to skip it.
Other platforms link to Ollama's official download page instead of running an
installer script when `--install-runtime` is requested; they do not install it
automatically from this command.

```bash
jarvis init --preset ace
jarvis model setup
jarvis chat
```

In the desktop app, open **Settings → Inference source**, choose Ollama, and
enter the exact tag you want as the startup model. On first setup, the consent
screen requires that tag before starting Ollama. On later changes, saving the
setting applies after restart. If the exact tag is not installed, ACE requests
only that tag from Ollama; it never substitutes a different installed or
fallback model. In **Settings → Models**, you can explicitly download a tag
ahead of time or choose one of the installed tags as the startup model. The
desktop app reads that list through its native bridge to Ollama; an unavailable
runtime is reported as an error rather than as an empty model list. Custom
OpenAI-compatible servers also require the user to enter the model ID.
For LM Studio, choose **OpenAI-compatible server**, select **LM Studio**,
and enter its loopback URL and the exact ID advertised by that server.
For example, a running LM Studio server can serve `qwen/qwen3.5-9b` at
`http://127.0.0.1:1234/v1`.

### Optional Cloudflare Workers AI chat

Choose **Cloudflare Workers AI** to use a text chat model online. The default model is Gemma 4 (`@cf/google/gemma-4-26b-a4b-it`). The backend sends the current text conversation to Cloudflare. Automatic local memory retrieval and local tools are available in the local chat modes.

On Windows, enter the account ID and API token in the desktop setup screen and select a model. The A.C.E. stores the credentials in Windows Credential Manager. Open **Settings → Inference source** to change the model or manage the saved credentials. [Windows setup for Cloudflare](../getting-started/ace-windows.md#cloudflare-workers-ai)

#### Command-line setup

To configure the loopback API from a terminal, set `CLOUDFLARE_ACCOUNT_ID`
and `CLOUDFLARE_API_TOKEN` in the environment that launches the server, then
run `jarvis serve --engine cloudflare` to use Gemma 4. Choose another
model available on your account by passing its exact ID, for example
`jarvis serve --engine cloudflare --model '@cf/qwen/qwen3.8-27b'`. You can
also set `provider = "cloudflare"` under `[intelligence]` for `jarvis serve`.

### Local model and embedding downloads

For CLI setup, the model download remains optional. If the user skips it, ACE
saves the selected model and prints a command to download it later.
`nomic-embed-text` is a separate optional embedding model used for semantic
memory search. Start Ollama and run `ollama pull nomic-embed-text` if semantic
embeddings are wanted; FTS5 memory search remains available without it. If
local setup is unclear, run `jarvis doctor` for runtime status and model-specific
next steps.

### Hardware and model choice

The machine currently specified for ACE is a Ryzen 5 5600GT, A520M board, and
16 GB system RAM. The 5600GT includes Radeon integrated graphics, but ACE does
not assume a dedicated graphics card or dedicated VRAM. Start with a small
quantized model and expect CPU inference to be slower than a dedicated-GPU
setup. The selected model and its context length affect memory use. Check
`ollama ps` while a model is loaded to see whether Ollama is using the CPU, GPU,
or both. See AMD's [Ryzen 5 5600GT specifications] and Ollama's [GPU status
guidance] for current details.

[Ryzen 5 5600GT specifications]: https://www.amd.com/en/support/downloads/drivers.html/processors/ryzen/ryzen-5000-series/amd-ryzen-5-5600gt.html
[GPU status guidance]: https://github.com/ollama/ollama/blob/main/docs/faq.mdx#how-can-i-tell-if-my-model-was-loaded-onto-the-gpu

If you already have `~/.openjarvis/config.toml`, `jarvis init` will stop
instead of replacing it. Back up that file before choosing `--force`, which
replaces the active config.

The ACE preset also creates `SOUL.md`, `MEMORY.md`, `USER.md`, and the local
`skills/` directory when they are missing. `SOUL.md` sets ACE's conversational
style and memory rules; `USER.md` is for stable profile details, while
`MEMORY.md` is a small editable summary. Existing files are preserved. The one
exception is the stock OpenJarvis persona: when `SOUL.md` exactly matches that
default, the ACE preset replaces it with the ACE persona. Edit these files to
personalize ACE without changing the preset config.

When upgrading an existing ACE config, add these keys under its `[memory]`
section to turn on semantic recall for both durable facts and conversation
episodes without replacing the config:

```toml
facts_semantic_enabled = true
facts_embedding_model = "nomic-embed-text"
episodes_semantic_enabled = true
episodes_embedding_model = "nomic-embed-text"
```

When loopback Ollama and the configured embedding model are available, new
durable facts are embedded in the background and ranked by a hybrid of lexical
overlap, local vector similarity, and recency.
The vectors live in a private SQLite sidecar beside `memory_facts.jsonl`; the
JSONL file remains the source of truth. To index facts saved before semantic
recall was enabled, run `jarvis memory facts reindex --limit 100` repeatedly
until the command reports that no facts need vectors. The query embedding and
fact vectors never go to a remote host. If local Ollama or the embedding model
is unavailable, chat uses lexical fact ranking; run the reindex command later
to fill in vectors that could not be created.

During `jarvis chat` and `jarvis serve`, the preset enables automatic
extraction of durable facts, important project decisions, and outcomes into a
local JSONL store and injects relevant memory into later context. It also keeps
the server bound to `127.0.0.1`,
disables external MCP servers, analytics, telemetry, and raw interaction
traces, and enables input/output security scanning. Secret and PII scanning,
capability checks, and request rate limiting retain Python fallbacks when the
optional native Rust extension is unavailable. The preset also retains
substantial completed exchanges in a separate local SQLite/FTS5 episode store
for query-based recall. ACE combines lexical FTS5 matching with semantic vector
matching from local Ollama embeddings. Install `nomic-embed-text` as shown
above; if Ollama or that model is unavailable, episode search falls back to
FTS5 and chat continues. Episode text is sent only to a loopback Ollama host
for embeddings. Episodes are retained for up to one year and capped at 5,000
records. To index older episodes after enabling semantic recall, run
`jarvis memory episodes reindex --limit 100` repeatedly until no candidates
remain.

ACE's core local stores—facts, episodes, indexed knowledge, the knowledge graph,
monitor history, pending approvals, agent/session state, scheduled tasks,
code-improvement proposals—plus user configuration, credential backups, the
encrypted vault, research attachments, editable profile/memory files, and
approved procedure snapshots—use restrictive file helpers. POSIX applies
owner-only modes. Windows sets a protected NTFS ACL granting access only to the
current user; it does not rely on `chmod(0o600)` to isolate accounts. If the ACL
cannot be set, creating or updating the data fails instead of silently treating
it as private.

`provider = "local"` is enforced by engine discovery: cloud engines are not
probed or added as fallback, even if cloud API keys exist in the environment.
HTTP inference endpoints must use loopback (`localhost` or a loopback IP); a
remote OpenAI-compatible host is rejected in local-only mode. In-process local
engines remain eligible. Local inference clients ignore HTTP proxy environment
variables and do not follow redirects, so a configured system proxy cannot
forward prompts away from a loopback runtime. ACE research and connector
sync use only loopback Ollama for embeddings in local-only mode and fall back
to lexical retrieval if it is unavailable. Agent ticks fail closed if no
verified local engine is available.

The ACE preset enables `engine.auto_start_local`. If the configured loopback
Ollama service is down when a local-only chat or server starts, ACE launches
the installed `ollama serve` executable without a shell and waits up to 12
seconds for the service. It sets `OLLAMA_HOST` to the verified loopback
endpoint, does not install Ollama or download models, and leaves the runtime
running after ACE exits. The generic OpenJarvis config keeps this option off;
set `auto_start_local = false` under `[engine]` to disable it for ACE. This
currently starts Ollama only; other local runtimes must be started separately.
Chat startup never installs software. On Windows, `jarvis model setup
--install-runtime` offers a separate, explicit WinGet-based Ollama installation.
During a session, ACE checks local runtime health after a failed request and can
restart the configured loopback Ollama and retry once before stream output has
been emitted. This also works when ACE started Ollama itself at the start of
the session. If you pass `--engine ollama`, ACE keeps that runtime pinned.
Without an explicit engine choice, it can retry on another local backend.
If a stream fails after sending partial output, ACE reports the error and
preserves the text already sent. A model error from a healthy service can
trigger model fallback without restarting the service.
The ACE preset starts with no default or fallback model. CLI chat, server
startup, one-shot questions, and proactive reasoning do not silently pick an
arbitrary installed model; choose one with `jarvis model setup`, use
`jarvis chat --pick-model`, or pass `-m MODEL` to `jarvis chat` or
`jarvis serve` for a one-time choice. After a model is selected, ACE rechecks
available local OpenJarvis engines after a runtime error and can retry through
another local backend using the selected model or a backup you explicitly
configured in `[intelligence].fallback_model`. A secondary backend can try
both models if the first fails before sending any output. All these attempts
stay on local runtimes. At least one local runtime must be available when
starting a chat or server session.

If every local runtime is unavailable, or no local model can run on the
machine, text generation cannot continue. Memory and local files remain on the
machine, but they do not replace an inference model. To use a runtime on
another computer, switch away from local-only mode; generation then depends on
that network service. Run `jarvis doctor` to see whether a second local
backend is available. When it finds supported runtime CLIs on `PATH`, it also
shows commands to start them if their local services are down. In local mode,
that diagnostic skips cloud probes and remote HTTP endpoints.

Voice is configured for local processing only: faster-whisper for speech
recognition and Kokoro for neural speech synthesis. On Windows, if Kokoro is
unavailable, ACE can use an installed system voice through the built-in SAPI
backend without a Python TTS package or network connection. It prefers an
installed Portuguese (Brazil) voice when available. In interactive CLI chat,
the SAPI fallback can play its WAV through Windows directly, without Python
audio playback packages. Kokoro playback and microphone input still require
the audio dependencies. If local speech recognition is unavailable, voice
input is disabled; text chat remains available. Install the local voice
dependencies with `pip install 'OpenJarvis[desktop,voice]'` when you want
microphone and Kokoro playback support. Run `jarvis doctor` to check local
speech recognition and TTS.
For Windows system-output capture, also install
`pip install 'OpenJarvis[audio-loopback]'`. `[speech].offline_only = true` also
filters hosted backends from interactive voice chat and the server's TTS
resolution. Faster-Whisper requires a complete local model bundle containing
`config.json`, `model.bin`, and `tokenizer.json`. Prepare it separately or
point `[speech].model` at an existing model directory, then run `jarvis doctor`.
The Media model cache check lists any missing files. If a microphone or speech
model is unavailable, `jarvis chat --voice` still accepts typed messages.

`jarvis chat --hands-free` is an explicit, foreground listening mode. The
microphone remains open until Ctrl+C. Local voice activity detection keeps
only a short pre-roll in memory, and local faster-whisper checks each spoken
phrase for a wake word from `[speech].wake_words`. Speech without a wake word
is not added to chat history or persistent memory, and is not sent to the
conversation model. Audio stays in memory during local transcription. Only a
request after the wake word enters chat. This uses
speech recognition to detect the phrase, so it is more compute-intensive than
a dedicated low-power wake-word chip. It requires
`[speech].offline_only = true` and an inference endpoint verified as local.

To recognize speech from computer playback instead, start
`jarvis chat --hands-free --listen-system-audio` on Windows and install the
optional `audio-loopback` extra. This captures audio from the default speaker
output while the command is active, including any app using that output. It
uses local faster-whisper and the same wake-word gate; unactivated speech does
not enter the conversation. This option is off by default and requires the
same local-only inference and offline speech settings.

The profile exposes `think`, `calculator`, memory management, local knowledge
search, web research, `system_health`, `system_security`, `process_snapshot`,
`local_monitor_history`, `kg_query`, and `kg_neighbors`. The health tool
reports basic OS, CPU, memory, disk, and local runtime executables on PATH
without reading files, starting programs, or connecting to the network. On
Windows it also reports display-adapter names exposed by SetupAPI. It does not
report dedicated VRAM or verify GPU inference; use `ollama ps` while a model is
loaded to inspect CPU/GPU allocation.
`system_security` runs only when explicitly requested. On Windows it reads
Microsoft Defender and Firewall profile status through local PowerShell
cmdlets; it does not scan files, change settings, or use the network. It is a
limited status check, may be unavailable, and does not establish that the
computer is secure or that another antivirus is absent. Ask before sending its
results to a remote inference endpoint.
`process_snapshot` runs only for a requested performance diagnosis. It returns
the top processes by CPU and resident memory with process name, PID, and
resource percentages; it omits command lines, executable paths, environment
variables, and file contents. It does not prove what caused an incident.
Automatic process context for new CPU and memory incidents is described below.
If you change inference to a remote endpoint, ask before sending this process
metadata to it. Enable host commands only for an interactive session with
`--allow-shell` on `jarvis chat` or `jarvis ask`; each
`shell_exec` call then requires your confirmation, and the flag is refused when
stdin or stdout is redirected. To grant one chat read-only access to files,
pass each workspace directory explicitly:

```bash
jarvis chat --workspace "A:\Projects\MyProject"
```

For a web task, `jarvis chat --allow-browser` or `jarvis ask --allow-browser`
adds OpenJarvis's isolated, headless Playwright browser to that agent request.
It does not use or control existing Chrome tabs, cookies, or logged-in
sessions. The SSRF guard checks navigation, redirects, page subrequests, and
WebSockets; service workers are disabled so they cannot bypass route checks.
Every navigation, click, typed input, and screenshot requires confirmation in
the interactive terminal.
For a click, ACE checks for a single visible match across the main page and
frames unless you specify a position such as the first video result. Use a
more precise selector if several elements match. If the browser cannot confirm
a click after trying it, inspect the page before retrying. ACE stops that call
without clicking a second target.
Page contents are untrusted. Messages, purchases, account changes, and
destructive actions still require an explicit request and confirmation. If
inference is not verified as local, ACE asks before page content can be sent to
that endpoint. Install the optional dependencies with
`uv sync --extra browser` (Playwright 1.48 or newer). The `--allow-browser`
flag cannot be combined with `--research`.

For visible desktop applications, `jarvis chat --allow-desktop` or
`jarvis ask --screen --allow-desktop "..."` enables coordinate-based clicks,
text entry, and a limited key/shortcut list. Chat attaches one temporary
primary-screen image per normal user turn; `ask` requires the explicit
`--screen` option. Both commands require an interactive terminal, an agent,
and inference verified to run locally. Every individual action displays its
arguments and waits for confirmation. The keyboard shortcut allowlist excludes
clipboard copy and paste. The controls do not read application internals or use
clipboard contents. Native desktop actions are currently supported on Windows.
Windows may block control of elevated applications or the secure desktop; ACE
does not bypass that boundary.

The `file_read` tool is then restricted to those directories and blocks known
credential files. `--workspace` alone does not enable file writes. If you also
pass `--allow-shell`, approved shell commands can change host files; use the
reviewed `jarvis improve` workflow for ACE's own source code. The separate
local resource monitor runs during `jarvis chat`; it does not inspect screens,
applications, processes, or file contents.

## Inspect and clear memory

```bash
jarvis memory list
jarvis memory forget 3
jarvis memory clear
jarvis memory episodes list
jarvis memory episodes show <episode-id>
jarvis memory episodes search "OpenJarvis local fallback"
jarvis memory episodes forget <episode-id>
jarvis memory episodes clear
jarvis memory episodes recover
```

`memory clear` asks for confirmation before removing extracted facts. Memory
is enabled by this preset, so durable facts may be stored automatically.
Review the list periodically and remove anything you do not want retained.
Use the 1-based index shown by `memory list` with `memory forget` to remove a
single fact; the command checks that the entry has not changed since review.
The command clears extracted facts only. `memory episodes clear` separately
removes saved conversation episodes. Episodes are captured only when an
exchange is substantial (or contains a decision, problem, solution, or outcome
cue), and both the local secret and input scanners are available and accept it.
This reduces retention of small talk and sensitive text but cannot guarantee
detection of every secret. Search combines local FTS5 matching and local vector
matching when the configured embedding model is available. Recalled excerpts
are marked as historical, untrusted context;
ACE does not treat earlier assistant messages as verified facts. The fact and
episode stores are separate from `~/.openjarvis/MEMORY.md`, the web-research
database, and the knowledge graph database. Review or edit `MEMORY.md`
independently; inspect and remove graph records in **Knowledge Graph**. These
stores remain local.
`memory episodes forget` also removes graph evidence and durable facts
attributed only to that episode, while preserving independent evidence from
documents, web research, or another tracked episode. `memory episodes clear`
selects the episodes and conversation evidence already present when the command
asks for confirmation. It removes facts linked only to the selected episodes.
Facts whose episode has already expired and facts with incomplete provenance
stay available for manual review with `memory list`. Both commands report
removed and retained facts.

If a storage error interrupts cleanup, run `jarvis memory episodes recover`
after resolving the error. The saved selection survives a process restart and
keeps conversation data captured afterward. Older pending clears without a
saved selection require a fresh `jarvis memory episodes clear` confirmation.
The stores commit separately, so some facts or graph evidence may already be
gone before recovery completes.
Do not use the memory feature to store passwords, API keys, recovery codes, or
financial account identifiers.

CLI chat and ask, tool agents, the server chat endpoint, the Python SDK, and
local incident reasoning retrieve documents, facts, episodes, procedure
outcomes, and graph evidence through one bounded context builder. Graph claims
remain labeled as sourced but unverified, and the graph receives a capped share
of the context budget. The databases remain separate; recall combines them at
request time. When a relevant graph relation links to a retained conversation,
ACE can add up to two linked episodes to that same context, ahead of ordinary
episode search results. Expired or deleted episode records are skipped.

## Save reusable procedures

When you explicitly ask ACE to remember a successful workflow, it can save a
small procedural skill in `~/.openjarvis/skills/`. The `skill_manage` tool asks
for interactive confirmation before each operation, including reads. ACE does
not turn conversations into executable routines on its own and must not save
credentials or unrelated conversation text. A saved routine becomes available
after restarting ACE; inspect it with `jarvis skill list` and `jarvis skill
info <name>`. Running it still uses the active tool allowlist and each tool's
normal permission checks. A skill is a reusable recipe, not an authorization
to carry out its steps without your request.

The ACE preset also enables review-only procedure proposals. In interactive
`jarvis chat`, if you explicitly confirm that a recent multi-step answer worked,
ACE may save that answer as a draft proposal. You can inspect it with
`jarvis memory procedures list` and `jarvis memory procedures show <id>`, then
reject it or approve it. Approval asks for confirmation and saves a
`SKILL.md` whose instructions say to use it only when you ask by name; approval
does not run the proposal. Restart chat to discover an approved skill, then ask
ACE to use it by name. The heuristic only uses the immediately preceding
captured episode in the current conversation, within 30 minutes, when the
answer has at least two numbered or bulleted steps. The web app sends its stable
conversation ID with each chat request so the server can connect a success
message to the earlier episode. Other API clients must send the same
`session_id` on consecutive turns to enable procedure proposals; without it,
the server still stores eligible episodes but cannot link a success signal to
the prior exchange. `jarvis memory procedures show <id>` also displays up to
five similar request titles from other conversations, with their review status,
success signal, and a short draft excerpt. These are lexical matches for manual
comparison; ACE does not merge them or treat them as verified procedures. If
you explicitly report that a recent multi-step answer failed within 30 minutes,
ACE keeps that outcome with its source episode. The report must share the same
conversation ID. Similar failure reports from other conversations appear in
the procedure review so you can compare what worked and what did not. These
results remain unverified user reports; ACE does not change an approved skill
automatically. When a later request has a similar title, ACE also adds matching
failure reports to the bounded memory context. The model is told these are
historical user reports, not instructions or proof that the current situation
is the same. Matching uses lexical title tokens and may miss differently
worded tasks.

Approved ACE procedure skills keep immutable version snapshots outside the
active skill directory. Updating one requires explicit proposal approval and
shows the full replacement first. The following commands list versions, create
a confirmed patch update, or restore an older body as a new patch version:

- List: jarvis memory procedures versions <skill-name>
- Update: jarvis memory procedures approve <proposal-id> --update-skill <skill-name>
- Restore: jarvis memory procedures rollback <skill-name> 0.1.0

These commands only manage ACE procedure skills. They do not run a skill or
change its existing tool permissions.

List saved failure reports with:

```bash
jarvis memory procedures outcomes
```

The output includes the source episode ID. Use `jarvis memory episodes forget
<episode-id>` to remove that report and its linked outcome.

## Change the local model

Run `jarvis model setup` to choose and optionally download a model for Ollama.
On Windows, pass `--install-runtime` to install the Ollama runtime with WinGet;
the interactive setup also offers this on Windows if a model download needs a
missing runtime. The installation remains optional and may show WinGet or
installer prompts.
For another local runtime, configure `[engine].default` and
`[intelligence].default_model` in `~/.openjarvis/config.toml`; `jarvis model
pull MODEL --engine ENGINE` can download models for supported runtimes. Verify
the choice with `jarvis doctor`.

## Research and the knowledge graph

In the desktop app, open **Web Research**. Enter a topic or public page URL
and select **Search and save on the web** to capture up to three sources.
Topic searches use You.com with DuckDuckGo as a fallback. A pasted page URL
is fetched directly. ACE stores the page text or search excerpt in the shared
local knowledge archive with its URL and capture date. Sources also appear
as entries in the knowledge graph.

Select **Search saved sources** to query this archive offline. You can search
by keywords or a saved URL. When live research fails and saved material
matches, the panel shows the original capture date. This panel works without
an inference model, including in the desktop app's local services mode.

For a saved page, select **Extract to graph** to ask the chosen local model
to find entities and relationships in the archived text. ACE uses the
selected capture without fetching the page again. In **Memory Graph**, the
extracted relationships are reported claims, with a supporting excerpt, the
source URL, and the date ACE captured the page. Search excerpts can still
be read in the archive. Extraction supports installed text models served by
Ollama, LM Studio, or llama.cpp on the same computer.

The preset enables `web_research`, which searches public web pages when the
user asks for research, fetches pages through OpenJarvis's redirect-aware SSRF
guard, and saves page text, URL, title, query, and content hash in the local
`knowledge.db` SQLite/FTS5 store. The built-in keyless search provider is used
when no search key is configured. Only the research query is sent to the
provider; saved page contents stay local.

CLI and web Deep Research use the selected planner model and inference engine,
or the explicit `[deep_research].engine` and `[deep_research].model` settings.
The CLI reuses the selected chat route when it can serve the planner model;
otherwise it resolves the configured Deep Research route and fails with an
actionable error if that exact engine/model pair is unavailable. The local-only
ACE profile does not substitute the legacy `gemma4:31b` planner when no model
has been chosen; select one with `jarvis model setup` or pass `-m MODEL` to the
CLI. If a planner model does not support tool calls, ACE performs the initial
local knowledge search itself and provides only those retrieved records for
synthesis; with no matching evidence, it returns a no-evidence response.
Citation numbers that do not map to retrieved records are labeled as unresolved
instead of being presented as source-backed references.

ACE asks a model to extract named entities and relations only when the selected
model's actual route is verified local; remote custom endpoints are skipped
even when they report themselves as non-cloud engines. Each accepted entity
name must occur in the saved text, and each relation must include a verbatim
quote found in that text. Claims are stored as `reported` evidence with source
URLs and confidence, not as verified facts. Repeated relations are deduplicated
while retaining distinct evidence. If local extraction fails, the page is
still saved and cited.

If a search provider fails or returns no usable results, `web_search` tries
DuckDuckGo. If live search still yields no usable source, `web_research` checks
the local store for previously saved pages on the same topic. A request
containing a URL also checks for that exact saved page first, even if its text
does not contain the URL. It returns the saved URL and timestamp and clearly
marks the result as unrefreshed historical material. If no matching local
pages exist, it reports that the request could not be completed; it does not
create an empty knowledge database just to attempt a fallback.

If the search engine finds a link but its page cannot be read, ACE can save
the search snippet when it has no better copy. It keeps any previously saved
page text and shows its original capture date. Earlier captures with no record
of whether they came from page text or a snippet are also preserved until a
fresh page can be fetched. The text and its search index update together.

The ACE preset also enables background graph capture from conversations. It
uses only the local model and only the user's own message; assistant-generated
text is not treated as evidence. Every saved relation must have a verbatim
quote from that message. The configured secret scanner can prevent capture of
a message containing detected credentials. This capture is best-effort and
does not delay the reply.

When you explicitly index a file or folder with `jarvis memory index` or the
Data Sources page, ACE also extracts quoted claims into the local knowledge
graph. This uses a local model and retains each document path and evidence
quote as provenance. Secret and prompt-injection scanners can reject unsafe
quotes. A single indexing request extracts from up to 12 documents and reads
up to 10,000 characters per document for graph capture; the rest of the indexed
chunks remain available through normal memory search. The indexing result
reports whether graph capture completed, was partial, or was skipped. Image
files are skipped by default. To index text from common image files or scanned
PDF pages, enable
**Use local OCR** in Data Sources or run `jarvis memory index PATH --ocr`.
This uses a locally installed Tesseract executable and its Portuguese or English
language data; OCR output stays local and is stored with the source path.
Scanned PDFs also need the local Poppler pdftoppm executable. The option is off
by default. Image OCR is limited to 20 MiB and 30 seconds per image. Scanned PDF
OCR is limited to 50 MiB, 20 image-only pages, 180 seconds per file, and 100,000
OCR characters. OCR text remains untrusted document content. Audio/video
transcription can be enabled separately with:

    jarvis memory index PATH --transcribe-media

This option uses Faster-Whisper and PyAV locally. It requires the selected
speech model to already be available in the local model cache. Install its
runtime with uv sync --extra speech and prepare the configured model cache
before indexing while offline. This indexing path never downloads a model or
sends media to a hosted API. Video indexing transcribes its audio track and
does not inspect video frames. Source media is limited to 50 MiB and 30 minutes
per file, with transcript output capped at 100,000 characters. The option is
off by default. Faster-Whisper uses a short-lived local temporary copy during
transcription and removes it after normal completion.

`jarvis doctor` checks whether Tesseract can find its installed language data,
whether Poppler's `pdftoppm` is available for scanned PDFs, whether the
Faster-Whisper and PyAV packages are installed, and whether the configured
Faster-Whisper model is present in its local cache. The model check uses an
offline cache lookup; it does not install packages, download models, or load the
model to verify device compatibility. It also reports whether the optional
microphone, speaker, Windows system-audio, and Kokoro packages are installed;
package checks do not open audio devices or record sound.

The ACE preset also enables a local resource monitor during `jarvis chat`.
Every two minutes it checks CPU, available memory, and free disk space, saves
new threshold incidents to `proactive_events.db`, and reports incidents and
recovery in the chat terminal. When local incident reasoning is enabled, ACE
saves its bounded assessment with the incident. Later analyses can consult up
to three earlier assessments as unverified hypotheses, and the
`local_monitor_history` tool can retrieve them. The preset takes one process
sample when a new CPU or memory incident starts. It stores the top three
process names and CPU/memory values without PIDs, paths, command lines, or file
contents. Set
`[proactive].local_monitor_process_context_enabled = false` to disable this.
The default resource-only monitor does not inspect screens or network services,
and it does not execute repairs. Ask ACE to use
`local_monitor_history` to review recent incidents. This monitor stops when
the chat session exits. To keep resource monitoring active without an
interactive chat, run `jarvis monitor --background`; it runs detached from the
terminal, stores incidents locally, and uses no inference API or network
connection in resource-only mode. Use `jarvis monitor --history` (or
`--history-limit 20`) to review
saved incidents and their latest local assessments. Run `jarvis monitor
--history --unacknowledged` for incidents awaiting review. Run `jarvis monitor
--acknowledge 42` with the ID shown in history to record your review. An active
incident stays active until measurements show recovery. If a reviewed warning
becomes critical, it returns to the review queue. Use `jarvis monitor --status`
and `jarvis monitor --stop` to manage the background monitor. It follows the same process-context
setting as the ACE profile. The
background monitor stops when the computer shuts down. Install opt-in login
startup to start monitoring again in your next user session:

```powershell
jarvis monitor --install-startup
jarvis monitor --startup-status
jarvis monitor --remove-startup
```

On Windows, the task runs as your current user with limited privileges. On
Linux, the command installs a per-user systemd service and requires a systemd
user manager. It starts with the user session and does not enable lingering to
run after logout; some WSL configurations do not provide the required manager.
On macOS, it installs a per-user LaunchAgent under `~/Library/LaunchAgents`.
These integrations do not require administrator privileges. Installing the
entry does not start monitoring immediately; run `jarvis monitor --background`
to start now. Removing the startup entry affects future logins; use
`jarvis monitor --stop` to stop a monitor that is already running.

For opt-in background screen observation, run
`jarvis monitor --background --monitor-screen`. It requires
`provider = "local"` and an available local model. The primary-screen images are
held in memory and sent only to that verified local model; ACE does not save the
images or perform actions. Short text assessments are written to the same
rotating local monitor log, so they may include text visible on screen. Review
the log path with `jarvis monitor --status`; screen assessments are capped at
256 KiB per run. The screen option is off unless it is included in the command
and is not enabled by `jarvis monitor --install-startup`.
Use `jarvis monitor --stop` to stop both resource and screen monitoring.
ACE records incidents without model analysis unless
`[proactive].local_monitor_reasoning_enabled = true` and the inference endpoint
is verified local. If a local runtime fails during analysis, the shared
resolver can retry through another healthy configured local backend. If
all local backends fail, ACE skips analysis and continues recording incidents.
For a known CPU, memory, or disk incident, the background worker records one
rule-based fallback with measured values and safe read-only checks while local
reasoning is enabled. It does not infer a cause or change the host; the incident
stays pending for model analysis when a local runtime returns. A remote provider
does not receive pending monitor data.
When analysis succeeds, bounded advice is written to the monitor log;
it is also saved in local incident history. When the evidence supports it, the
assessment may describe a reversible next step and how to undo it. The monitor
does not execute changes: you must request and confirm an action in the
interactive tool workflow.

The detached monitor can show category-only local desktop notifications for
new incidents and recovery when
`[proactive].local_monitor_notifications_enabled = true`. The ACE preset enables
this option; set it to `false` to disable notices. Notifications identify only
the affected resource category and whether it needs attention or recovered.
They do not include measurements, process names, file data, or model analysis,
and are delivered through the local operating system without network access.
If the operating system has no usable notification channel, ACE keeps the event
in its local history and monitor log.
Continuous monitoring requires `[proactive].local_monitor_enabled = true`.
Stop it with Ctrl+C. `jarvis monitor --once` takes a single sample, even if
continuous monitoring is disabled. CPU alerts require two high samples;
memory and disk alerts use one. Resolved records are retained for up to
90 days, with a 2,000-record cap. Repeated incidents report their local count
over the last 30 days. In interactive chat, when
`[proactive].local_monitor_reasoning_enabled = true` and the inference endpoint
is verified local, each new incident triggers a bounded model assessment using
relevant local memories and graph evidence. ACE prints measured facts, clearly
labeled hypotheses, and read-only checks without using tools or changing the
host. ACE saves the assessment for later inspection and reasoning. Inference is
serialized with normal chat requests. If local reasoning is
disabled, unavailable, or fails, the incident remains available for the next
chat turn. Pending monitor data is not automatically sent to a remote endpoint.
The foreground `jarvis monitor` command records and reports incidents without
model analysis; background analysis is available only with the explicit local
reasoning setting described above.

Resolved incidents can also be consolidated into episodic memory when both
[proactive].local_monitor_memory_capture_enabled and
[memory].episodes_enabled are true. This separate opt-in stores only aggregate
CPU, available-memory, or free-disk measurements and timing. Process names,
PIDs, paths, commands, and model assessments are excluded. Secret and injection
scanners must both pass; otherwise the incident stays pending and can be
retried. Consolidated episodes are searchable as historical sensor evidence,
not as a diagnosis or record of an action.

In the ACE desktop app, open **System Monitoring** to start or stop local
resource monitoring and view its recent incidents. It works with the local
services even when chat has no model. Desktop activation samples CPU load,
available memory, and free disk space. It stays running after you close ACE
until you stop it or sign out. The panel shows whether an incident is active
or recovered and displays any assessments already saved in its history.

Select **Show desktop notifications for resource alerts and recovery** before
starting monitoring to allow local
notifications for new incidents and recoveries. The choice defaults to off
for each new monitoring run. Stop and start monitoring to change it. An
alert shows the resource category and its state. The history panel shows
the measurements and offers **Mark as reviewed**. Reviewing an alert
records the review time. Recovery is recorded when the resource measurement
returns to normal, and a reviewed warning returns to the queue if it
becomes critical.

Dashboard monitoring also records a measurement assessment when a new alert
starts or an active warning becomes critical. These assessments work without a
model and offer read-only checks based on CPU, memory, or disk measurements.

Select **Analyze measurements** to review a saved incident, including a recovered
one. When a compatible local text model is selected, **Analyze with local model**
can add a separate suggestion. The model receives a snapshot of the aggregate
measurements, timing, and severity. If local inference is unavailable, ACE
returns the measurement assessment. Review and recovery remain independent of
either analysis. The Cloudflare chat choice supports measurement assessments.

Dashboard activation stores aggregate measurements. The existing episodic
memory settings determine whether resolved incidents become episodes.
The panel identifies monitors started from the terminal or at login so you
can manage those workers from the terminal. Use the dashboard or `jarvis monitor --stop` to stop a
worker started from the dashboard.

## Run previously approved routine tasks

The separate Proactive Agent can review connected personal sources on a daily
schedule. Its schedule is off by default. To opt in, set
`[proactive].enabled = true`, configure the source connectors you want it to
use, and start `jarvis scheduler start`. On Windows, the optional
`jarvis scheduler install-startup` command can start the scheduler at future
logins; inspect or remove that per-user task with
`jarvis scheduler startup-status` and `jarvis scheduler remove-startup`. ACE
uses the configured local model for the analysis; connectors and a configured
notification channel may still access external services. Stop it with
`jarvis scheduler stop`; it finishes its current task before exiting.

Before an action enters the queue, ACE checks that its message or event ID came
from the connector results for that run. Email IDs, SMS recipients, and calendar
event IDs are bound to those connector records; the model cannot set a
permission fingerprint. ACE derives remembered email rules from the sender
domain, SMS rules from the connector contact, and calendar rules from the
specific event. Proposals with missing or mismatched targets are discarded.

The agent can automatically produce the no-send `sms_draft_reply` summary or run
low/medium action patterns you explicitly remembered as `always yes`. A model
labeling an external action as trivial cannot bypass approval; high-tier actions
always require approval. Other proposed actions remain in the approval queue
until you approve or deny them. Configure `[proactive].notification_channel`
for outbound summaries and review pending approvals in the web app's **Agent
Approvals** panel. The panel expands each action payload by default. Selecting
**Approve & execute** records approval and immediately invokes the built-in
executor for that action type. A connector failure returns the item to the
pending list for review; unsupported action types are not reported as
successful. An action whose outcome cannot be confirmed appears in the panel's
status-check section, and ACE does not retry it automatically. Check the
connected service before trying again. **Deny** never calls the connector.
For supported email, SMS, and calendar actions, the panel also shows the exact
permission pattern that a remembered decision would match. **Approve & remember
pattern** both approves and executes the selected action, then saves that exact
allow rule in the same database transaction. It is offered only for low- and
medium-tier actions. **Deny & remember pattern** saves an exact deny rule,
including for high-tier actions. High-tier actions always require manual
approval; a remembered allow rule cannot run them automatically. This memory is
limited to the displayed pattern, and the panel does not let the agent create
or broaden the rule itself.

To approve from a channel, configure the exact destination and sender IDs:

```toml
[proactive]
notification_channel = "telegram:123456789"
approval_senders = ["123456789"]
```

The sender allowlist is empty by default. ACE processes approval text only
from a listed sender in the configured destination; replies from other users
cannot authorize actions. A reply such as `{action-id} yes` runs that one
approved action immediately, while `yes all` applies only to currently pending
actions. The acknowledgement reports status without echoing connector payloads
or error details. Keep the destination private and use stable sender IDs from
an adapter that authenticates inbound users; leave the allowlist empty for an
integration that cannot verify sender identity. The message must contain only
approval commands; quoted approval text in a longer sentence is ignored. The
ACE preset keeps this daily agent disabled until you opt in; its resource
monitor and local incident notices are separate features.
Recurring tasks that a model creates with the general `schedule_task` tool
also stay pending until you inspect their prompt, schedule, agent, and tools
with `jarvis scheduler approve <task-id>` and confirm in the CLI. The
displayed tool list is the exact scope used when the task runs; an empty list
means the task runs without tools.
When `[proactive].proactive_action_memory_capture_enabled = true` and
`[memory].episodes_enabled = true`, ACE saves a local episode after a built-in
proactive connector action runs. The record contains only its action category
and success or failure status; message content, recipients, targets, and
connector response text are excluded. Both local memory scanners must be
available and accept the record. These episodes are historical reports, not
independent verification of an external side effect. The daily proactive agent
can retrieve up to three matching action outcomes from this local history to
inform its next proposals. The history does not approve actions or bypass the
normal permission checks.
Each proactive run also retains the newest 1 MiB of diagnostics in
`logs/proactive_debug.log`. This private local log includes the collected
connector summary and raw model response, which may contain personal content.
It is separate from ACE memory and can be deleted whenever it is no longer
needed.

Research text and the graph are separate SQLite files configured by
`[tools.storage].knowledge_db_path` and `graph_db_path`. The graph is injected
as bounded context in both `jarvis ask` and interactive `jarvis chat`, and
`kg_query` / `kg_neighbors` expose inspection tools. Open **Knowledge Graph**
in the app to search and filter real stored nodes, switch between global and
selected-node views, and see a color legend for entity types. Inspect source
quotes, add or edit entities and relations, and delete graph records. Relation
editing lets you change its source and target entities, type, polarity, status,
validity dates, supporting quote, or source URL. If you change a relation's
source, target, type, or polarity, its
earlier evidence is retained as superseded history. Entity deletion also
removes connected relations and their evidence. In the graph view, drag an open
area to pan with a mouse or touch; use the arrow keys while the graph has focus
for keyboard panning. **Reset** restores the default zoom and position.
Parallel and reverse relations have separate curves. The arrow points to
the target entity. Select an edge, or focus it and press Enter or Space, to
inspect its evidence. **View relation details** moves focus to that card.
The **Change history** panel shows recent graph additions, edits, and deletions
in newest-first pages. Expand an event to inspect bounded before-and-after
snapshots; arbitrary entity properties are omitted, and evidence quotes are
limited in the history response. Deleting an entity or relation also removes
its earlier graph-history snapshots, then keeps only a minimal deletion record
with counts. Clearing the graph removes its earlier history. Entity properties
can be corrected as a JSON object in the selected entity's details; the editor
rejects invalid JSON and non-object values before saving.
Evidence links open only valid HTTP(S) URLs without embedded credentials; other
source values stay visible as plain text.
Relations from conversation capture also show up to 50 linked episode IDs when
the exchange passed episode-retention checks. Episodes remain in a separate
archive, so an ID can remain after its episode expires or is deleted. Inspect a
retained exchange with `jarvis memory episodes show <episode-id>`.

The graph view and contextual retrieval also surface possible conflicts when
the same entity has active relations with the same predicate but different
targets, or when evidence explicitly affirms and negates the same relation.
These are review candidates, not automatic contradiction judgments: claims
with different targets use a small allowlist of single-value predicates. ACE
keeps each claim and its evidence; review the quotes and sources before
changing a claim's status to `contradicted`. Relation polarity can also be
edited directly in the graph viewer.

Evidence can include an inclusive validity start and end date (`YYYY-MM-DD`).
The conversation and web extractors retain dates only when the exact ISO date
string is present in the supporting quote; you can also enter dates when
editing graph evidence. When competing claims have explicit, non-overlapping
date ranges, the graph keeps them as historical claims without flagging them
as a possible conflict. If either claim has no validity dates, its time scope
is unknown and the graph keeps the conflict candidate for review. Dates do not replace the
source quote or establish that the source is correct. Retrieved context labels
date ranges as historical or future relative to today's local date so the
model can distinguish them from evidence with an unspecified period.

The graph also shows an **Evidence review due** reminder for selected mutable
relations when their recorded evidence is older than its review interval.
Location and status claims use 30 days, version/model/deployment claims use 90,
default backend/model/provider claims use 180, and address claims use 365.
These intervals are review prompts based on the local SQLite record time, not
the source's `observed_at` or publication date. They do not prove a claim is
wrong or out of date, and they do not change its status. Review the saved quote
and source before correcting or superseding the claim.

Web research needs internet access at the time of the request. Saved material,
memory, and graph queries remain local and available offline. Extraction runs
only when a local model is configured and available.

### Inspect the primary screen

Open **Screen Analysis** to inspect the current primary display. Select
**Read screen text** to extract text with local OCR, including when chat has
no model. Windows uses an installed OCR language from the system or a local
Tesseract installation. Enter a question and select **Ask about screen text**
to use the recognized text with a verified local text model. For layout,
icons, and images, select **Capture and analyze** with a verified local
Ollama vision model.

If you selected Cloudflare Gemma 4 26B, enter a question, select **Analyze with
Cloudflare Gemma**, and confirm **Capture and send once**. ACE sends your
question and one image of the full primary screen to Cloudflare. The image and
result do not enter ACE's conversation history or long-term memory.

Use the three-second countdown to switch to the window
you want to inspect. **Cancel capture** stops the countdown. ACE keeps the
image in memory for the request and presents the result in the panel. Use
**Clear result** to remove it from the panel.

**System Monitoring** tracks CPU, available RAM, and free disk space on this
computer. Choose **Enable monitoring** to collect alerts even when the chat
provider is Cloudflare or no model is configured. **Live resource usage**
shows the current readings while the panel is open.

Use `jarvis ask --screen` to attach a single in-memory capture of the primary screen
to an agent or direct inference request. Select a vision model served by a
verified local engine. The command checks the model route before capturing.
For a text-only model, run `jarvis ask -m MODEL --screen-text "Read the text on screen"`.
ACE checks local OCR and the model route before capturing the primary screen.
It uses installed Tesseract language data or, on Windows, the built-in OCR
with an installed Windows language. The native path resizes large captures to
fit Windows OCR and was verified with synthetic PNGs up to 4K on Windows.
The image stays in memory, and up to
4,000 characters of extracted text go to the local model as evidence for that
one request. The mode uses direct inference without agents or tools and
keeps the capture and OCR text out of retained conversations. OCR reads text
and can miss or misread it. Use a vision model for visual layout or images.
Use `jarvis ask --image path.png` to attach a file you select. ACE warns if the
model route for an explicitly attached file may send it off the machine.
`--research` cannot currently be combined with image input. For a desktop task grounded in a screenshot, use
`jarvis ask --screen --allow-desktop "..."` or `jarvis chat --allow-desktop`.
Desktop control requires a verified local route for the selected model, and each
click, typing action, or key press requires interactive confirmation. After
each action, ACE captures the primary screen in memory and sends it to the same
local model to check the result before deciding what to do next. These images
are not saved in conversation history or long-term memory. If capture fails,
ACE tells the model that the action could not be verified. Shell commands can
also be enabled explicitly with `jarvis ask --allow-shell "..."`; each command
requires its own interactive confirmation. In an interactive session,
`jarvis chat --watch-screen` captures the primary screen in memory before each
normal user message. It verifies the selected model route is local for each turn
and sends the image only with that turn. Conversation history and long-term
memory contain text. Captures are triggered by normal user messages. Continuous
screen observation is available only when explicitly enabled for a foreground
session with `jarvis chat --monitor-screen`. It requires a verified local
route for the selected vision-capable model. In a mixed-engine pool, cloud and
unknown model routes are rejected before screen content is analyzed. ACE waits
20 seconds between capture cycles by default, plus time spent capturing and
analyzing. It compares
captures locally and sends changed images to the local model no more often
than once per 60 seconds.
Set `[proactive].local_screen_monitor_interval_seconds` and
`[proactive].local_screen_monitor_min_analysis_interval_seconds` to change
those intervals; supported ranges are 5–300 seconds and 15–900 seconds.
Captures stay in memory. Images and observations from the monitor are separate
from chat history and long-term memory. The observer has no tools and cannot act on the
desktop; visible screen content is untrusted input. It stops when chat exits.
Microphone listening outside explicit voice input and unattended desktop
control are not enabled by this option.

## Controlled improvement of ACE's own code

In interactive `jarvis chat`, ACE can also start a proposal with
`code_improvement_propose` after you explicitly request a code improvement.
Provide or confirm the repository path and source files. ACE asks before
reading them, uses the local-only inference resolver (including another
available local backend after a failure), validates the proposal in a
disposable copy, and returns its ID and diff. It cannot apply the patch from
chat; review and approve it separately with the CLI commands below. The tool
is unavailable to non-interactive agents that lack a confirmation callback.

The `jarvis improve` workflow lets a local model propose code changes without
giving it filesystem or shell tools. You choose which source files it may read.
The local credential-pattern scanner blocks selected source files or generated
diffs when they contain detected credentials. Proposals are limited to those
files and any related files you explicitly include through
`--discover-related`. ACE hashes the exact source file bytes read for the
model's context and refuses the proposal if those files change before it is
saved.

ACE saves the returned diff and the source hashes in the local
`~/.openjarvis/improvements.db` database; it does not change the active
checkout while generating or validating a proposal.

```bash
jarvis improve propose --task "Fix the parser edge case" \
  --file src/openjarvis/example.py \
  --discover-related
jarvis improve show <proposal-id>
jarvis improve validate <proposal-id>
jarvis improve apply <proposal-id>
```

Validation copies the current checkout to a temporary directory, applies the
diff there, and checks Python syntax. To also run pytest in Docker, first build
the local sandbox image:

```bash
docker build -f deploy/docker/Dockerfile.sandbox \
  -t openjarvis-sandbox:latest .
```

Then use `jarvis improve validate <proposal-id> --run-tests`; use repeated
`--test-path` options for selected Python tests. Add `--frontend` to run
`npm test` and the production build, or `--rust` to check Rust formatting and
run Rust crate tests offline in the same sandbox. `--runtime podman`
selects Podman, and `--image` can select another compatible image already
available locally. The test container has no network, a read-only root
filesystem, no Linux capabilities, and no host environment variables. It reads
the temporary copy through a read-only mount, then tests an ephemeral 2 GiB
`tmpfs` copy. The sandbox image contains Python development dependencies and
an offline npm cache from `frontend/package-lock.json`; dependencies newly
added by a proposal cannot be fetched during validation.

`jarvis improve apply` prints the saved validation result and exact diff before
asking for confirmation. Declining leaves the proposal unchanged. It rechecks
that every touched source file still matches the saved hashes before applying.
If a file changed since proposal time, ACE refuses and requires a fresh
proposal. Before modifying files, ACE records the original contents and modes
(up to 64 MiB per proposal) and the expected result of the diff in a local
SQLite journal. If an apply is interrupted, ACE checks every touched path. It
restores paths that match the expected result and leaves paths already matching
the original alone. If any path has a later edit or another unexpected change,
ACE reports the path and keeps the files and journal for review. An older
journal without the expected result also pauses if files have changed.
After recovery returns the proposal to `approved`, `jarvis improve apply <proposal-id>`
shows the diff and requests fresh confirmation. Recovery requires the
improvements database and source repository to be available. Reject proposals
with `jarvis improve reject <proposal-id>`.

In the ACE desktop app, open **Code Improvements → New code proposal** with a
supported local model selected. Enter the absolute path to the ACE Git checkout,
describe the change, and list the source files by repository-relative path, one
per line. **Generate proposal** saves a diff and checks it in a disposable copy.
The panel shows saved proposals, validation reports, and terminal commands
without a model or an internet connection. Applying and rolling back changes
require confirmation in the terminal.

ACE keeps the affected file originals in `improvements.db` after a successful
apply. Backups can use up to 64 MiB per proposal and stay in the local database
until the rollback completes.

```bash
jarvis improve rollback <proposal-id>
```

The rollback restores file contents and modes, recreates files removed by the
proposal, and removes files it added. ACE checks the entire applied result
before starting. If any affected file has changed, it leaves the checkout
untouched. Once confirmed, an interrupted rollback resumes when the improvement
manager next starts, provided every affected path matches a recorded file state.
Earlier applied proposals whose original files were discarded remain available
for review.

When using the Windows offline installer, the bundled Python can run these
commands from PowerShell. Substitute the folder where you installed ACE.

```powershell
& '<ACE install folder>\ace-backend\python.exe' -I -B -m openjarvis.cli improve list
```

Git must be available to create, validate, or apply proposals.
