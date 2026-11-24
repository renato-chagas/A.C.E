# A.C.E. Memory Layers

A.C.E. keeps several local stores because durable facts, original conversation
episodes, research documents, and graph claims have different retention and
trust rules. They complement one another; the episode archive does not replace
the fact store, document store, or knowledge graph.

```text
completed exchange
  ├─ local extraction -> memory_facts.jsonl (short durable facts)
  ├─ quoted relations -> knowledge_graph.db (entities, claims, evidence)
  └─ substantial exchange -> episodes.db (original user/assistant text)
       └─ explicit success + structured steps -> review-only procedure proposal

query -> FTS5/vector documents + FTS5/vector episode recall
      + lexical/local-vector durable facts
      + procedure outcomes + bounded graph evidence
      -> shared context selection and budget -> model context

explicitly approved proposal -> ~/.openjarvis/skills/*/SKILL.md (manual skill)
```

## Resume a named CLI chat

Run `jarvis chat --session "Project planning"` to start or resume a named chat.
The CLI restores completed user and assistant text using the current local model,
persona, and permissions. Open other chats with a different name. Use
`jarvis chat --list-sessions` to see names, or
`jarvis chat --forget-session "Project planning"` to remove one from storage.
Both management commands work with inference turned off. In a named chat,
`/clear` resets that chat's saved text and starts a new session ID.

Open **Shared chats** in the desktop and select the same name to resume the
conversation. Both interfaces use the same `working_chats.db` file when they
use the same OpenJarvis data directory. The desktop can read, create, and delete
shared chats while running the local services without a model. Sending text
uses the server's configured local model and waits for a complete reply.

The store retains up to 12 completed exchanges, with at most 3,000 characters
per message and 8,000 characters of total text per chat. Both interfaces check
for secrets before saving. The desktop sends saved history only to a verified
local inference route. A saved reply can describe content from a screen or
voice request. The store holds text and version information. When two interfaces
open the same name, the session ID and revision protect the most recent write.
If a reply arrives after the chat changed, the desktop displays it as unsaved
and offers to reload. Long-term facts, episodes, and graph records are managed
in their respective stores.

## Back up or move local memory

In PowerShell, use a new folder for the backup and another new folder for the
recovered profile. Both parent directories must already exist.

```powershell
jarvis memory backup "$env:USERPROFILE\ACE-memory-backup"
jarvis memory restore "$env:USERPROFILE\ACE-memory-backup" --into "$env:USERPROFILE\.openjarvis-recovered"
$env:OPENJARVIS_HOME = "$env:USERPROFILE\.openjarvis-recovered"
jarvis memory list
```

The bundle includes available local facts, episodes, graph claims and evidence,
indexed SQLite text, research chunks, and named CLI chats. SQLite snapshots retain
committed WAL changes. Each store is copied consistently on its own. Finish
active memory writes before starting a backup that needs all stores to represent
the same moment. The manifest is written after the copies finish. Restore
verifies the size and SHA-256 of each member and checks the SQLite copies before
writing the new profile's configuration. Both commands work without inference.

Restore only bundles you control. The destination is a new local profile with
the saved retention limits. Existing profiles are protected against overwrite.
The bundle excludes credential files, the old configuration, skills, original
indexed attachments, and desktop browser history. Text in the stores can still
contain personal information. The backup has owner-only permissions and is
unencrypted. If your shell defines `OPENJARVIS_CONFIG`, clear it for the restored
session so it uses its own local `config.toml`.

## Correct a durable fact

Copy the existing wording from `jarvis memory list` and run
`jarvis memory facts revise "old wording" "corrected wording"`. Both local
scanners must accept the corrected text before it is stored. The fact store
writes the correction and marks the previous wording as superseded in one
atomic update. A superseded fact remains visible in the list but is excluded
from text and vector recall. If automatic capture sees the same wording again,
the previous fact stays superseded while its record is retained. An explicit
revision can restore it. The same local scanner interface now exposes the
secret scanner result to episode and document capture checks.

The correction is user-approved and independent of any episode that supplied
the original wording. Dialogue and graph evidence preserve their own
provenance and remain independently reviewable. The fact cap still applies.
If there is no room for a new correction, remove an entry or raise the limit.

## Episode capture and retrieval

`EpisodeStore` uses a local SQLite database with an FTS5 index and an optional
vector table. When semantic recall is enabled, ACE asks the configured Ollama
embedding model on loopback for vectors, then combines FTS5 and cosine-ranked
vector candidates with reciprocal-rank fusion. It never sends episode text to
a remote embedding host: a non-loopback `OLLAMA_HOST` disables semantic recall,
and local embedding requests ignore proxy environment variables and redirects.
If Ollama or the model is unavailable, search continues with FTS5 alone.

Durable facts remain in the inspectable JSONL store. When
`facts_semantic_enabled` is set, a separate SQLite sidecar stores only
model-versioned vectors keyed by a hash of each recallable fact. New clean
facts are indexed by the existing background memory worker. Query-time
selection fuses lexical and semantic ranks, with recency as a tie-breaker.
Embeddings are requested only from loopback Ollama; failures preserve lexical
recall. Existing facts can be indexed in bounded batches with
`jarvis memory facts reindex --limit 100`. Forgetting or quarantining facts
removes their vectors on the next context retrieval, and the CLI removes
vectors immediately when facts are forgotten or cleared.

`jarvis ask` and the Python SDK write eligible, scanner-approved local text
exchanges to the episode archive once the reply completes. The SDK requires an
explicit terminal status for streaming capture. `--no-context` in the CLI and
`context=False` in the SDK disable lookup and capture for that request.

### Derive facts and relations from one-shot requests

The archive queues extraction for new CLI and SDK episodes in the same SQLite
transaction as the transcript. Local JSONL fact memory enables the fact task.
`[memory].graph_capture_enabled` enables a separate task for claims supported
by a quote from the user's message. The request waits only for the archive write.
`jarvis chat` and `jarvis serve` use their memory worker to advance the queue
when no live exchange is waiting.

If you only use one-shot requests, run
`jarvis memory episodes derive --limit 5` to process up to five due tasks
using the configured local extraction model. Use `--model` to choose one
explicitly. One episode may have two tasks. A temporary failure reschedules only
that task, and a new process can reclaim an interrupted task after its lease
expires. A successful empty extraction completes the task.

Before extraction, ACE verifies the configured model's local route and scans
the retained dialogue again. Generated facts and graph claims link back to the
episode ID. If deletion starts while the model is generating, the worker
discards the result. Deleting or pruning the episode removes its pending tasks.
The queue is part of the episode database and travels with its backup.

The memory service skips short exchanges unless they contain a decision, problem,
solution, or outcome cue. Other exchanges are retained only when both the user
message and assistant response are substantial. Each transcript is capped at
12,000 characters per side. Retention is bounded by age and total record count.

Because episodes contain original dialogue, capture requires both the local
secret scanner and injection scanner to be present and to accept the exchange.
Scanner failure disables capture for that exchange. Search results are labeled
as historical, untrusted text; earlier assistant responses are not treated as
facts or instructions. Retrieval returns at most three matching episodes by
default and reserves part of the context budget for facts and document results.

The ACE preset stores at most 5,000 episodes for 365 days in
`~/.openjarvis/episodes.db` and enables local semantic recall with
`nomic-embed-text`. Install the model with `ollama pull nomic-embed-text`.
Episodes captured before semantic indexing can be embedded in batches with
`jarvis memory episodes reindex --limit 100`; repeat until the command reports
zero remaining candidates. Users can inspect and search with
`jarvis memory episodes list` and `jarvis memory episodes search`; `jarvis
memory episodes show <episode-id>` displays the full exchange. Delete one with
`jarvis memory episodes forget <episode-id>` or clear the archive after
confirmation. `jarvis memory clear` continues to operate only on extracted
facts.

## Episode links on graph claims

When an exchange passes the episode capture checks and is stored, its stable
episode ID is attached to entity evidence and to each relation evidence item
extracted from that same user's message. The graph keeps up to 50 distinct
episode IDs per relation, and the Knowledge Graph view displays them alongside
the quoted evidence. A relation can still be captured when the exchange is too
short or unsafe to retain as an episode;
in that case it has no episode link. The assistant response is retained only
inside an eligible episode and is never used as graph evidence. Episode IDs
identify related historical context; they do not verify the graph claim.
Select the episode ID beside a quote in Knowledge Graph to open the saved
exchange. The dialog displays the conversation as historical text. You can also
inspect it with `jarvis memory episodes show <episode-id>`.

The graph's `GET /v1/memory/episodes/{episode_id}` route reads a single record in
SQLite read-only mode. It honors the age and count limits without pruning the
archive. The route requires a direct loopback connection with a local host.
When a browser sends an origin, that origin must be local. Any configured server
API key still applies. Responses
containing conversation text use `Cache-Control: no-store`. The episode may be
unavailable if it expired, was deleted, or episode memory is disabled. Tests
cover local and remote requests, retention, and reading an existing archive
without deleting records.

Automatic graph recall follows evidence-specific episode IDs for claims that
fit the graph context limit. Contradicted and superseded quotes do not
supply supporting episode links. Relation-level IDs are used for older
conversation records that have no evidence-specific links.

Episodes and graph records retain separate databases. Forgetting one episode
removes evidence attributed to that episode and its old graph-history
snapshots. A relation with evidence from another episode, a document, or the
web is retained with that independent evidence. Clearing the episode archive
removes all conversation-derived graph evidence while preserving independently
sourced claims. Older graph records that predate per-evidence episode IDs are
cleaned conservatively when their relation links the forgotten episode. The
clear command also removes leftover conversation graph data when the episode
archive is already empty, after confirmation.

The fact, graph, and episode stores commit separately. Before purging related
stores, the CLI saves one episode ID in `episode_deletion_journal` or a confirmed
archive selection in `episode_clear_journal`. A clear stores the episode IDs,
graph episode links, and conversation source IDs selected before confirmation.
Facts are purged only by IDs present in the selected episode archive. The
episode deletes and archive journal update commit in the same transaction after
the derived memory cleanup succeeds. Individual deletion intents remain
independent from the archive selection.

`jarvis memory episodes recover` repeats the saved cleanup after an error or
process exit. An old unscoped clear requires a new confirmation. Graph history
for affected entities and relations is scrubbed because its snapshots may still
contain evidence that was removed.

Deleting an episode also removes its FTS entry and cascades to its local
vector, procedure proposal, and procedure outcome records. Durable facts now
carry bounded episode source IDs when extraction came from a retained exchange.
Forgetting an episode removes a fact only when every tracked origin is that
episode; facts with another tracked origin stay available. Clearing the episode
archive removes facts whose complete set of origins belongs to the selected
episodes. Facts with source IDs from expired episodes are kept for review. Legacy
facts and extractions from exchanges that were not retained as episodes have
incomplete provenance, so they are kept and the CLI warns that manual review may
be needed. Fact-vector sidecars are pruned after the authoritative JSONL update.

The `relation_evidence` table stores the originating episode ID as part of its
uniqueness key, so the same quote from separate conversations remains separately
attributable. Startup migration adds this column to existing graph databases.
On 2026-11-18, the focused episode, graph, memory CLI, and conversation graph
suites passed 65 tests; the Knowledge Graph history UI passed 4 tests and TypeScript no-emit
checking. Ruff checks passed. The API history sanitizer passed a direct local
TestClient smoke check; its pytest module remains skipped in the project
environment because FastAPI is not installed there.
After adding episode provenance to durable facts, the focused fact-store,
memory-service, and memory CLI suites passed 85 tests; Ruff check, formatting,
and diff checks passed.

Persistent ACE stores are created through `secure_create` and atomic text writes
use `secure_write_text`. POSIX applies `0700` to directories and `0600` to
files. Windows replaces inherited DACLs with a protected ACL for the current
user SID; chmod mode bits alone do not provide account isolation on NTFS. The
fact JSONL file and its cross-process lock, document/graph/episode databases,
proactive monitor, approval, and code-improvement databases use this path.
Managed-agent, session, and scheduled-task databases use it as well. SQLite
sidecars inherit the restricted parent-directory ACL.

During shared context retrieval, ACE follows up to two episode IDs attached to
the graph relations relevant to the current query. These episodes are placed
ahead of ordinary lexical/vector episode matches, duplicate IDs are included
once, and the common context budget still applies. Missing or expired IDs are
skipped; disabling episodic memory also disables this cross-store lookup.

## Procedural memory and review proposals

The ACE profile exposes OpenJarvis's `skill_manage` tool for user-directed
procedural learning. When the user asks ACE to preserve a successful routine,
ACE can store a small reusable workflow in the local skills directory. The
tool requires interactive confirmation, validates names and step targets,
refuses to overwrite an existing skill, and writes owner-only TOML files.
Discovered skills are offered to the agent on the next start. A stored skill
does not grant additional permissions: every step still passes through the
current enabled-tool list, capability policy, and any per-tool confirmation.

When `procedure_proposals_enabled` is true, the interactive CLI assigns a
random session ID to each unnamed chat. Named chats reuse their stored ID,
and `/clear` starts a fresh ID in either mode. The web
app sends its stable conversation ID with each chat request. After the user
gives a direct success signal such as “funcionou” or “that worked”, ACE checks
the immediately preceding captured episode from that session. If it is less
than 30 minutes old and the assistant response contains at least two numbered
or bulleted steps, ACE stores a review-only proposal in `episodes.db`. The
proposal copies only its numbered or bulleted list as an untrusted draft; no
model rewrites it, no tools run, and no skill is created automatically. Other
API clients must supply the same `session_id` on consecutive turns or those
exchanges cannot be linked for proposal consolidation.

Review proposals with `jarvis memory procedures list` and
`jarvis memory procedures show <proposal-id>`. `reject` records a declined
proposal. `approve` displays the draft again, asks for interactive confirmation,
and writes an owner-only `SKILL.md` under the configured skills directory. The
approved skill is visible to the model after restarting ACE, and its description
and instructions say to use it only when the user asks for it by name. Approval
does not execute it at that moment or verify that the historical procedure is
correct. It also does not bypass current tool allowlists or confirmation prompts.
Deleting the source episode also removes its linked proposal. A skill already
approved by the user remains as a separate saved skill.

`procedures show` also compares the request title with proposals from other
sessions. It shows up to five lexical matches, their review status, success
signals, and short draft excerpts. `jarvis memory procedures outcomes` lists
saved failure reports with their source episode ID so the user can inspect or
delete the source record. When the user reports an unambiguous failure
within 30 minutes of a recent structured answer, ACE links that result to the
answer's episode. `procedures show` also shows up to five lexically similar
failure reports from other sessions. Both comparisons are review aids only:
proposals are not merged, outcomes are not ranked as truth, and no skill is
approved or changed automatically. Matching failure reports are also injected
as bounded context for future similar tasks. They are labeled as user-reported
history, not instructions or proof that the current conditions are identical.
Similarity is based on normalized title tokens, not semantic meaning.
The command jarvis memory procedures approve <proposal-id> --update-skill <skill-name> creates a new patch version only after showing the full replacement and receiving confirmation. Immutable snapshots live outside the skill loader under the local config directory. The commands procedures versions <skill-name> and procedures rollback <skill-name> <version> list history and restore an archived body as a new patch version after confirmation. These commands only target ACE procedure skills and do not execute them.

## Unified retrieval across entry points

CLI chat and ask, tool agents, the OpenAI-compatible server, the Python SDK,
the orchestrator, and local incident reasoning use the same memory-context
builder. It can combine retrieved documents, trusted durable facts, recalled
episodes, explicit user-reported procedure outcomes, and local graph evidence
in one system message. Durable facts are ranked by lexical overlap and,
when enabled and indexed, local vector similarity; recency is a tie-breaker.
Facts without timestamps use their supplied order as a recency fallback.
When available, saved dates
identify when ACE recorded a fact, not when its claim became true. Older notes
remain in memory; the prompt tells the model that a later date is only a clue
and to ask when a current value matters but remains unclear. Documents,
episodes, and optionally durable facts use separate semantic indexes. The
graph source receives at most one quarter of the configured approximate
retrieval-token budget and no more than
384 tokens;
unused graph capacity remains available to the other sources. Counts use
whitespace-separated words as a lightweight estimate, not the active model's
tokenizer. Graph retrieval is best-effort and reads the configured SQLite file
only when it already exists.
If the document backend raises during retrieval, the context builder logs the
failure and continues with the graph and other available local recall sources.

The underlying stores remain separate because facts, dialogue, documents,
procedures, and sourced graph claims have different retention and trust rules.
Their retrieval is now coordinated, but this does not merge or deduplicate the
stored records across those databases.

Focused validation passed 54 context and graph tests, plus the SDK graph-recall
test and the CLI memory-history test. Ruff passed for the changed Python code.
A server-chat graph-recall test was added, but could not run in the current
environment because the optional `python-multipart` package required to import
the server app is unavailable. Full server and ACE end-to-end validation remain
necessary.

The conversation-to-graph episode-link stage passed a focused 61-test set
covering episode persistence, procedure handling, graph capture, and graph
storage. The existing proactive-capture test verifies idempotent episode
storage; the new graph-link case verifies accumulation across 51 exchanges and
the 50-ID relation bound. The broader memory test directory still has unrelated
failures where the native `openjarvis_rust` extension is unavailable in this
environment.

The graph-to-episode retrieval stage then passed a focused 98-test set across
context assembly, graph capture, episode persistence, procedures, and graph
storage. It verifies the conversation-capture-to-context-recall path,
recent-linked-episode priority, deduplication against ordinary episode search,
and the shared context limit. Full server and ACE end-to-end validation remain
necessary. Two CLI tests also passed for opening a full episode by ID and
rejecting malformed IDs.

Query-aware durable-fact selection and saved-date conflict context passed all
41 tests in `tests/memory/test_context.py`; Ruff and `git diff --check` passed.
This focused result does not replace full server or ACE end-to-end validation.

The durable-fact vector sidecar, hybrid context ranking, background indexing
for newly extracted facts, batched Ollama requests with legacy fallback, and
bounded reindex command passed a focused 105-test run across fact indexing,
context assembly, memory capture, CLI memory management, embedding transport,
and ACE preset initialization. The vector client uses the shared loopback-only
Ollama resolver; remote `OLLAMA_HOST` disables vector recall. Tests use a fake
embedder and do not verify a live Ollama model. Full server, live-runtime, and
ACE end-to-end validation remain necessary.

Conversation graph extraction verifies the selected model route through
instrumentation, security, and local-failover wrappers before any user text is
sent. Mixed-pool routes known to be local are permitted; cloud and unknown routes are skipped.
The privacy check does not refresh engine discovery, since that could contact a
provider. Focused tests across graph capture, memory-service construction,
engine routing, telemetry, and security passed 139 cases. These tests use fake
engines and do not measure a live provider.

## Boundaries and remaining work

Episode deletion coordinates the SQLite episode archive, JSONL facts, and
SQLite graph as separate local stores; they do not share one transaction. The
episode is retained until derived cleanup succeeds. If one store fails after
another has committed, some derived data may already be removed. The command
reports that state and can be retried after the store error is resolved.

Semantic recall depends on a locally running Ollama service and the selected
embedding model. When either is unavailable, ACE does not block chat; it uses
the FTS5 index. Episodes that were never captured cannot be reconstructed, and
older episodes need explicit reindexing before vector search can find them by
meaning alone. The capture heuristic and scanner gates can also omit an
exchange that the user later wishes had been remembered. Always-on proactive
source collection is not retained wholesale. Resolved monitor incidents and
approved connector action outcomes enter episodic memory only under separate
opt-ins; connector payloads and personal message content are excluded. When
enabled, the daily proactive agent retrieves a few validated category-and-result
episodes through source-filtered local FTS5 and treats them as unverified
history; they do not change permission rules.

The ACE Windows persistence update applies protected owner-only NTFS ACLs to
facts, episodes, indexed knowledge, graph evidence, monitor history, approval
state, session and agent state, scheduled tasks, and code-improvement records.
On 2026-11-18, the Windows security suite passed 587 tests with 49 skipped; the
focused cross-store and persistence batch passed 129 tests. A wider ACE
regression run covering memory, graph, web research, desktop and host tools,
controlled code improvement, local inference failover, proactive monitoring,
agents, scheduling, and sessions passed 577 tests with 32 skipped. Ruff and
`git diff --check` passed for the changed modules. The engine and desktop tests
use fakes or mocks; these checks do not validate a live local model, real
desktop input, or monitor behavior across a machine reboot.
