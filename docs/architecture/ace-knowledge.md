# A.C.E. Research and Relational Memory

## Audit

Before the ACE integration, OpenJarvis provided three reusable pieces:

- `WebSearchTool` searches the web and can fetch a page through a redirect-aware
  SSRF check. Search output is formatted text and is not persisted.
- `KnowledgeStore` stores source-aware documents in local SQLite/FTS5 and
  supports BM25 retrieval. It is separate from the generic SQLite memory
  backend and from automatic JSONL fact extraction.
- `KnowledgeGraphMemory` stores entities and relations in SQLite. Its initial
  schema has no edge deduplication, evidence table, confidence/status fields,
  or audit history. Existing `kg_*` tools are registered but the normal ACE
  profile does not enable them or inject a graph backend.

The implementation enables research persistence, evidence extraction,
graph-context retrieval, graph inspection tools, graph API editing, and a
frontend viewer. It keeps the existing stores separate and connects them
through stable source IDs and URLs rather than replacing existing stores.

## Implemented components

- `WebResearchTool` uses `WebSearchTool`, stores fetched pages/snippets in the
  configured `KnowledgeStore`, and writes source and claim evidence to the
  graph. Extraction is best-effort and runs only after the selected model route
  is verified as local, including custom OpenAI-compatible endpoints. Remote
  endpoints marked non-cloud are skipped. If the provider fails, the tool
  searches saved web pages locally and labels them with their saved timestamp
  as unrefreshed historical evidence. When a result supplies only a snippet,
  ACE retains any existing page capture, including earlier captures whose kind
  is unknown. A successful refresh updates the stored text and FTS index in
  one SQLite transaction. Tests cover fallback, rejected writes, and direct URLs.
- `ResearchAgent` does not accept an initial model answer as research evidence.
  If the selected model does not call its search tool, ACE runs the original
  query against the local knowledge corpus and asks the model to synthesize
  from those retrieved records with tools disabled. With no matching records,
  it returns a clear no-evidence response instead of an unsupported answer.
  Retrieved source text is explicitly marked as untrusted data in both the
  normal tool loop and this compatibility path.
- The optional `memory.graph_capture_enabled` path extracts explicit claims
  from the user's own conversation messages on the existing background memory
  worker. It requires verbatim user-message evidence, ignores assistant output,
  and is enabled by the ACE preset only. When that exchange is also retained
  as an episode, its stable episode ID is recorded on the relation as a bounded
  provenance link; this does not verify the claim. Episodes and graph evidence
  use separate databases, so the memory CLI explicitly purges attributed graph
  evidence when an episode is forgotten; independent document and web evidence
  remains. It uses the native secret scanner or its local
  pattern fallback; capture fails closed if both are unavailable. Before
  sending text for extraction, it verifies that the selected model route is
  local through engine wrappers. Unknown routes in a mixed-engine pool fail
  closed without refreshing provider discovery.
- Explicitly indexed files also contribute document-sourced graph claims. The
  `/v1/memory/index` route reuses indexed chunks, selects a local model even
  when the server has a mixed local/cloud engine pool, and accepts only claims
  with exact quotes from the source text. Local secret and injection scanners
  gate stored evidence. Each indexing request processes at most 12 documents
  and 10,000 characters per document; the response reports skipped documents,
  stored claims, and extraction failures. Entity evidence keeps up to 50
  source-linked quotes, while relation evidence remains attached to each edge.
- Relation API objects expose their newest evidence source and URL while
  retaining the full evidence list. Repeated entities keep up to 50 distinct
  source-linked quotes in local properties.
- Graph schema migrations add stable edge IDs, confidence/status, source
  evidence, and audit events while preserving legacy entities and relations.
- The standard query orchestrator injects a bounded, provenance-linked graph
  context when a graph database already exists.
- `kg_query` and `kg_neighbors` include evidence. `kg_add_entity` and
  `kg_add_relation` support provenance-aware updates.
- `/v1/memory/graph` serves bounded graph data and edit/delete endpoints. The
  app's Knowledge Graph page renders those real records and supports search,
  type filters, entity correction, relation editing, and deletion. Relation
  editing includes source and target entity selectors. Changing an endpoint,
  type, or polarity clears the old evidence form and returns the claim to
  `reported`; the database retains prior evidence as superseded history when
  the claim identity changes. The editor starts with empty fields for adding
  evidence. Changing the status alone preserves the existing quotes and their
  provenance. Quotes display their originating conversation episode ID when
  one was recorded. The graph view supports bounded zoom and pointer-based
  panning, with arrow-key panning available when the graph itself has focus.
  Relations between the same pair of entities follow separate curves with
  arrows that point to their targets. Self-relations appear as loops.
  Select an edge with a pointer or the keyboard to inspect that relation's
  evidence and edit controls. The entity view can show all connected
  relations again.
  Filter by relation status to narrow the edges and the selected entity's
  relation list. Edges distinguish hypotheses, verified, contradicted, and
  superseded claims. Conflict cards distinguish differing targets from opposing
  polarity, and evidence with explicit dates shows historical, future, or
  stated validity relative to the viewer's local date. The frontend suite
  passed 133 tests, including the filter options and inclusive date
  boundaries. Five graph API tests passed with FastAPI TestClient, including
  the preservation of existing episode evidence during a status change.
  Evidence URLs are clickable only when they are valid HTTP or HTTPS URLs
  without embedded credentials; other source values remain visible as text.
- `GET /v1/memory/graph/events` exposes the persisted `graph_events` audit log
  as newest-first pages using an event-ID cursor. The API returns only bounded
  entity/relation snapshots, omits arbitrary entity properties, caps source and
  quote text, and the Knowledge Graph page displays history without loading the
  complete event table into memory.

## Document-capture validation and limits

Focused validation passed for document extraction, provenance deduplication,
unsafe-quote rejection, graph persistence, and local-engine selection in a
mixed local/cloud pool: 25 memory/graph tests and the dedicated API integration
case passed. Ruff passed for the changed Python sources and tests; TypeScript
no-emit checking passed for the indexing UI.

Document graph capture is limited to the first 12 indexed documents per
request and the first 10,000 characters of each document. Optional image and
scanned-PDF OCR is available only when explicitly enabled through the CLI, SDK,
API, tool, or Data Sources page. It invokes local Tesseract, prefers installed
Portuguese and English language data, falls back to one installed language, and
uses no network or mandatory Python dependency. Scanned PDFs also require the
local Poppler pdftoppm executable. Image OCR is bounded to 20 MiB and 30 seconds
per image. Scanned-PDF OCR is bounded to 50 MiB, 20 image-only pages, 180
seconds per file, and 100,000 extracted OCR characters. OCR is off by default.
Audio/video indexing is a separate opt-in using the configured local
Faster-Whisper model and PyAV. It requires the model to already exist in the
local cache, disables model downloads, transcribes only the audio track of
video files, and is bounded to 50 MiB and 30 minutes of source media, with
100,000 transcript characters per file. OCR and transcript text are treated
as untrusted source text; graph capture still requires exact quotes accepted
by the local secret and injection scanners.
Local-model availability and extraction quality remain runtime dependencies;
indexing still succeeds when graph extraction is skipped, and the API response
reports that state. Full ACE integration and deployment validation remain to be
completed.

The conversation privacy gate also passes focused tests for wrapped cloud
engines and mixed local/cloud model routing. This verifies route selection in
the configured engine map; it does not validate a live inference provider.

## Decisions

1. Keep SQLite as the only required persistence service. Use the existing
   `KnowledgeStore` database for fetched research text and a separate SQLite
   database for graph entities, relations, evidence, and change history.
2. Reuse `WebSearchTool` for search and its redirect-aware safe page fetcher.
   Web results remain untrusted input. A local extractor may propose graph
   claims, but each relation must include a verbatim evidence quote found in
   the saved page text.
3. Represent confidence and assertion status explicitly. Repeated evidence is
   appended, not used to overwrite prior evidence. Opposite claims remain
   separate edges; the system does not decide which source is true.
   The graph API also surfaces bounded review candidates when the same subject
   has active edges with the same normalized predicate and different targets,
   or has both affirmed and negated claims about the same target. This is not
   proof of a contradiction: different-target candidates use a small allowlist
   of single-value predicates. Evidence may carry inclusive `valid_from` and
   `valid_until` ISO dates. Conversation and web extractors accept a date only
   when that exact date appears in the verbatim evidence quote; manual graph
   edits can enter the same dates. Claims with explicit, disjoint date ranges
   are not presented as conflicting. Missing dates mean the time range is
   unknown, so those claims remain possible conflicts. Context labels a range
   as historical or future relative to the current local date; it does not
   treat that label as verification. Candidate detection
   does not change edge or evidence status; a user
   can review and explicitly mark a claim
   contradicted while retaining all source history. Affirmed and negated edges
   have separate stable IDs; older graph rows migrate as affirmed without
   changing their existing IDs or evidence.
   Mutable predicates also receive a bounded evidence-review reminder after
   their default interval: location/status 30 days; version/model/deployment
   claims 90 days; default backend/model/provider 180 days; address 365 days.
   The age uses the evidence row's local SQLite `created_at` timestamp, not a
   source-supplied `observed_at`, publication date, or a fact's actual
   valid-time. The separate evidence validity dates describe the period stated
   by a source and are not inferred from publication or record timestamps. The
   reminder never marks a claim stale or changes status automatically;
   the API, contextual retrieval, and viewer label it only as due for review.
4. Expose bounded graph search to model context and tools. Traversal depth and
   result limits are capped so a large graph cannot flood a prompt.
5. Build the viewer from the graph API's stored nodes and edges. Do not create
   demo nodes in the frontend.
6. Web research is opt-in per tool call to the internet and stores results
   locally by default. If external search is unavailable, the tool can retrieve
   previously saved web pages without presenting them as current research. No
   graph, vector, or search service is required beyond the local model runtime
   used for extraction.

## Data flow

```text
User query -> WebSearchTool -> safe page fetch -> KnowledgeStore (SQLite/FTS5)
                                     |                    |
                                     +-> local extraction +-> graph evidence
User message -> background local extraction -----------------+
                                                             |
User question -> text/semantic memory + bounded graph context -> local model
                                                             |
                                             local graph API -> graph viewer
```

## Browse a neighborhood

Select an entity and choose **Local focus**. The graph centers on that entity
and loads its incoming and outgoing edges from SQLite. Select a neighbor to
move the focus. **More neighbors** and **Previous** navigate the pages of
incident edges, including edges outside the initial global view. The default
page contains 32 edges. Type and status filters apply to the displayed page.

The graph API accepts a `focus_id` and returns `next_relation_id`. Pass that
value as `before_relation_id` to retrieve the next page. Exact entity IDs keep
similarly named nodes distinct. Pages use relation row IDs for ordering, so
editing an existing relation's text does not move it between pages.

The FastAPI tests use a SQLite fixture with an incoming relation outside the
global slice and verify that it appears on the second neighborhood page.
A headless Chromium fixture verified page navigation, changing the focused
entity, and returning to the global view.

## Validation targets

- Schema upgrades preserve existing graph rows and are safe to repeat.
- Re-adding the same edge does not create a duplicate; new source evidence is
  retained and survives reopening the database.
- Deleting a node removes connected edges and their evidence transactionally.
- Research persists fetched content and provenance locally, rejects unsafe
  URLs through the shared fetcher, and does not store unsupported extracted
  relations.
- Graph retrieval is bounded, includes evidence URLs, and treats graph claims
  as reported evidence rather than verified truth.
- Same-subject/same-predicate competing targets and opposite explicit
  polarities are surfaced in contextual retrieval and the graph viewer as
  possible conflicts when their explicit validity periods overlap, without
  deleting claims or deciding which source is correct. Known disjoint periods
  are kept as separate historical claims; unknown periods remain ambiguous.
- Validity dates are inclusive ISO calendar dates. Extracted dates must occur
  verbatim in ISO form in the supporting quote. Invalid, reversed, or
  unsupported dates are ignored by extractors, and manual writes reject
  invalid ranges. Repeated
  evidence with the same source and quote but a different validity period is
  retained as a separate evidence record.
- Old evidence for allowlisted mutable predicates is surfaced as due for
  review, while static identifiers and birth dates are excluded; no stored
  relation is invalidated or rewritten because of age.
- The viewer renders the API's real graph and can filter/search without a new
  frontend dependency.
- Graph history pages are newest-first, cursor-bounded, preserve add/edit/delete
  records after the graph item is removed, and do not expose arbitrary entity
  properties through the history endpoint.

The graph distinguishes a claim's current status from its retained evidence.
Changing a claim's identity does not erase earlier quotes: they remain marked
as superseded and are excluded from normal graph context. Research extraction
is optional and local-model-only; web text is still saved if no local extractor
is available.

The evidence-review reminder uses the local database record time. A recently
ingested source with an old publication or observation date is not treated as
old evidence, and a source's `observed_at` value cannot make a newly recorded
claim appear due. The API and contextual retrieval include the last locally
recorded active evidence, excluding superseded and contradicted evidence.

Focused validation for this stage passed 18 graph tests and the dedicated API
review-candidate test. The cases verify local record-age semantics, exclusion
of inactive evidence, and preservation of relation status. Ruff passed for the
changed Python sources and tests, and TypeScript no-emit checking passed for the
graph viewer. This does not replace full-suite or end-to-end ACE validation.

The validity-time update passed 38 focused graph, conversation-extraction, and
web-research tests, including migration from the previous evidence uniqueness
constraint and retention of repeated quotes with different validity periods.
Ruff and frontend TypeScript no-emit checking passed. The API route test module
was skipped because FastAPI is not installed in this environment. Extractor
tests use a fake local engine; they verify quote and date validation, not the
accuracy of a live model's date extraction.

The graph editor now allows manual correction of the relation's source and
target entities. TypeScript no-emit checking passed after this UI update. The
viewer received a headless Vivaldi smoke check with the Vite app and a loopback
mock graph API. It rendered three entities, two relations, and graph history;
selecting a node opened its details, zoom updated to 115%, background drag
changed the graph transform, and the type filter reduced the view to one
hardware entity. The drag check exposed an early `lostpointercapture` event
ending the gesture; panning now ends on pointer-up or pointer-cancel. This smoke
validates frontend rendering and interaction against fixture responses, not the
real graph database or API in the user's installation.

Graph evidence links now reject executable and non-web URL schemes and URLs
with embedded credentials. The focused helper tests and full frontend suite
passed after this change.

The graph-history update passed 23 Python graph/storage tests, 3 focused
frontend history-label tests, TypeScript no-emit checking, Ruff, and a direct
FastAPI route smoke check for newest-first paging and bounded snapshots. The
project virtual environment skips the API test module because FastAPI is not
installed there; the route smoke used the system FastAPI without installing
packages. The full frontend suite and broader regression validation were run later and
are recorded below.

## Validation update — 2026-11-18

The broader ACE regression run passed 566 tests with 32 skipped. It covered
memory, proactive behavior, graph and web research, desktop and browser tools,
code improvement, local failover, model setup, doctor, and API routes. The
project environment skipped API cases because FastAPI is unavailable there.
After browser cleanup and accessibility-tree error handling fixes, the focused
browser set passed 87 tests. Ruff, formatting, and diff checks passed for the
changed Python files.

The complete frontend suite passed 126 tests across 26 files, and TypeScript
no-emit checking passed. The headless Vivaldi smoke above adds a visual check
for the running frontend with fixture API responses; real database-backed
visual behavior remains outside that smoke's coverage.

On 2026-11-18, the actual `memory_router` was mounted in an isolated FastAPI
app backed by a temporary SQLite database created through
`KnowledgeGraphMemory`. Requests sent through the frontend's Vite `/v1` proxy
returned three entities, two relations with two evidence records, and five
persisted graph events. A headless Vivaldi session loaded that API through the
same Vite proxy: the DOM showed three nodes, two edges, two evidence records,
and five history entries; selecting an entity, zooming, panning, and filtering
to one hardware node worked. The screenshot contained duplicated compositor
tiles from headless Vivaldi, so it is not a visual layout sign-off. This
verifies the real route, SQLite serialization, frontend-origin proxy path, and
browser interactions against seeded data. It does not verify the user's
installed server startup or their existing graph database.

The selected-engine Deep Research path now reuses the chosen planner route when
it can serve the selected model, respects an explicit `[deep_research].engine`,
and retains local-only endpoint checks. If a planner cannot call tools, the
research loop performs a local retrieval itself and requests a tools-disabled
synthesis from those records; without hits it returns a no-evidence message.
Retrieved source text is marked as untrusted data in both the normal tool loop
and this compatibility path. A regression run across the research loop, CLI
routing and end-to-end flow, embeddings, and web search/research passed 179
tests. The extractor was then hardened to reject remote custom inference
endpoints regardless of their `is_cloud` label; a focused web-research, web
search, and research-loop run passed 152 tests. Ruff, formatting, and diff
checks passed. The tests use mock engines and stores; they do not verify tool
calling or synthesis quality on a live model.
