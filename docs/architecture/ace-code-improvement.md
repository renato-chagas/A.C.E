# ACE Controlled Code Improvement

## Flow

```text
selected source files -> local model -> unified diff -> SQLite proposal
                                                |
                                disposable repository copy
                                  | apply + syntax validation
                                  | optional Docker pytest, frontend, and Rust checks
                                  v
                          explicit user approval
                                  |
                    baseline check -> recovery journal -> apply
```

`CodeImprovementManager` records the proposed diff, target repository, touched
paths, baseline SHA-256 hashes, status, and validation result. Its states are
`proposed`, `validated`, `approved`, `applying`, `applied`, and `rejected`.
State transitions check the current state inside SQLite, so a concurrent reject
cannot be overwritten by a validation or approval that started earlier.
Validation does not mutate the active checkout. Before applying, the manager
checks every target path and verifies that its current hash matches the
proposal baseline.

The ACE preset also exposes `code_improvement_propose` in interactive
`jarvis chat`. The tool requires confirmation before reading the explicitly
selected files, uses the shared local-only patch generator, saves a proposal,
and performs the default disposable-copy syntax validation. It returns the
proposal ID and full diff when it fits within the response limit. It has no
apply operation. The user reviews and applies the exact saved diff separately
with `jarvis improve show` and `jarvis improve apply`; the latter asks for a
fresh confirmation. Non-interactive agents without a confirmation callback
cannot invoke the proposal tool.

The proposal generator forces local-only inference and uses the shared local
engine resolver. A failed primary can retry through another healthy local
backend or the configured fallback model; cloud engines are never considered.
If no local route can generate a patch, proposal creation fails without
changing the checkout. The generator receives only the files named by the user
and has no tool executor. Each source hash is computed from the exact bytes
read for the prompt. Proposals are rejected if a touched path was not supplied
or any selected source changes before the proposal is saved. The patch parser
also rejects path traversal,
repository metadata, existing or proposed symlinks, Windows junctions and
submodules, sensitive files, detected credential patterns, binary patches,
renames, and terminal or bidirectional control characters. The
default validation checks only diff applicability and Python syntax. It does
not execute project code.

When tests are requested, the temporary copy is mounted read-only into the
`openjarvis-sandbox:latest` container and copied into a 2 GiB `tmpfs`. The
container has no network, no host environment variables, a read-only root
filesystem, dropped capabilities, and resource limits. The active repository
is never mounted into the test container. The Docker image includes the
project's Python development dependencies and an npm cache derived from the
frontend lockfile. `jarvis improve validate --frontend` installs those exact
frontend dependencies offline in the temporary copy, then runs `npm test` and
`npm run build` there. `jarvis improve validate --rust` checks Rust formatting
and runs Rust crate tests with Cargo offline; the PyO3 binding crate is compiled
by the image's existing `maturin build` step and excluded from `cargo test`. The
sandbox image pins the workspace's Rust 1.89 toolchain and pre-fetches the
locked Cargo dependencies during image build. Rust build artifacts stay in the
disposable workspace; Cargo caches use a separate temporary filesystem.

Validation checks the touched file hashes again inside the copy before
applying the diff. It stores the resulting hashes and modes and calculates a
SHA-256 receipt over the proposal's title, repository, diff, file list,
baseline, validation report, and resulting file fingerprints. For sandbox
tests, the report includes the runtime and local image name.

`jarvis improve show` and `jarvis improve apply` print the receipt alongside
the validation report and exact diff. The apply command passes the receipt
from the review through approval and application. If the stored validation
changes during review, the user reviews and confirms the new version. Older
validated or approved proposals without a receipt can be validated again.
The apply preview must produce the same touched files as validation before
the recovery journal or worktree is changed. Validation covers the tested
copy. Later changes to untouched files can affect integration behavior.

Declining confirmation leaves the proposal unchanged. File hashes are checked
again immediately before applying the patch. Proposals with stale baselines
must be regenerated. If applying the patch or saving its `applied` state
fails, the manager uses the same guarded recovery as an interrupted apply.

Before applying, Git applies the diff to the exact original touched files in a
temporary directory. The SQLite journal records the original bytes and modes
(up to 64 MiB per proposal) alongside the expected resulting hashes, modes,
and absent files. The journal and applying state are committed before the
worktree changes. ACE checks the originals again before applying and checks the
result against the preview before committing `applied` and retiring the journal.

After an interruption, recovery checks all touched paths before writing. Files
already matching the original snapshot are left in place. A file matching the
expected result can be restored to the snapshot. Recovery pauses if a path has
different bytes, a different mode, or an unexpected presence or absence.
The CLI reports the conflicting paths and keeps the journal for review. With a
legacy journal, any changed path requires manual review because its expected
result was never recorded. Once the known state has been restored, the proposal
returns to `approved` and a retry still requires fresh confirmation.

Focused validation passed for proposal safety, disposable-copy validation,
baseline checks, rollback after a proposal-state write failure, and the
approval-gated CLI and interrupted-apply recovery.
The shared local patch generator, conversational proposal tool, CLI wrapper,
and tool registration suite passed 43 tests. The generator test verifies that
it forces the provider to local inference, hashes the exact selected bytes, and
rejects sensitive source paths. These tests use a fake local engine; they do
not evaluate patch quality from a live model.

Git-quoted diff paths are decoded before repository containment checks. This
prevents octal-escaped traversal segments from being recorded as a different
safe path; valid quoted names with spaces remain supported. The focused code
improvement, generation, CLI, and tool suites passed 41 tests after this
hardening. Git also rejects the crafted traversal at patch application, so this
change makes proposal validation match the path Git will interpret and rejects
malformed proposals earlier.

## Current limits

- The optional improve propose --discover-related flag adds up to eight nearby
  Python modules by following static imports for two hops. Discovery skips ignored
  and sensitive paths, skips ambiguous module names, and stops at 10,000 files
  or 64 MiB of source. Dynamic imports, generated modules, and semantic impact
  still require the user to select relevant files explicitly.
- Frontend validation requires dependencies present in the image's lockfile
  cache; it has no network access to fetch a newly introduced dependency. GPU
  and hardware-specific test suites are not currently available in the sandbox.
- A validated patch still needs human review. Passing syntax or tests does not
  establish that a change is correct or secure.
- Recovery requires the local improvements database and target repository to
  remain available. If a user changes a target path into a symlink or otherwise
  makes restoration unsafe, startup leaves the journal intact and reports the
  recovery error instead of following that path.
- Multi-file replacement is recoverable after process interruption, but is
  not a single filesystem-atomic operation. A reader may briefly observe a
  partially applied patch before recovery runs after a crash.
