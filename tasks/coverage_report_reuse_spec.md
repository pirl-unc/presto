# Reuse invariant coverage counts — follow-up to #48

Agent/model: Codex / GPT-6. Base: PR #65 merged as
`8b5f60fd40684b8af5edfae3c3c52a42fc8b2bfa`. The independent review passed 141
focused tests; both final CI runs passed 2,063 tests / 3 skipped. No deployment
workflow exists. This change addresses report cost, not supervision adequacy.

## Problem and evidence

The uncapped merged census retained 2,081,922 examples. At 18:47 UTC its report
had reached approximately 207 count queries, with recent queries taking 65–75
seconds each; it was still reporting at 20:01 UTC. Raw progress and SQLite
evidence remain in the registered canonical-coverage family. Categorical columns
select the same observations, but currently repeat every distinct-count and
facet query. A sole evidence group also repeats the already computed population.

## Minimal implementation

- Keep source rows, observations, SQL selection rules, output schema, zero-support
  entries, exact distinct identities and all label/gate semantics unchanged.
- Cache only categorical endpoint counts that are independent of the requested
  class. Key by the complete selection and facet option. Each class still uses
  exact SQL for positive/negative observation and distinct-identity counts,
  including repeated source rows and conflicting labels. Do not derive unique
  balances from training-row totals.
- Clear cached categorical counts before each added batch, including failed
  additions; return independent dictionaries so report/caller edits cannot
  mutate cached values or another claim cell.
- When an endpoint has exactly one role/family/source group, copy its already
  computed overall and column counts into that group's report entry. Preserve
  independent nested dictionaries. Multiple groups retain separate queries.
- Keep reuse local to one census object; do not introduce a persistent cache or
  modify the active remote run, its database, or historical source snapshots.

## Verification

Add meaningful regressions for class-specific response counts, conflicting
identities, empty support, every filter dimension, facet options, cache
invalidation after adding data, and independent returned dictionaries. Check
SQLite traces to show categorical columns avoid repeated invariant/facet scans
and sole-group reports avoid duplicate queries. Retain mixed-group behavior.

Compare complete reports and selected counts with the unmodified merged-base
implementation on representative canonical row/bag/panel fixtures. Existing
coverage/preflight/support tests remain required; run pinned lint/format checks.

Before any real-data timing comparison, register a separate experiment family
with exact baseline/candidate source hashes, read-only retained database identity,
selected split/endpoints and repetitions, CPU/memory/timeout limits, invocation,
environment, timing and exact output comparisons. Compare both implementations
under the same contract; do not claim a full-report speedup from a selected subset.
Preserve and close any newly completed original census before further launches.

## Delivery

Preserve independent review notes, record the PR #65 merge and its CI evidence,
review the final implementation, and open a new PR against main. #48/#50/#53
remain open until their real-data/update/prediction requirements are fulfilled.

## Revision after review — 2026-09-13

Agent/model: Claude / Opus 5. Five independent reviews found no wrong counts for
current callers but confirmed defects in the first implementation (`3532ae1`).
This revision fixes them:

- Class columns re-entered the public `counts()`, so a subclass override such as
  the canonical-coverage wrapper ran twice per class column. Counting now lives in
  private helpers and each public call reaches an override once.
- The cache was cleared only when `add_batch` began, so a rollback or a write that
  bypassed `add_batch` could combine stale totals with a live balance. Reuse now
  happens only inside `reusing_counts()`, a read-only block that `report()` runs
  in. Outside it nothing is cached, so no write, rollback, schema change, restore
  or other connection can serve stale counts, and a closed census raises for every
  endpoint. A first attempt keyed validity on the connection's `total_changes`; an
  adversarial review showed that DDL table swaps, temp tables, backups and second
  connections bypass that counter. Supported-output claims are evaluated outside
  the block, at the base statement cost.
- `add_batch` is atomic through a savepoint: a rejected observation undoes only
  the batch's own rows, split counts, evidence fingerprint and objective hits,
  leaving earlier uncommitted writes intact. This supersedes "clear cached
  categorical counts before each added batch, including failed additions" above.
- Cache keys compared bound values with Python equality, so `1` and `1.0` shared
  an entry that SQLite treats as different selections. Keys now use `repr`.
- Each class column still scanned its selection for its balance. One grouped
  statement now yields every class's exact balance. Negative and unique-negative
  counts follow by subtraction because `target` and `observation_key` are NOT
  NULL, and targets match by exact equality, as `o.target=index` does. A cold
  single-column request instead folds its class into the population pass, so it
  issues the same statements as before reuse. Balance aggregates have one
  definition shared by binary and class responses, and population fields are
  mapped by name rather than position.
- The single-group report shortcut is decided once and documented. A test compares
  every evidence group, columns included, with its own filtered counts, which
  also guards the assumption that evidence filters are plain IN restrictions.
- Evidence-group columns, failed batches, rollbacks and statement totals lacked
  tests. Each new test fails against `3532ae1` or kills a targeted mutant.
- Parity evidence moved from `/tmp` and the gitignored `artifacts/` directory into
  the canonical-coverage family's `analysis/` and `results/verification/`.

Unchanged, with evidence: cache memory (a few MB at most on the real report),
deep-copy cost (well under a millisecond per hit), reusing facet-bearing entries
for facet-free requests (only a test disables facets), and grouping every endpoint
by column or evidence group (measured slower; column grouping cannot serve classes).
