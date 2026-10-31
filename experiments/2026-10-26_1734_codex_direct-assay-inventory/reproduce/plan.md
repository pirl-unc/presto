# Direct-assay source inventory before expanding the fitted task set

Prospectively registered 2026-10-26 by Codex / GPT-6. Data-only inventory of the
already pinned Hitlist ci-corpus-v4 binding and other-assays exports, while the
fixed IC50 family runs. This is a literal source-label census, not admission,
model fitting, endpoint relabeling or evidence of sufficient biological support.
No source update, network request, sequence export or cohort materialization.

Use only the existing local release directory
`/Users/iskander/.hitlist/corrected-corpus/public-downloads/v4/bundles/2026-10-16T07-20-43Z/`:
- binding.parquet, SHA2562b3f6ea2dd647e7fc54d214eef420feaebc4e81fab97ffc0565a2de1f8695ba0,
  expected814,268 rows;
- other_assays.parquet, SHA256c40f5227ded408830885ecba8336dbdd701172df9573b0d32e5e8a5f8fc947c2,
  expected26,537 rows.
This is the frozen diagnostic export, not a claim that it is the latest release.

Stream batches of4096 selected metadata columns on one local CPU thread. Preserve
literal source, response_measured, assay_method, measurement_units,
measurement_inequality, mhc_class, has_ptm, is_binding_assay and is_ms_observation,
including null separately from empty strings. Count each joint combination plus
numeric-value state: missing, nonfinite, nonpositive or positive finite. Report
raw row counts, distinct nonempty assay/reference IDs, and per-field missingness.
Do not infer a raw missing qualifier's meaning, convert units, assign endpoints,
extract sequences, invent partners, deduplicate measurements or make splits.

Independently replay joint counts by Arrow group-by on separately streamed batches
and verify all marginal totals, ID counts, missingness, source hashes and parquet
row totals. Both readers must handle all840,805 source rows. Retain the literal
label/unit/method census so the next head's admission specification can distinguish
KD, stability, melting, EC50 and proxy outcomes from what is actually present.
After inventory choose the next source qualification, without claiming any row
has been admitted to a training endpoint.

Register family `experiments/2026-10-26_1734_codex_direct-assay-inventory/` before
execution. Freeze launcher/independent verifier, invocation, git state, environment,
source hashes, CPU/package details. Use existing Python with no installs. Each
scan: maximum300seconds, peak process RSS ceiling3GiB, startup available RAM10GiB
and disk15GiB; live floors5GiB RAM/10GiB disk, polling0.25seconds. Preserve failures,
no automatic retries or file overwrite. No neural work or extra numerical threads.

Required closure: raw joint census JSON/CSV, per-source summary, independent replay
receipt, elapsed/peak memory, README and canonical log. No predictive metrics,
training parameters, validation/test splits or prediction dumps apply: no model
is trained/evaluated and no training cohort is produced. Synthetic data, loss
weights, assay-output mappings and requested/observed GPU are all inapplicable.
