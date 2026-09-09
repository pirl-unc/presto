# Preserve merged publication lineage — Presto #60

## Scope and boundaries

Build directly on merged PR #61. Restore existing PMID, DOI and reference text
through every modality emitted by the canonical merged adapter, sample creation,
collation and held-out observation exports. Preserve authentic observation/assay/
reference identifiers when those columns exist. Keep absent original identifiers
absent: a PMID is a publication identifier, not a unique assay observation, and a
generated input-row ID must not be advertised as an original assay ID.

The #61 source-routing contract and supported populations are fixed. No assay
classification, source curation, model inputs, target masks, loss terms, weights,
sampling or split policy changes belong to this PR. The current source contains
publication fields but no recoverable original assay IDs. This PR cannot by
itself turn the current merged corpus into a fully traceable original-observation
corpus or complete #48/#50/#53.

## Implementation decisions

1. Add publication fields at the end of public data classes to preserve positional
   constructor compatibility. Binding/kinetics/stability/elution already expose
   PMID and source IDs; processing/T-cell/TCR evidence need the missing metadata
   path. Use a shared publication/observation copy helper, not seven independent
   normalization policies. Preserve source strings and explicit missingness.
2. Parse the canonical merged columns once and supply them to the unified and
   emitted typed records. Ensure panel/focused selectors retain available
   provenance too. Do not encode publication or assay identity into sequences,
   model context tensors or sampling strata.
3. Carry DOI/reference text and existing source identifiers through PrestoSample,
   host-side batch lineage and the common held-out lineage export schema. Cover
   single rows and multi-instance observations. Preserve current sample IDs for
   this source, whose original observation IDs are absent. Any genuine IDs must
   retain their distinct role and cannot be replaced with publication IDs.
4. Keep the census's fallback/original-identity distinction. Do not loosen the
   original-identity gate to accept a PMID alone or manufacture assay IRIs from
   numeric row offsets. Diagnostic metadata hashes may change when real metadata
   is restored; demonstrate model-input/label parity using their actual tensors,
   rather than pretending an all-fields sample hash excludes provenance.
5. Document the restored publication path and remaining source identity gap.
   Record additional upstream data needs as Hitlist issues only if the missing
   contract belongs there; adapter omissions are Presto defects.

## Registered validation

Create a separate timestamped lineage experiment before its first launch. Use
the same immutable merged source hash and pinned isolated environment as #61.
Stream the actual loader before/after the metadata correction, observing every
pre-cap constructed record while retaining one per modality to bound memory.
Capture the input row's publication fields at the runner's reader boundary;
compare them to each emitted record's metadata, with per-modality and per-field
availability/mismatch counters. Do not globally replace Python's csv module:
isolate instrumentation to the runner and restore all hooks in finally.

Fingerprint all non-lineage record payloads in source order and require exact
before/after parity per modality, with identical routing, skip reasons and cap
statistics. Preserve source/input hashes, invocation/git state, immutable
launcher/production snapshots, complete summary CSV/JSON and failed receipts.
No MHC filtering or trained-model quality claim follows from these counts.

Semantic fixtures must cover all seven emitted modalities, optional missing
columns, duplicate publications with separate genuine observation IDs, no
invented identity for publication-only rows, head/reservoir caps, stable
sampling/labels, collator/device-transfer metadata and row/MIL held-out export.
Supply explicit fixture MHC sequence inputs rather than consulting shared
registry state. Verify that changing only publication metadata leaves actual
model-forward inputs and loss targets exactly identical. Review any changed
diagnostic hash separately from input parity.

## Closure

- [x] Branch from verified merged #61 and register the experiment.
- [ ] Capture before metadata loss, implement the shared propagation path.
- [ ] Prove semantic/end-to-end metadata preservation and unchanged inputs/labels.
- [ ] Run the after scan and reconcile every row and non-lineage payload hash.
- [ ] Close experiment README/canonical log, review full diff and verify CI.
- [ ] Merge #60 repair, then resume #48/#50 coverage/update evidence.
