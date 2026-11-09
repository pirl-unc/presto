# Modular Presto implementation handoff

Authority: [issue #130](https://github.com/pirl-unc/presto/issues/130), frozen in
`issue130.md`, SHA256
`55e9211b175c0866ceb67bc9c006bd6cfc048d9af8d2f709ccc2e6bb26013f2b`.
Base: `ee4ad60f9750996a4a8e0ecda73e84f583085dd1`.
Work in the isolated `codex/modular-presto-model-20261020` branch. Preserve the
original dirty checkout, shared Python environment and separate draft PR #129.

## Implementation boundary

Implement the issue's software foundation first, without calling untrained
outputs qualified or claiming a new biological result. The registered real-data
identity, 32-example memorization, held-out IC50 and source-transfer milestones
remain separately gated experiments. TCRvdb author access remains pending.

## File and interface plan

1. `data/modular.py`: versioned typed observations, candidate bags, independently
   overridable role metadata and component states; strict canonical sequences;
   raw-unit/qualifier preserving endpoint transforms and training-only scalers.
2. `models/modular.py`: small shared encoder, optional identity heads, deterministic
   resolver, explicit class-I and class-II candidates with 43 PFR features,
   endpoint-masked context, optional TCR fusion, independent digestion and shared
   MS response. Allocate only the declared subset's reachable modules.
3. `training/modular.py`: one observation loss, separate identity replay, strict
   checkpoint metadata, gradients versus updates, train/evaluation support.
4. A normalized, hash-pinned JSONL manifest enters the existing unified training
   CLI through an explicit model-kind dispatch. Existing model behavior stays
   on the existing branch. Preparation adapters must preserve source lineage;
   they may not infer the unavailable TCRvdb raw schema.
5. Tests cover real routing/gradient/invariance contracts and a small synthetic
   software integration. No synthetic result qualifies a biological endpoint.
6. Documentation states shipped capabilities, deferred source qualification,
   unsupported requests and exact next experiment gates.

## Clarifications needed for an executable specification

- An elution candidate set is one observation. Use log-mean-exp of candidate
  cellular scores before adding the shared detector score and endpoint offset.
  This is a declared statistical bag aggregator, not a physical independence
  assumption. Deduplicate identical candidate inputs; preserve provenance.
- Canonical prepared observations require source-declared groove segments;
  full-MHC preparation delegates to the existing strict `mhcseqs` extraction.
- Initial software supports canonical unmodified peptides. PTMs and unknown
  units are explicit admission failures, not silently simplified inputs.
- Optional quantitative MS endpoints require a declared numerical target scale;
  initial binary observation endpoints do not silently accept intensities.
- Unsupported pretrained composition/resume must fail explicitly until the
  exact module/vocabulary/scaler contract can be checked.

## Verification and release

Focused tests first; then pinned Ruff, full regression and strict docs. Save
software verification evidence separately from predictive experiments. Publish
an isolated reviewable PR; leave #130 open for scientific qualification and
remaining data admissions. Do not merge an unreviewed architecture replacement.
