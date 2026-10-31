# Source-backed positive class-II MS admission

Codex / GPT-6, 2026-10-26. Parent: `mhc2_improvement_spec.md` and the registered
1825 MS admission family. The completed literal profile finds1,339,673class-II
MS rows,653molecular groups and279context groups. It is not an admission result.

## Required behavior

1. Add a reusable adapter for the native Hitlist training-row contract. Require
   explicit class-II MS positive evidence, human MHC, indexed source-row and
   contributor IDs, a canonical peptide9–512residues, and explicitly unmodified
   chemistry. Modified and unspecified chemistry are separate dispositions.
   No missing measurement becomes a negative or an inferred MS intensity.
2. Use the source's complete candidate set, not a selected binder. Accept exact
   candidate provenance or sample-derived candidates with a uniquely attributed
   arm. Defer study/locus pools, unresolved sets and predicted restrictions.
   Resolve every candidate using the unchanged pinned HumanMHCCatalog; one
   incomplete/ambiguous candidate defers the entire observation. No DRA default,
   sample-wide allele borrowing, chain cross-product or source-genotype inference.
   Preserve all names when molecularly equivalent candidates collapse together.
3. Only uniquely attributed, annotated conditions can supply cell-type and typed
   interventions. Preserve primary-source versus curated-text provenance. Map
   knockout, knockdown, overexpression, transfection, transduction and drug tokens
   to separate namespaced assertions; cytokines remain Exposure inputs. This
   does not establish activity, delivery, processing pathway or a causal effect.
4. Build the vocabulary of mentioned factors within each study from unambiguous
   context records. A literal `none` in that same condition family supplies
   absent states for those declared factors. Blank/unspecified fields supply
   no negative state. A control label alone never supplies an absence claim.
   Reject contradictory `none;factor` fields. Do not infer route activation from
   TAX1BP1, HLA-DM, cytokines or cell identity.
5. Return one positive elution Observation per original row with `split=predict`
   and unpartitioned status. Preserve source hash/ordinal, original row identity,
   source/contributor links, full raw record, candidate and context provenance.
   This is usable admitted evidence, not a training split or calibrated probability.
6. Stream all source rows and preserve every class-II MS disposition. Join the
   exact cached upstream contributor table for retained provenance IDs and retain
   every linked original study/record, including unresolved overlap states. Any
   missing contributor join prevents qualification. Never equate row IDs with
   independently sampled biological replicates.

## Verification and bounds

Artificial tests cover all-candidate conservation, partial-chain deferral,
predicted-restriction exclusion, source identity, null/modified chemistry,
positive-only outcomes, independent simultaneous assertions, explicit-none
versus unknown, pooled-context masking and original provenance preservation.
Run relevant adapter/context/model regressions. Freeze executed source and pinned
input hashes before the real scan. Keep local CPU to one numerical thread, each
analysis invocation<=300seconds/3GiB RSS, and new serialized outputs<=2GiB.
Preserve failed attempts rather than relaxing bounds or dropping rows silently.

Independently replay retained source ordinals/IDs and literal eligibility with
Arrow predicates, verify every candidate groove against eligible catalog rows
and independently reconstruct context claims and contributor memberships.
Report stagewise exclusions, deferred reasons, studies, molecules, cells,
conditions and usable present/absent contrasts. Source qualification does not
promote the broader model. The separate predictive experiment must still freeze
controls, lineage-aware splits, generated-contrast semantics (if used), fitting,
selection and held-out metrics before training.

- [x] Implement reusable row/context admission and focused tests.
- [x] Run bounded source admission and contributor joins.
- [x] Independently verify every retained record and preserve complete dispositions.
- [ ] Register actual comparative training on qualified evidence.

Verification recovery: the initial independent reader failed on an empty selected
batch because Arrow inferred a null index type. Preserve its source and failure
under `reproduce/verification-attempt01/`; use explicitly typed int64 indices,
probe empty and nonempty batches, then replay identical source/acceptance checks.
The admission run and all its outputs remain unchanged; no scientific criterion
or source eligibility rule is relaxed.

The second independent-reader attempt exposed a numeric-versus-text PMID key
mismatch in its own context lookup. Native PMIDs are numeric; the grouped profile
stores string keys. Preserve that failed reader and the first differing record's
metadata, normalize study keys in the checker and replay unchanged admission
artifacts. The adapter already normalizes study keys and correctly retains the
source's explicit absence of CIITA transduction on the parental condition.
