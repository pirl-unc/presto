The published NetMHCstabpan stability archive could expand half-life coverage, but importing its numbers as independent exact measurements would duplicate existing IEDB labels and mishandle unresolved construct/zero semantics.

Sources:

- [Primary paper, PMID 27402703](https://pmc.ncbi.nlm.nih.gov/articles/PMC4976001/)
- [Authors' Stability.txt](https://services.healthtech.dtu.dk/suppl/immunology/NetMHCstabpan-1.0/Stability.txt)

Presto's source-only audit found:

| Finding | Observation rows |
|---|---:|
| Unique reported peptide–MHC rows | 28,166 |
| Positive half-lives | 22,487 |
| Literal zero hours | 5,679 |
| Positive rows with a matching current TRAIN pMHC half-life | 5,012 |
| Positive rows with an exactly matching half-life value after unit conversion | 4,991 |
| Three explicitly mutated MHC identifiers unresolved by the current canonical parser | 1,135 |

There are 75 reported MHC constructs. The archive size matches the paper's measured subset, whereas the paper separately describes adding 1,000 predictor-selected zero-hour negatives per allele. That count agreement does not establish the meaning of each exported zero. Do not treat zero as an arbitrary positive floor or assume every zero is synthetic.

Requested source integration:

- Preserve source URL/hash, line locator, original reported construct, quantity, units and qualifier provenance.
- Resolve source-backed censor/nondetection semantics, including zeros and potential ceilings, with assay-local evidence. Preserve uncertainty when not resolvable.
- Retain explicit mutant identity; do not strip mutations and merge with wild-type MHC. Use canonical parsing/resolution with a documented exclusion when construct sequence cannot be established.
- Recover study/measurement lineage and explicit modification evidence; unknown chemistry is not automatically unmodified.
- Deduplicate against IEDB by assay lineage where possible; pMHC/value equality is an overlap warning, not proof that all remaining rows are independent.
- Keep generated predictor labels separate from measured half-lives. Export qualified records through the existing observations/binding source interfaces with round-trip tests.

The audit has not admitted these rows or changed active training. Counts are specific to Presto's current curated TRAIN subset; whole-study/peptide holdout checks are additionally required downstream. Reproducibility: Presto experiment `2026-10-29_1653_codex_measurement-curation-recovery`, `code/review_stability.py`, `results/stability-review.json`, and preserved initial parser-failure attempt.
