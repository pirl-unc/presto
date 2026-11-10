# Recover paired HLA-DP/DM source observations

Date: 2026-11-10. Agent/model: Codex / GPT-6. Base 04a9d2d9, clean before planning.
Previous goal turn was progress: the completed audit and follow-ups were published.
This family advances Hitlist895 and Presto155 by recovering original source data.
Concurrent Atlas work lives in a separate checkout and is not modified or relaunched.

## Scientific questions and source contract

PXD017154 / PMID33025005 reports engineered HeLa HLA-DP immunopeptidomes with and
without HLA-DM, with CD74 and B7/21 capture. Recover original processed results
and exact chain/run/condition mapping. Verify DPA identity from primary methods,
construct tables or source metadata; do not supply a population-frequency default.
Community SDRF is a locator to cross-check, not sufficient molecular evidence.
No-DP controls retain their own capture-control role and are not pMHC positives
or measured affinity negatives. Technical replicates and redundant search exports
must not inflate independent biological examples. Native peptide forms, search
modifications, decoy status, confidence and protein mappings must remain explicit.

## Procedure and boundaries

1. Retrieve the complete paginated PRIDE file inventory, primary article and
   supplements, retaining URLs, dates, bytes, hashes and failed retrieval receipts.
   Reuse prior pinned metadata; allow up to 24 metadata/article requests and
   processed mzTab/identification files up to 250 MiB total. Do not fetch raw
   spectra or proprietary search databases. Expand a documented budget only
   when concrete file evidence requires it; no arbitrary model RAM limit applies.
2. Resolve sample-to-acquisition, biological/technical replicate, capture,
   DP alpha/beta constructs, DM/CD74 and experimental controls. Preserve unknowns.
3. Inspect processed-file schema before implementing a parser. Recover source
   PSM/peptide identities and modification/confidence/decoy fields. Explicitly
   differentiate PSM evidence, protein mappings, peptide-by-sample observations
   and biological samples. Quantities are only accepted if original measurement
   semantics are documented; no intensity from identification scores.
4. Verify row-level extraction and sample joins independently against source
   files. Summarize source counts and exclusions. No model evaluation, held-out
   prediction access, scientific selection or training occurs in this family.
   Outputs stay unpartitioned/not train-eligible until lineage, peptide overlap,
   chemistry, molecule and source-control qualification pass.
5. Once evidence closes, implement upstream through an isolated Hitlist branch
   under its own workflow; do not create an experiment-only training sidecar.
   Update existing Hitlist895 with concrete recoverable records and remaining
   source gaps. Feed the next cohort through the common Presto adapter.

## Artifacts and success criteria

Family: experiments/2026-11-10_1454_codex_dp-dm-source-recovery/.
Required: README, launch.sh/json/source snapshots, full file inventory, retrieval
receipts, sample/construct map with evidence, parsed source audit/counts, independent
verification and exact raw artifact locators. Publish raw public data to pinned
Modal storage where needed, keep large files/copyrighted article text out of Git.
Update canonical experiment log and task state before commit/push to PR156.
No GPU, pretraining, optimization, synthetic labels, loss or predictive metrics
apply. No TEST split is used because this is source recovery, not a predictive
experiment. Success is recovery of verified usable source facts; any missing
scientific admission criteria remain explicitly open. The full binding, ligand
and processing improvement goal cannot be completed by this source audit alone.

## Evidence-driven confidence recovery amendment

Primary supplemental Methods p2 resolves DPA identity: DP401/402 use DPA1*01:03;
DP10 uses DPA1*02:01. Original mzIdentML Sample names provide acquisition joins.
However both inspected mzIdentML files report passThreshold=true for all PSMs
and a peptide FDR threshold of 10, whereas the Methods requires high-confidence
PSMs at FDR<1%. Do not treat this conversion flag as the published filter.

The archive has 32 original Proteome Discoverer `.msf` processed result databases,
6,397,444,096 bytes total. Recover the original confidence fields from one file
first; if successful, permit those 32 pinned processed databases (up to 6.5 GB)
using streaming downloads/read-only SQLite. This supersedes the initial exclusion
of proprietary search databases for these openly distributed processed results;
no `.raw`, search rerun or new mass-spectral analysis is authorized here. Preserve
all source flags and conversions; source confidence disagreement is a source
qualification gap, not a reason to silently reduce the accepted threshold.

### Native-file transport continuation

The local native 70 MB download is progressing at only tens of kB/s. One CPU-only
Modal pilot (1 CPU, 2 GiB, 15-minute function cap, max 2 containers) will retrieve
and inspect the same public original result near its European repository. No
model data or secrets are uploaded. The first deployment stopped before a
function was created because AWS-style eu-west-1 is not a Modal region. Preserve
that attempt; retry with documented broad region eu (1.15x CPU/memory multiplier).
The next step depends on the native confidence schema, not a model fit.

### Native protein-mapping qualification

All 32 native files recovered successfully. Extraction v1 stopped on high-confidence
PSMs without original protein references; this also explains the pilot's ten
high-confidence records missing from mzIdentML. Retain every native high-confidence
record, with explicit incomplete-protein status. The conservative source-qualified
candidate tier additionally requires complete native protein mappings. This is a
qualification mask, not deletion of raw evidence or an assertion that unmatched
peptides are biologically invalid. Preserve v1 and write corrected v2 outputs.
