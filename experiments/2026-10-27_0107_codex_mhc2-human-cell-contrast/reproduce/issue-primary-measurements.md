Sarango2022 provides a useful human endogenous MHC-II condition comparison, but the merged IEDB surface does not preserve the original run-level measurements needed to fit and evaluate it. Please add a source-qualified PXD024417 measurement path and reconcile its sample/replicate identities with the existing curated arms.

This is separate from closed #70 and the control/engineering fixes in #576/#716. Current `pmid_overrides.yaml` already preserves Mock, siCTRL and siT6BP, shared memberships and explicit control absences. A current default-branch read matches the pinned2.0.42 Sarango block line for line (GitHub blob `97b951a201686104f8acc1c1482e86494dfaa80b`). Searching issues for PXD024417 returned no existing owner.

Evidence:

- The [primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9724678/) reports pan-MHC-II capture, peptide MS intensities subsequently aggregated by PLAtEAU, and computational DRB1*01:02 affinity estimates. These are distinct evidence types. Expressed DRB1 does not establish peptide-specific restriction after broad capture. The Methods report five technical acquisitions per sample; the text reports two biological siRNA experiments.
- The [PRIDE deposit](https://www.ebi.ac.uk/pride/archive/projects/PXD024417) inventory returns81files:40raw,40MSF search outputs and one community-annotated SDRF. All40SDRF raw filenames match the deposit exactly. It maps eight source/capture groups, each with five technical replicates: Mock1/Mock2/siCTRL/siT6BP for each MHC class. The relationship to the paper's second biological siRNA experiment needs explicit qualification.
- The community SDRF supplies additional MHC typing but marks the typing method unavailable. Preserve this secondary annotation separately; do not promote it to primary genotype or restore withdrawn class-I alleles without verification.
- In the corrected-v4 exported `observations.parquet` (SHA256 `a665f37c0b5917fefb6d12eade530d7e2001de1f0ee73479b21d2c393d173ab4`), an independently replayed PMID36215666 census finds14,903rows:3,028classI and11,875classII. Every class-II row has class-only restriction, an empty candidate-allele set and no quantitative value/unit. Among class-II rows,5,980have one literal curated condition and5,895have shared memberships. This is an export-specific finding, not a claim that the raw deposit lacks intensities.

Requested outcome:

1. Recover and qualify the original human endogenous peptide/peptidoform evidence, measured intensity scale, qualifiers and aggregation. Preserve original measurement/search IDs, source hashes and raw-file lineage; keep PLAtEAU core summaries distinct from peptide measurements.
2. Represent biological material, condition, class/capture and technical acquisition separately. Resolve the biological-replicate mapping or retain it as unknown; five technical files are not five independent biological samples.
3. Preserve source-attested TAX1BP1 perturbation/control metadata without asserting measured activity or assigning each peptide a mechanistic pathway. Shared IEDB condition memberships must not be expanded into independent measurements; missing detections must not become measured negatives.
4. Keep pan-class-II detection, source genotype and computational binding estimates separate. Preserve missing MHC partners/unknown candidates; do not manufacture a DRA allele.
5. Reword the current study note's blanket reduction claim to repertoire alteration unless a specific measured quantity and scope support reduction.

Reproduction and evidence are in Presto's `experiments/2026-10-27_0107_codex_mhc2-human-cell-contrast/` on `codex/mhc2-improvement-20261026`, including source-row indices, independent census replay, current-curation receipt and deposit/SDRF hashes. No model was fitted on this source and no quantitative or causal qualification is claimed.
