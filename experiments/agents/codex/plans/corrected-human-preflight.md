# Corrected human corpus: prospective consumer qualification

Agent/model: Codex (GPT-6). Use the verified corrected human native manifest
`b180eba804532f633a6d61bb350584dbe24f8df4b4b96f02c6ec492c60144372`, produced by
released Hitlist 1.78.0. It has 4,071,463 observations with unchanged source IDs
and contributor links; metadata corrections affect 89,567 observations. Original
human baseline membership/results remain immutable and do not qualify this input.

Use the independently verified released Presto 0.1.13 wheel (source main
`a6f8af39256db799ddff417cd16386ea66cea0c4`, wheel SHA-256
`6bdc3f9cce3272ffe02261652ebb3bf1b97da8dfc21c8ea358c8d01382f9cd81`). This PR
versions the recipe as 0.1.14; it must not mislabel the actual installed producer.
Freeze the exact wheel, code, config, dependency lock and source pins before launch.

Canonical frozen Hitlist route only: cap binding/elution at 5,000 each and
kinetics/stability at 1,000 each. Explicit reservoir selection, data seed 42,
canonical source seed offset retained; explicit relative_deficit_v2 whole-component
allocation, split seed 42, 80/10/10 train/validation/test, peptide-disjoint policy.
Both MHC classes remain eligible. Strict MHC sequence filtering, traceable lineage,
no flanks, live/merged/bulk inputs, generated controls, MHC-only augmentation or
new expression/reference backfill. All regularization weights are explicitly zero.

Run data_preflight_only with a forbidden model constructor. Preserve the complete
source funnel, cap mode/stream receipts, original/resolved allele sets, every
measured-root assignment, model membership/parent graph, full per-output/family/
source/class/censor/split census and root/model leakage audits. Unknown biological
identities and shared studies are limitations, not pass. All sources and reusable
products use datacache under a new experiment root. Only the existing exact
mhcseqs 2.6.13 catalog may be reused through its unchanged hash-bound datacache
recipe and the same two official commit-and-hash-pinned FASTAs; no old cohort or
assignment reuse. Retain catalog hashes and complete merge/validation reports.

Prospectively register a candidate for the subsequent fitting diagnostic:
`assays.IC50_nM`, scalar, direct `binding:IC50` evidence from `iedb` or `cedar`.
Require at least 50 distinct observations, 50 distinct peptides and 50 exact values
in each of train/validation/test. These thresholds are fixed before this new
preflight; never select thresholds from its counts. After actual source and
membership fingerprints exist, bind these fixed claims into a prospective
supported-output manifest for the next canonical fit. Preserve the canonical
preflight's undeclared-output report; an evidence census is not an executed
manifest gate or a trained-output claim. If counts fall short, record the failure
and do not relax thresholds or claim readiness. No model/predictions/optimization
or performance metrics exist in this data-only family.

CPU only: eight CPUs, 64 GiB RAM, 512 GiB scratch, one Torch/BLAS thread, no loader
workers, 50-minute function limit and 45-minute work alarm. Save failures unchanged,
all logs, dependency versions/pip check, runtime and runner-only RSS. Publish a
complete archive and inventories through datacache, independently verify public
downloads, and close the canonical experiment registry with explicit decisions.

Version 0.1.14, separate PR. Format/lint/full tests and strict docs, review and
final-head checks, merge and verified clean-main GitHub distributions are required.
Hitlist #678/#680 and Presto #48/#53 remain open for scientific acceptance. The
next bounded real-data fit must require the new frozen manifest and retain exact
membership; no conclusions transfer from the historical within-search model.
