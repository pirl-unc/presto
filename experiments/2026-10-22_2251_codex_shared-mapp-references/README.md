# Shared MAPP protein references with unchanged measurements

October 22, 2026, 22:51 America/New_York; Codex / GPT-6.

**The shared Hitlist MAPP adapter now materializes full protein references,
all exact peptide occurrences and original parent claims as separate relations.**
Independent complete-reference verification passes. All eleven original MAPP
typed tables and all PXD057588 source artifacts are byte-identical to the
[previously verified generation](../2026-10-22_2212_codex_shared-mapp-import/README.md).
Adding candidate protein origins neither changes quantities nor creates extra
measurement rows. Implementation is in [Hitlist draft PR758](https://github.com/pirl-unc/hitlist/pull/758),
commit `1ad9aae`.

## Source and representation contract

Thirteen pinned inputs: the eleven original MAPP source assets plus a historical
reviewed human canonical DAT and the reviewed human 2026_03 isoform-inclusive
FASTA. The original `Human_UP_2016_02.fasta` is unavailable. Both supplied references
are explicitly later model references; the historical local snapshot's precise
UniProt release remains unresolved. Neither is presented as the original search
database. Full sequences, sequence hashes/versions, headers and reference-local
primary/secondary accessions remain inspectable. Reference receipts, locators,
sizes and attribution are frozen in `reproduce/source/mapp_contract.json` and
`mapp_assets.json`; original bytes remain in the local package/datacache.

Seven new relations separate reference descriptors, protein records, accession
claims, exact occurrences, original parent claims, their resolutions and coverage.
Peptide identity connects these relations to measurements. Every exact occurrence
survives, including repeated positions, duplicate full sequences under different
accessions and additional source-unlisted candidates. Coordinates are one-based
inclusive; immediate flanks are literal. No isoform stripping, I/L substitution,
approximate matching or cross-reference alias transfer occurs. Missing IDs and
resolved proteins lacking the peptide remain explicit. Source leading-coordinate
agreement is separate from occurrence elsewhere in the protein. Original parent-list
membership and leading-parent membership are separate flags.

Full-sequence sharing defines reference-local connected components across all
source peptides, including unmatched singletons. These constrain potential splits
but establish neither homology nor specimen independence. They assign no parent
mixture weights or biological origin. Admission remains deferred to the shared
endpoint recipe and its alternative-parent policy.

MAPP remains proteasome-associated captured fragments from human HEK293, with the
original 6–40 search scope, six source conditions, missingness, selection,
confidence and specimen uncertainty preserved. No MHC-binding filter is imposed.
Filtered/unobserved products are not biological negatives. These measurements do
not directly identify free-pool concentration, yield or clearance. Independent
precursor quantities and MS/capture response remain necessary for the broader goal.

## Complete-reference results

| Independently checked | Historical canonical | Modern isoform-inclusive |
|---|---:|---:|
| Full protein records |20,422|42,562|
| Primary/secondary accession claims |90,852|42,562|
| Source peptides with an exact occurrence |8,753|8,790|
| Exact peptide positions |14,258|23,463|
| Original source-parent claims |32,269|32,269|
| Source claims with missing accession |18,730|19,088|
| Resolved candidate proteins |13,811|13,181|
| Resolved candidates lacking the peptide |9|5|
| Components, including unmatched singletons |3,436|3,407|

The modern reference increases sequence support but its FASTA does not carry the
historical DAT's secondary-accession claims. These are different reference
contracts, not interchangeable completeness scores. Reverse/contaminant and
unmapped rows remain for endpoint-specific admission decisions.

The package has 120 files, manifest SHA256
`55f7a3d50df5d162e55bf1d66fbcf413a35fc927fd08c266f90d981068a053c2`.
Its seven new relations contain 2 reference descriptors, 62,984 protein records,
133,414 accession claims, 37,721 occurrences, 64,538 source claims, 26,992 candidate
resolutions and 21,346 coverage rows. See
[independent verification](results/independent-verification.json) for all status
counts and unchanged-file hashes.

## Verification and runtime

The shared producer completed in **54.80 seconds** locally. Independent verification
took **20.22 seconds**, with macOS process peak RSS **0.987 GiB**. These are source
materialization/replay costs, not training benchmarks. No GPU or Modal was used.
The outer `/usr/bin/time -l` wrapper returned exit1 after the producer completed
because the sandbox denied `sysctl kern.clockrate`. The completed artifact and
independent exit0 replay are preserved; no complete producer memory profile is
claimed. See [execution notes](results/execution-notes.json).

The verifier independently parses all proteins with BioPython 1.88 and rescans the
complete references with a character-level Aho-Corasick implementation, unlike
the producer's standard-library parser and prefix index. It verifies full
sequences/metadata/aliases, every occurrence and flank, source claims and all
positive/negative candidate resolutions, coordinate comparisons and coverage.
A separate graph traversal checks shared-sequence components. It also checks the
entire package inventory and hashes, all eleven original MAPP typed tables and
PXD057588 source artifacts against the frozen prior generation. The prior
independently checked mapping outputs are hash-verified before reuse.

Twenty-one focused reference/MAPP/catalog/external tests and Hitlist format/lint
pass. Fixtures exercise overlapping positions, duplicate full sequences, alias
splits, missing isoforms, negative candidates, coordinate conflicts, leading-only
claims, malformed inputs, bad hashes and immutable outputs. The initial full-suite
attempt refused launch at 0.77 GiB available versus its 2.5 GiB minimum; the guard
was preserved. A normal retry admitted one worker at 3.89 GiB and passed **3,236 tests**
in **320.81 seconds** (10 warnings). This is the normal suite; indexed-corpus
integration CI remains a separate gate. See [software validation](results/software-validation.json).

There is **no model, training, pretraining, synthetic biological label, optimizer,
loss, assay head or predictive split** in this family. Held-out regression/binding
metrics and prediction dumps do not apply to a source-conservation audit. No
predictive winner or concentration inference follows from these checks.

## Reproduction and downstream work

[launch.sh](reproduce/launch.sh) gives the local producer and verifier commands;
`producer-executed.sh` preserves the actual timing wrapper. Exact code, source
receipts, parent manifest, input paths and dirty Git states are in `reproduce/`.
The producer ran at dirty Hitlist `fa54fbf` and Presto `396f771`. Adapter schema2
and package version1.82.25 are development state, not a published release. The
only post-run launcher change is import ordering; the executed copy remains
frozen. Reproduction needs supplied original assets and a fresh output path/family
copy; existing packages are not overwritten. Source workbooks are not committed
or bundled in the Python distribution; publisher redistribution rights remain
unverified. The reference receipt records UniProt attribution and CC BY4.0.

Raw shared package: `/private/tmp/hitlist-v2-mapp-reference-second`.
Its manifest is copied to [shared_manifest.json](results/shared_manifest.json).
This external-source generation includes PXD057588 and the supporting predecessor;
it does not claim later T-cell/numeric/query stages have been rebuilt. The schema1
MAPP generation remains immutable and uses its frozen verifier. Parent-package
hardlink limitations remain tracked in Hitlist748.

Next: shared query/recipe admission ([Hitlist739](https://github.com/pirl-unc/hitlist/issues/739))
and a separately registered biological exact-versus-sampled experiment with the
same observable objective and holdouts. [Hitlist757](https://github.com/pirl-unc/hitlist/issues/757)
tracks missing independent precursor estimates and parallel-proteome pairing.
[Hitlist752](https://github.com/pirl-unc/hitlist/issues/752) stays open for reviewed
release and consumer handoff. No source audit completes the availability goal.
