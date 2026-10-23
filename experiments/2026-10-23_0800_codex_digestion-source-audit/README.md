# Known-substrate digestion: original data qualification

The public processed dataset is recovered and supports a substantially better
kinetic test than conditional cellular MAPP composition. **It is not yet admitted
for training.** One source time-label conflict must be reconciled, and the raw,
filtered and reported measurement layers need a shared Hitlist adapter.

Registered October 23, 2026 before processed-data retrieval; Codex / GPT-6.
Starting Presto 1110d49 with unrelated dirty tasks recorded in launch.json.
No model fitting, pretraining, synthetic biological labels, validation/test split,
loss, predictive metrics or prediction dumps in this source audit. This family
qualifies measurements and changes the next admission decision. It does not
establish intracellular concentrations or close the availability goal.

## Original source and retrieval

[Soh et al. 2024](https://doi.org/10.1038/s41467-024-45339-3),
[MassIVE MSV000092631 / PXD044463](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000092631),
public CC0 dataset. The exact 505-entry file inventory was recovered in the prior
controlled-source audit; the source page explicitly gives
`ftp://massive-ftp.ucsd.edu/v06/MSV000092631/`.

The website DownloadResultFile route returned429, HTTPS on the FTP host timed
out, and plain anonymous FTP returned421 requiring TLS. Explicit TLS FTP with
certificate verification succeeded. A 65,536-byte tail first recovered the ZIP
directory. The complete processed archive is 963,763,954 bytes, SHA256
`939a87fc73f8321d177011ad68380aec0b3ea84020bf07f88e221cee2103e0ff`.
Full download 258.563 s, approximately 3.73 MB/s. No raw spectra or 7.8 GB search database
was downloaded. Failed attempts and successful transport evidence are retained.

All 18 selected original files are pinned by URL, original descriptor, expected
size, SHA256 and local path in [source-assets.json](results/source-assets.json):
the processed archive, Identifier.xlsx, deposited methods DOCX and 15 literal
substrate FASTAs. Each construct exactly matches the corresponding final-table
substrate sequence; three Ffh conditions share one construct. No modern canonical
protein substituted for a deposited construct.

## Actual coverage and metadata

| Original layer | Verified coverage |
|---|---:|
| Nested condition archives |17 across15 substrates|
| Nested archive entries |391; every outer and inner CRC verified|
| Main-search acquisition records |388|
| Raw Skyline table rows |5,792,969|
| Raw peptide/run quantity records |545,664|
| Quantified final peptide/biological-replicate rows |53,144|
| Additional annotation-only rows |269|
| Positive reported final values |213,022|
| Reported zeros |62,860|
| Literal NaN reported values |4,782|
| Final rows with repeated time labels |9,716|

Counts refer to distinct grains; they must not be summed as independent training
examples. Most time courses use0/1/2/4/24h; alpha-synuclein uses
0/0.25/0.5/1/3/6/24h. Two biological digests and repeated technical acquisitions
are explicit. The 17 archives include Ffh at 1:6.25, 1:12.5 and 1:25 enzyme:substrate
ratios. Substrate-only controls are separately named in the workbook. Instrument
identity varies across substrates and is retained. Human enzyme origin does not
supply a species label for every substrate, and enzyme preparation is not an APC
sample. The deposited methods' broad organism field is unknown.

Every raw quantity source name joins to an original main-search acquisition.
Biological replicate aliases are retained through those exact source joins,
including1/2,b1/b2,A1/A2 and other literal labels. The final CSV alone loses these
run identities and the distinction between control and digestion time-zero
entries. The adapter must retain the earlier layers and their contributors.

## Concrete source conflicts and measurement semantics

1. **RF1 time conflict:**
   `HRoetschke_221222_180123_HFGoe_RF1_b1R1_0h` and the corresponding`b1R2_0h`
   file are 0 h in the metadata and filenames, but 1 h in QUANTITIES_raw.RData.
   This affects 2,278 rows. All other raw quantity time joins match. Preserve
   both original values and an explicit conflict; do not silently reinterpret
   the already aggregated reported trajectory. RF1 is pending reconciliation
   before kinetic training. [Exact rows](results/raw-run-crosswalk.json).
2. **Control remapping:** the tagged postprocessor maps technical labels 3→1 and
   4→2, pools background information from time zero and emits repeated zero-time
   values in some final vectors. These are not additional independent temporal
   observations. Original run-level control identities must survive curation.
3. **Zeros and missingness:** the postprocessor sets time-zero corrected values
   to zero. Later values at or below the estimated background become missing,
   and the reported mean ignores missing technical replicates. A forced zero
   is not a measured absence; NaN is not a biological negative.
4. **Selection:** the archive includes raw and filtered RData, assignments,
   Skyline precursor quantities/background, final vectors and logs. The final
   CSV includes annotation-only rows and is not by itself the paper's additional
   positive 4 h/3 h selected panel. Do not infer terminal-time selection or label
   all final rows as eligible without an explicit policy. Source synthesis-error
   filtering and I/L aggregation also require provenance.
5. **Metadata conflict:** six contaminant-database-generation rows have enzyme
   labels inconsistent with `_Try_`/`_Chymo_` filenames and associated search
   names. This is a conflict, not evidence sufficient to rewrite either label.
   It is distinct from the main proteasome digestion endpoint.

These are source-admission constraints tracked in Hitlist #770, not reasons to
replace measurements with plausible inferred biology.

## Independent quantitative replay

We read original serialized tables with local base R 4.5.3, without installing
packages or executing the authors' pipeline. A separate Python implementation
uses the published aSPIRE 1.0 operations as a specification. The reference source
is pinned to tag 1.0, commit`c3d6c9112b2fd88d7e9453b1f23dc338f52c172b`;
MIT source snapshots and license are in reproduce/source/aspire-1.0. This is not
proof that the deposited run used exactly that commit.

For H4, independent arithmetic reconstructs:

- All 27,432 raw peptide/run quantities from 65,712 distinct precursor rows,
  preserving the declared charge/assignment deduplication and background addition.
- Exact original run/time/biological-replicate joins.
- 26,568 filtered quantity entries from 8,856 serialized rows, I/L aggregation,
  control-replicate remapping, global/per-peptide baseline subtraction and
  technical averaging.
- All 13,284 final vector entries, including 444 missing entries and 1,455
  precursor time-point values censored by background.

The global baseline median is 97,886.2119140625 in original MS intensity units.
Maximum relative error is 5.04e-15 for raw quantities and 4.92e-15 for final values.
[Replay receipt](results/h4-arithmetic-replay.json). Runtime 1.98 s.

Scope is explicitly H4 for arithmetic replay. Mapping/synthesis-error removal
has not been independently reproduced, nor has quantitative arithmetic for all
other substrates. All-archive CRC/schema/run-crosswalk verification is broader
than this one-substrate arithmetic proof; the two must not be conflated.

## Consequence for the model experiment

The next model should explain within-peptide production/removal over measured
time, with substrate and enzyme amounts supplied when documented. It should
retain the full fragment pool, including short fragments/nonbinders, while a
separate observation relation handles MS response, background and selection.
No MHC-affinity filter belongs in this source admission.

The existing [kinetic design](../agents/codex/plans/controlled-fragment-kinetics.md)
compares exact expected fragment amounts from independent bond cuts with a
matched discrete partition sampler. Both must represent the same model and
loss. A test of within-peptide temporal shape can reduce a static MS-response
nuisance, but does not identify absolute molar concentrations. The observation
likelihood, any supplied reference-time measurement and nuisance treatment must
be frozen before fitting; finite-sample nonlinear plug-in losses are not
unbiased REINFORCE estimators by default.

Next: implement the original layers and conflicts under Hitlist's shared evidence
interface, replay source derivations, assess biological/technical repeatability,
and freeze grouped substrate evaluation. Keep all Ffh ratios and overlapping
fragments grouped. Do not tune against the previous MAPP test outcomes. This
source uses purified standard proteasomes and does not train APC-specific TAP,
clearance, machinery KO, MHC selection or transferable MS detectability.

## Reproduction and artifact locations

`sh reproduce/launch.sh` verifies/downloads the 18 pinned sources, inventories and
checks all archive CRCs, exports original RData with base R, audits metadata and
replays H4 arithmetic. No environment changes. Python 3.12.6 and local R 4.5.3;
exact source snapshots and configuration are in reproduce/. The initial source
schema registration preceded retrieval; audit code was frozen before execution.
Later metadata reporting adds literal missing-value and conflict counts without
changing source data or performing fitting. Full archive audit 32.69 s. Peak memory
was not recorded; no memory-performance claim. No Modal/GPU use.

Raw assets: `/private/tmp/presto-digestion-assets/`; decompressed condition ZIPs,
selected original layers and derived R exports (decimal text; original RData retained) under
`original-layers/`. The raw files remain outside Git; source-assets.json and
archive-inventory.json freeze hashes/member CRCs/URLs and exact locations.
README summaries, all literal workbook cells, per-archive schema/value counts,
conflict rows, source receipts and independent replay are under results/.
No validation/test predictions are applicable because this is a source audit,
not a predictive evaluation. The next training must consume a registered shared
Hitlist recipe; these experiment scripts are not an alternative curator.
