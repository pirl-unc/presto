# Native intracellular PSMD2-knockdown quantities recovered

Codex / GPT-6, October 25, 2026. Source audit registered before quantitative
retrieval at Presto `8454ef9c40913c89d56bca5cd02f923701af91f6`, with recorded task
dirt. [Frozen plan](reproduce/plan.md), [registration/environment](reproduce/launch.json),
[executed commands](reproduce/executed-commands.json), [closure](results/source-closure.json).

**A new native intracellular quantity source is available:** Sahu2021 Dataset7
contains 10,647 distinct peptide sequences and six sequence-level intensity
columns for control versus PSMD2/Rpn1-knockdown HEK293T cells. This advances the
next biological-data choice beyond a one-substrate purified digest. It is not
professional-APC data or a calibrated concentration endpoint, and no model was
trained during this source audit.

## Source identity and measurement scope

Primary study: [Sahu et al., Nature Communications2021](https://doi.org/10.1038/s41467-021-26427-0),
[repository article](https://pmc.ncbi.nlm.nih.gov/articles/PMC8548400/),
[PXD018722](https://www.ebi.ac.uk/pride/archive/projects/PXD018722).
The study distinguishes intracellular peptide extraction from proteasome-trapped
peptides, purified-substrate reactions and tryptic proteasome composition. The
recovered Dataset7 is the control/Hi20S intracellular panel. Whole-cell hot-water
extraction is described in the primary methods; source genotype is inducible
PSMD2 knockdown, not a complete knockout or a measured numerical catalytic loss.

Dataset7 is `41467_2021_26427_MOESM10_ESM.xlsx`, SHA256
`f29811e09144f6d526ce1f2955681bb5f07a7ed0a4957e1a9f596740fc492ea5`,
`peptides` sheet, 10,647 data rows and 35 columns. All protein alternatives,
source PEP, decoy/contaminant flags, identification type and source zeros remain.
There are 6,202 literal protein-accession tokens before reference resolution or
curation. This is not a count of independently measured parents or training groups.

| Original sample label | Positive signal cells | Source zero cells |
|---|---:|---:|
| Cont1 | 3,866 | 6,781 |
| Cont2 | 4,181 | 6,466 |
| Cont3 | 4,043 | 6,604 |
| KD1 | 7,418 | 3,229 |
| KD2 | 6,968 | 3,679 |
| KD3 | 7,560 | 3,087 |

The 34,036 positives and 29,846 source zeros are measurement cells, not biological
replicates. No amount means, condition effects or predictive outcomes were computed.
A zero does not establish biological absence or an assay-specific censor bound.
Numbered labels do not establish biological independence or pair Cont1 with KD1.
The source column is `Intensity`, not `LFQ intensity`; enabled LFQ configuration
does not justify renaming the exported values. These sequence-level quantities do
not identify unmodified peptidoform abundance, precursor-normalized availability,
extraction recovery or molarity. No matched precursor quantity has been admitted.

## Original run and search metadata

The official project inventory contains 48 unique file records; the next page is
empty. No raw spectra were downloaded. The deposited 156-byte design table and
22,385-byte `Hi20S_mqpar.xml` match official sizes and SHA1 checksums. All six
labels agree independently between the design and configuration and join to
original RAW file identifiers. [Exact maps and settings](results/run-search-context.json).

The search specifies lengths **8–25**, matching the observed range, while the
article broadly describes minimum length7. Preserve both assertions and use the
actual configuration for this panel. Products shorter than8 are unsearched, not
negative examples. Matching between runs is enabled, with its FDR option false;
retain per-sample identification types and do not treat every matched signal as
an independent MS/MS identification. Peptide/protein FDR settings are .01; these
are distinct from row-level PEP. Variable modifications and original FASTA identity
are retained. The deposited reference is available as a33,003,906-byte file but
has not been downloaded under this audit's20MiB per-asset cap.

## Fidelity and explicit defects

Two original workbooks were recovered: Dataset7 above and Dataset6, a branched
peptide-identification panel (34 data rows across six sheets). Dataset6's scores,
masses and localization evidence are not an abundance trajectory. Independent
openpyxl and OOXML readers account for **293,009 populated cells**, preserving
original coordinates, numbers, strings and styles; there are no formulas.

The initial inventory correctly stopped on **13 date-shaped cells in `Gene names`**.
The original workbook stores numeric Excel serials with date style17. Independent
serial/epoch reconstruction matches the rendered dates, and every other cell
matches directly. This verifies the stored representation while flagging an
unusable gene annotation. No gene names were reconstructed from dates and no
original values were rewritten. The failed inventory and
[pre-verification amendment](reproduce/date-field-amendment.md) remain.
[Independent recovery/date verification](results/independent-recovery-verification.json).

The publisher served client challenges. A public EuropePMC supplement ZIP crossed
the unchanged20MiB cap, leaving a preserved partial stream. Its first12complete
members were decoded using local records and original data descriptors; only the
two complete XLSX files were extracted. A separately rebuilt ZIP index and Python's
ZIP reader independently verify every complete member's CRC/bytes, followed by
both workbooks' inner ZIP CRCs. The next, large Source Data workbook is truncated
and was not admitted. The incomplete outer central directory and later supplements
were not verified. [Offsets, lengths and checksums](results/member-recovery.json),
[retrieval amendment](reproduce/retrieval-amendment.md).

## Complementary kinetics candidate and access limits

[Liepe et al., eLife2015](https://elifesciences.org/articles/07545) links
[Dryad10.5061/dryad.nk453](https://datadryad.org/dataset/doi:10.5061/dryad.nk453).
The official API identifies a61,018-byte `Data.xlsx` for three fluorogenic
substrates and mouse/human standard/immunoproteasomes. Its public-page download
returned403 and the API download401. No numeric workbook was admitted and no
access control was bypassed. Fluorophore release would be a distinct kinetic
constraint, not a natural fragment-distribution label. This candidate is deferred;
the native human source is the immediate next work.

Two small PRIDE metadata reads were refused when shared free space briefly fell
below the fixed reserve; both failed/empty bodies remain. A fresh bounded retry
on the verified route succeeded after capacity recovered. The fetch helper now
returns a failure exit code on failed retrieval; the original executed helper is
preserved. An accidental trailing registration command ran from the wrong
working directory and failed before any source work; its log is retained.

## Reproduction, resources and next action

Use the registered Python3.12.6 environment without installation changes.
openpyxl3.1.5 and numpy2.5.4 are recorded. Retrieval receipts freeze every exact
URL/name/command, access status, source hash and time. Source snapshots include
both fetch versions. Launch source stages via `sh reproduce/launch.sh STAGE`.
They refuse existing outputs; replay in a fresh evidence directory rather than
replacing failures or prior receipts. No Modal/GPU, pretraining, fitting, synthetic
biological labels, model losses or assay-head updates occurred. Test/validation
predictions and regression/binder metrics are deliberately absent because source
qualification precedes canonical admission and predictive registration.

Workbook inventory:35.03s,315,588,608B peak RSS; member recovery:.047s;
independent recovery:.059s; support inventory:2.03s,55,705,600B; closure:.89s,
69,926,912B. These are single-run local observations, not performance comparisons.
Per-request30s/20MiB and total64MiB download limits remain; the archive cap failure
is preserved. Keep the1GiB free-space reserve plus32MiB margin. Raw sources and
literal-cell dump exist once under
`data/proteasome_quantity_sources/2026-10-25_0344_codex_proteasome-quantity-source/`.
[All asset hashes/paths and retrievals](results/source-assets.json),
[schema/support](results/panel-support.json), [support CSV](results/panel-support.csv).

**Status:** source fidelity and sample/search mapping qualified with explicit
exceptions; shared canonical training admission is tracked in
[Hitlist837](https://github.com/pirl-unc/hitlist/issues/837). **Next:** add this
native panel to Hitlist's common evidence system, verify the original reference
and aggregation, preserve all parent alternatives and sample dependencies, then
preregister a simple sequence control and matched processing comparison. Do not
substitute peptides-derived protein groups for independent precursor abundance.
**Open questions:** preparation/replicate independence, exact reporter/exposure
context, raw-to-published quantity aggregation, precursor pairing, calibration
and transfer to APCs. The full precursor-relative APC availability goal remains
active and unachieved; there is no new RL or biological prediction result here.
