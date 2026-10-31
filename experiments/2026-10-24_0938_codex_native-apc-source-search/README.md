# Native APC peptide-source search

Registered October24,2026 by Codex/GPT-6, before downloading the PeptidomicsDB
metadata workbooks. Execute the frozen source-qualification plan in
`reproduce/plan.md`; prospective priority is non-infected human/mouse APC material
with native intracellular peptide extraction and quantities. Database membership,
protein digest measurements and MHC-eluted ligands do not establish this endpoint.

Primary resource DOI10.1093/nar/gkag1032, published October23,2026. Official public
download page exposes project, sample and dataset metadata workbooks as well as
per-sample peptide CSVs. Only metadata are requested initially. The downloaded
HTML inventory and receipts from discovery are retained under the preceding
0924 audit's raw-data folder and will be linked explicitly in the results.

No biological model, pretraining, synthetic data, losses, predictions or validation/
test split is involved in this source search. No quantities are admitted to training
before primary-source and shared Hitlist qualification. Existing biological
baselines and reserved HAP1 outer outcomes are unchanged.

## Outcome

The public metadata are accessible and reproducible, but this screen has not
qualified a native APC free-peptide cohort. It identifies a material metadata
hazard: a whole-cell sample can be labeled intracellular even when the assay
purifies HLA complexes. Training admission must establish the measured population
from the original study rather than infer it from the database category.

| Retrieved metadata | Observed records |
|---|---:|
| Projects | 151 |
| Sample rows | 3,992 |
| Unique raw-file names | 3,983 |
| Pairwise dataset rows | 249 |
| Context-specific dataset rows | 14 |
| Projects matching the initial immune-cell term screen | 29 |
| Human/mouse projects without an MHC category label | 35 |

Sample rows match the project-level declared sample counts and have unique
PeptidomicsDB sample identifiers. These checks do not establish biological
independence. Raw-file names are not globally unique. All values and identifiers
are preserved without case normalization, missing-value filling or collapsing
sample and acquisition identities.

## Eligibility findings

- The fixed initial immune-cell term screen finds 29 projects. Their project
  categories all include MHC immunopeptidomics. Matches can come from background
  descriptions, so even a term match does not establish the sample's cell identity.
  This lexical screen is not an exhaustive cell-line ontology.
- Twenty samples have the single category `Intracellular peptidomics`; all belong
  to PPD016/PXD033369, enteroendocrine cells. These may be useful for another
  processing question but do not establish APC availability.
- PPD017/PXD021177 includes JY whole-cell sample PPD0170004 with both MHC-I and
  intracellular tags. The original study explicitly purifies HLA complexes from
  whole-cell and extracellular-vesicle lysates. It measures selected HLA ligands,
  not the free intracellular peptide pool. [Primary source](https://pubmed.ncbi.nlm.nih.gov/34211107/).
- Manual review of the 35 generic human/mouse non-MHC-label projects identifies
  PPD070/PXD009801, which the initial term screen misses because its tissue labels
  are U937, RPMI and JY cell lines. Despite its generic `Peptidomics` category,
  the primary study concerns dynamic isotope labeling of HLA-I ligands and source
  proteomes. It is not admitted as free-peptide data. [Primary source](https://pubmed.ncbi.nlm.nih.gov/24715725/).
- `Enzyme=None` is retained as a source/search-setting field. It is not evidence
  of native extraction or absence of laboratory digestion. Project-wide categories
  can also combine distinct assay populations and must not replace sample-level
  capture/preparation evidence.

No per-sample peptide CSV, quantitative training label or new biological fit was
admitted. This is a negative source-eligibility result with reusable metadata,
not proof that no suitable APC dataset exists elsewhere or in unreviewed generic
projects. The database's broad coverage remains useful for source discovery.

## Integrity, reproduction and limitations

The three original workbooks total 468,673 bytes. Exact URLs, sizes, SHA256 hashes
and retrieval receipts are linked in `results/inventory-v1/inventory.json`. The
raw assets remain under `data/native_apc_source_search/2026-10-24_0938_peptidomicsdb`.
The source is [PeptidomicsDB](http://www.inbirg.com/ppd/download/) and its
[primary resource article](https://doi.org/10.1093/nar/gkag1032).

The registered script preserves all source metadata columns and records the search
regex. No production source parser or training recipe was added to Presto. The
inventory takes 0.255 seconds with 51.7 MiB peak RSS. An independent OOXML reader
compares every nonempty value against openpyxl: all 63,761 values match, no formulas
are present, and blank source payloads remain separate from zero. Verification
takes 0.559 seconds with 90.1 MiB peak RSS. It verifies source fidelity, not biological
replicate independence or the correctness of the database's annotation choices.

`reproduce/launch.sh`, `launch.json`, `plan.md` and frozen sources preserve the
commands and prospectively declared search. Local bundled Python3.12/openpyxl3.1.5;
no environment overrides, installations, Modal or GPU. Raw sources were not
modified, recalculated or exported. No synthetic labels, pretraining, optimization,
loss terms, assay-output mapping, validation/test metrics or prediction dumps apply
because this is source search before biological admission. Reserved HAP1 outcomes
and all previous practical baselines remain unchanged.

Status: metadata search and fidelity verification closed; APC source admission open.
Shared curation owner: [Hitlist811](https://github.com/pirl-unc/hitlist/issues/811).
Next: shared Hitlist curation must distinguish capture from specimen compartment,
preserve reanalysis identification provenance, and review original studies before
using candidate quantities. Continue searching for independently measured native
APC peptides and substrate/product constraints. Full availability goal stays active.
