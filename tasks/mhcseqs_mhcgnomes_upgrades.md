# Upstream `mhcseqs` & `mhcgnomes` Upgrade Tracker for Presto

Last audited: `2026-11-26`

## Current State vs Local Upstream HEAD

| Package | Pinned in `v2.8/v3.0` | Adopted in Codebase (`#165` fix, `pyproject.toml`) | PyPI & Local Upstream HEAD | Gap |
|---|---|---|---|---|
| **`mhcseqs`** | `2.12.3` | **`>=2.17.0` (`2.17.0` installed)** | **`2.17.0`** | **0 (up to date!)** |
| **`mhcgnomes`** | `3.79.2` | **`>=3.84.0` (`3.91.0` installed)** | **`3.91.0`** | **0 (up to date!)** |

> **Verified live (`2026-11-26`, `pirl-unc/presto#165` resolved)**: Upgraded local `shared-virtual-env` and `pyproject.toml` to **`mhcseqs==2.17.0`** & **`mhcgnomes==3.91.0`**. Replaced the hand-coded 58-slot `_CANONICAL_CONTACT_SPECS` in `models/imgt_groove.py` (which mixed mature `DRA`/`DRB1` residue numbers with IMGT coordinates, missing 9 real 3D pocket contacts including `DRB1 β26/β77/β81`, `DRA α24/31/52/53`, and `HLA-A2 W147/A150`) with direct imports of `mhcseqs.IMGT_PAN_CLASS_POCKET_SLOTS` (46 exact 3D union slots), `IMGT_PAN_CLASS_NETMHCPAN_SLOTS` (34 Class I), and `IMGT_PAN_CLASS_NETMHCIIPAN_SLOTS` (34 Class II), with full `legacy58` checkpoint auto-detection and lossless 37-slot warm-start migration (`legacy58 -> mhcseqs_union46`) for `v3.1+`!

---

## High-Impact Upstream Upgrades Available Now

### 1. `mhcseqs` (`v2.7.0` → `v2.12.2`)

| Version | Commit / PR | Feature / Fix | Direct Impact on Presto | Status in Presto |
|---|---|---|---|---|
| **v2.8.0** | `04ac702` (#118) | Delegate allele parsing to `mhcgnomes`, upgraded CLI `lookup` | Cleaner normalization across non-standard IEDB/hitlist restrictions | Not yet adopted (still on `2.7.0`) |
| **v2.9.0–2.9.1** | `c404bbc` (#122), `e11ff45` (#129) | `FLA` (feline) alleles, curated non-IPD allele mapping, cached exact IMGT alignments, **IEDB fragment mutation application (`lookup(allele, mutations=...)`)** | Unlocks mutant MHC binding/elution assays & non-IPD vertebrate alleles | Not yet adopted |
| **v2.10.0** | `6fee08c` (#123) | Multi-clade Class I IMGT structural profile, fixed-index frame, `a3m_seq`, `AlignedResidue` tokens | Exact 3D structural pocket alignment across all Class I alleles & vertebrate clades | Not yet adopted |
| **v2.11.0** | `85d9fc1` (#133) | **Multi-clade Class II (`G-ALPHA`/`G-BETA`) & shared 202-slot pan-class IMGT alignment (`align_groove_to_imgt(...).pan_class_fixed_seq`: 100 HALF1 + 102 HALF2, 22-token alphabet `'-'` vs `'X'`)** | **MASSIVE (`v2.6` live!)**: Fixes 1-aa indel frame shifts across `DQA1*05:01` (`55->61` helix deletion, 85 aa) vs `DQA1*03:01` (86 aa) vs `DRA` (83 aa) vs `DPA1` vs Class I (90 aa); gives Presto exact 3D structural pocket coordinates across ALL 202 groove positions pan-class | **Implemented & launched in `Presto-Omni v2.6` (`models/imgt_groove.py`, `mhcseqs==2.12.2`, `mhcgnomes==3.79.0`, `15/15` tests passing!)** |
| **v2.12.0** | `172bef2` (#135) | Adopt `mhcgnomes>=3.75` `Species.clade`/`ncbi_taxid`, filter non-MHC `~ref` rows | Clean vertebrate clade provenance + eliminates non-MHC contamination | Not yet adopted |
| **v2.12.1** | `a7e3d47` (PR #139 / #137) | Filtered 26 leaking IMGT/IPD `TAP1`/`TAP2` rows (`pipeline.py`) + `HMHA1` (`domain_grammar.py`); unwrapped `~ortho:`/`~ref:` via `_parse_diverse_gene_class()`; sanitized `? -> X` in `_iter_fasta()`; deduped `b2m_sequences.csv` (`357 -> 278` species, removed 4 >200 aa non-B2M); fixed 23 mislabeled Class II rows (`Q31152`, `Q9TQA7`, SwissProt) in `diverse_mhc_sequences.csv`; dropped 11 `<70 aa` fragments from `genbank_saha_sequences.csv` | Cleaner Class II & non-human groove database (`mhc-full-seqs.csv`) with zero TAP/HMHA1/B2M contamination | Adopted in `v2.6` runtime (`mhcseqs==2.12.3`); rebuild cached `mhc-full-seqs.csv` for `v2.7` |
| **v2.12.2** | `5819737` (PR #140 / #138) | Rejected C1-set Ig Cys pairs (`domain_type == "ig"`) as Class I `alpha2_cys` primary anchors (`domain_parsing.py`, eliminating `355/367` Ig-as-groove2 false positives); split N-truncated Class I alleles (`EquusCaballus-1*010:01`, `EquusCaballus-2*003:04`, `SaguinusOedipus-F*03:01`) into `alpha2_only` (`g2=92, ig=95`); rejected runaway `groove1 > 120 aa` and downgraded both-half-truncated (`<70 aa`) fragments to `short` | Prevents corrupted Class I α2 groove domains on partial/divergent/equine alleles | Adopted in `v2.6` runtime (`mhcseqs==2.12.3`); rebuild non-human caches for `v2.7` |
| **v2.12.3** | `e6ce139` (PR #142 / #141) | Required downstream α3 Ig pair before `_class_i_fragment_result()` on `len > 200` sequences (fixing `A0A8C8TC05` `1,077 aa` groove2); bumped `mhcgnomes>=3.78.1` (removed temporary `DAA-1` workaround now that `mhcgnomes#286` shipped); mapped UniProt `Q31145 -> H2-D*q` and `Q31151 -> H2-L*q` in `curated_uniprot_alleles.csv` | Unlocks mouse `H2-D*q` / `H2-L*q` Class I rows & tightens partial-fragment guardrails | **Adopted in `v2.6–v3.0`** |
| **v2.16.0** | PR #160 / #154 | Audited & fixed catalog outliers, runaway fragments, non-MHC contaminants, and PDB benchmarks | Cleaner `mhc-full-seqs.csv` groove database across all 36,693 `ok` grooves | **Adopted locally (`2.17.0`); pin for `v3.1` Colab** |
| **v2.17.0** | PR #161 / #155 | **Verified 3D-aligned pan-class IMGT pocket exports**: `IMGT_PAN_CLASS_POCKET_SLOTS` (46 exact NetMHCpan-34 ∪ NetMHCIIpan-34 slots: 22 shared + 12 Class I-only + 12 Class II-only), `IMGT_PAN_CLASS_NETMHCPAN_SLOTS` (34), `IMGT_PAN_CLASS_NETMHCIIPAN_SLOTS` (34) | **CRITICAL (`pirl-unc/presto#165`)**: Recovers 9 missing 3D pocket contacts (`DRA α24/31/52/53`, `DRB1 β26/β77/β81`, `HLA-A2 W147/A150`) and eliminates bogus `HALF1:58` gap & `HALF2:11` disulfide slots in `IMGTPocketCrossInteraction` and `DirectPocketBilinearHead` | **Adopted in `models/imgt_groove.py`, `models/presto.py`, `training/modular.py`, `pyproject.toml` (`21/21` pytest passing)** |

### 2. `mhcgnomes` (`v3.72.0` → `v3.79.2`)

| Version | Commit / PR | Feature / Fix | Direct Impact on Presto | Status in Presto |
|---|---|---|---|---|
| **v3.73.0** | `af18d90` (#274) | Genus/trinomial CamelCase aliases, case-insensitive decorated names, `Species` hierarchy helpers | Removes brittle manual species alias tables in Presto data builders | Adopted in `v2.6` |
| **v3.74.0** | `03cec18` (#202) | **Distinguish domestic dog (`Canis lupus familiaris` / `DLA`) from wolf** | Eliminates manual `CanisLupus -> CanisSp` workaround in `data/hitlist_binding_expansion.py:FastGrooveResolver` | Adopted in `v2.6` (cleanup TODO) |
| **v3.75.0–3.76.0** | `eaf6c23` (#276), `7d9d46a` (#277) | Expose `Species.clade` & `ncbi_taxid`; add 66 missing vertebrate species from `mhcseqs` | Enables clade-aware pan-vertebrate transfer & clean species provenance | Adopted in `v2.6` |
| **v3.77.0** | `a4c813b` (#282) | **Fix `H2-IEg7` haplotype allele and non-human MHC gene/prefix gaps** | Unlocks previously dropped mouse Class II haplotypes (NOD `H2-g7` / `H2-d` gaps!) | Adopted in `v2.6` |
| **v3.77.1–3.77.3** | `1629c41`, `b50a605` | Fix slash-pair tokenization, chain-selector, `Result`/`Pair`/`Haplotype` bugs | Cleaner Class II α/β slash-pair parsing (`DQA1/DQB1`, `DPA1/DPB1`, `H2-AA/AB`) | Adopted in `v2.6` |
| **v3.78.0–3.78.1** | `dc8954c`, `309a588` (#286) | Promote reusable `function_api` helpers; fix Class II chain inference for hyphenated alpha genes (`DAA-1`/`DAA-2`) | Cleaner non-human Class II alpha/beta chain resolution | Adopted in `v2.6` |
| **v3.79.0** | `5fbd80f` (#290) | Dedup record fields and rename CLI gene column to `gene_name` | Cleaner structured metadata output | Adopted in `v2.6` |
| **v3.79.1** | `987e972` (#291) | Fix subspecies tree placement, umbrella `old_mhc_prefix` inheritance, and `Malo-B` class | Cleaner non-human primate / subspecies resolution | Adopted in `v2.6` |
| **v3.79.2** | `b872ed2` (#292) | **Fix `Pair`/`Haplotype` class metadata, `H2-IA` locus collision, and Unicode dashes** | Fixes mouse `H2-IA` Class II locus collision & Class II `Pair`/`Haplotype` class metadata + Unicode dash alleles in IEDB/literature | **Adopted in `v2.6` (`mhcgnomes==3.79.2`)** |

---

## Presto Integration Roadmap (`v2.6+`)

1. **Bump pins to `mhcseqs>=2.12.1` (or `2.12.2` once merged) & `mhcgnomes>=3.78.1`** in Colab runners (`runplz_colab_omni_v26.py`) and regenerate clean `mhc-full-seqs.csv` + non-human MS/binding groove caches.
2. **Add `imgt_structural_groove: bool = False` (`Presto-Omni v2.6`)**:
   - Precompute & cache `mhcseqs.align_groove_to_imgt(groove1, groove2, mhc_class).pan_class_fixed_seq` (202-slot pan-class IMGT alignment) per unique `(groove1, groove2, mhc_class)`.
   - Add a zero-init learned 202-slot **IMGT structural position embedding** + gap/unknown token (`'-'` / `'X'`) residual into `_groove_states(...)` (100% backward-compatible warm-start via `allow_interaction_expansion=True`!).
   - Directly eliminates indel frame shifts (e.g., `DQA1*05:01` 85 aa vs `DQA1*03:01` 86 aa vs `DR` 83 aa vs `DP` vs Class I 90 aa) and gives exact 3D pocket homology across all Class I, Class II, and non-human alleles.
3. **Remove obsolete workarounds** once bumped:
   - Simplify `FastGrooveResolver` (`CanisLupus -> CanisSp` alias, manual `H2-EA/EB` mapping) to delegate cleanly to `mhcseqs>=2.12.1` + `mhcgnomes>=3.78.1`.
   - Re-audit `H2-d class II` (`H2-Eb1*d`) and `H2-g7` coverage in `mhcseqs>=2.12.1` to recover dropped mouse Class II MS/binding rows.

## Open Issues / Ideas to Feed Upstream to `mhcseqs` / `mhcgnomes`
- **Confirmed still missing (`2026-11-25`)**: `H2-Eb1*d` is still absent from `mhc-full-seqs.csv` (while `H2-Ea*d` is present at `g1=83`), which still blocks full `H2-d class II` (`H2-IE^d`) haplotype expansion in `data/hitlist_nonhuman_ms_expansion.py`.
- Re-export & upload a fresh `mhcseqs 2.12.3` `mhc-full-seqs.csv` to Modal Volume (`presto-checkpoints`) so `v2.7` data builders pick up `H2-D*q` (`Q31145`), `H2-L*q` (`Q31151`), `355/367` Ig-as-groove2 fixes, and `TAP1/TAP2`/`HMHA1` cleanup.
- Verify `align_groove_to_imgt` throughput/caching on all ~400 unique alleles in Presto's training + validation corpora (and confirm zero warnings/failures on `HLA-E/F/G`, `DQA1*05:01/DQB1*03:03`, `DRB3/4/5`, `H2`, `Mamu`, `DLA`, `Eqca`, `BF`).
