# Unresolved MHC Edge Cases (Strict Resolution Triage)

Snapshot source: `artifacts/analysis/strict_mhc_smoke_20260228/unresolved_mhc_alleles.csv`

Canonical policy:
- unresolved training alleles are hard errors (`--strict-mhc-resolution`, default true),
- no allele-string token fallback is allowed.

## Current unresolved buckets

The unresolved report now emits a deterministic `category` generated via
`mhcgnomes.parse(...)` (with a fallback regex path when parsing fails).

| Category | Example tokens | Why unresolved today | Recommended handling |
|---|---|---|---|
| `murine_allele_missing_sequence` / `murine_haplotype` | `H2-K*b`, `H2-D*b`, `H2-K*k`, `H2-b class I` | Allele/haplotype parsed, but sequence not present in current index | Keep as error unless mapped to explicit sequence upstream |
| `murine_pair_shorthand` | `H2-AA*b/AB*b`, `H2-AA*k/AB*k` | Represents alpha/beta pair shorthand, not one explicit chain allele | Keep as error; expand into explicit chain alleles during preprocessing |
| `human_serotype` | `HLA-A68`, `HLA-B27`, `HLA-DR1`, `HLA-DR15` | Serotype/family notation maps to multiple distinct sequences | Keep as error unless deterministic project-level mapping is adopted |
| `human_locus` | `HLA-DR` | Too coarse to identify one sequence | Keep as error; require specific allele |
| `human_nonclassical_gene` | `HLA-BTN3A1` | Not part of current canonical pMHC index target set | Keep as error or move to separate non-classical pipeline |

## Top counts in smoke run

- `H2-AA*b/AB*b`: 43
- `H2-K*b`: 28
- `H2-D*b`: 16
- `H2-b class I`: 16
- `HLA-DR`: 14
- `H2-K*k`: 11
- `HLA-A68`: 7
- `HLA-DR1`: 6

## Next step for allowlisting

Do not add broad regex allowlists. If exceptions are needed, maintain a
small explicit mapping file from unresolved source token -> deterministic
resolved sequence(s), with source-level provenance and tests.
