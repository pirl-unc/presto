"""Leak-free non-human MS elution expansion builder for Presto-Omni.

Builds a strictly vetted, zero-regression non-human MS eluted-ligand training
slice from hitlist `observations.parquet` + matched species reference proteomes:

  - Admits ONLY non-human classical Class I / Class II MS ligand-presentation
    positives (`has_ptm == False`, `is_non_peptide_ligand == False`,
    `is_potential_contaminant != True`, `restriction_evidence != "predicted"`).
  - Never fabricates alleles for class-only rows (`Saha class I`, `H2 class I/II`,
    `SLA class I`, `B19 class I`) or partially resolved haplotypes (`H2-d class II`,
    missing `H2-Eb1*d` in `mhc-full-seqs.csv`). Optionally expands genetically
    exact C57BL/6 `H2-b` haplotypes (`H2-b class I -> (H2-K*b, H2-D*b)` and
    `H2-b class II -> (H2-AA*b/AB*b,)` because `H2-Ea` is a deletion pseudogene
    in `H2-b`).
  - Passes EVERY admitted positive AND EVERY sampled negative decoy through
    `ProtectedIdentityIndex` (exact + 8-mer + equal-length Hamming-1 against
    `sealed-identity-embargo.txt.gz` and all validation JSONLs; `test_accessed=False`).
  - Excludes ALL known MS-observed peptides (human + non-human) from negative
    decoy sampling.
  - Maps positives per species against the exact host species proteome via
    `presto.data.presentation.map_peptides` (Aho-Corasick), extracts unanimous
    15-AA flanks via `source_junction`, samples deterministic length-matched
    same-possible-source negatives via `background_candidate`, and applies
    symmetric missing-junction masking on ambiguous pairs.
  - Emits `Context(source_n, source_c, n_terminus, c_terminus)` ONLY (never
    asserting human `cell_type`, human TPMs, `IFNG`, `ERAP2`, or interventions),
    preserving zero regression on human APC/perturbation/Stopfer counterfactuals.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from presto.data.hitlist_binding_expansion import (
    AA_SET,
    FastGrooveResolver,
    ProtectedIdentityIndex,
    _clean_peptide,
)
from presto.data.modular import Context, MHCInput, Observation, SourceJunction
from presto.data.presentation import (
    Protein,
    background_candidate,
    junction,
    map_peptides,
    protein_families,
    source_junction,
    stable_digest,
)

SPECIES_CANONICAL_MAP: dict[str, str] = {
    "mus musculus": "mus_musculus",
    "canis lupus": "canis_lupus",
    "canis lupus familiaris": "canis_lupus",
    "canis sp.": "canis_lupus",
    "macaca mulatta": "macaca_mulatta",
    "equus caballus": "equus_caballus",
    "gallus gallus": "gallus_gallus",
    "sus scrofa": "sus_scrofa",
    "sus sp.": "sus_scrofa",
    "bos taurus": "bos_taurus",
    "bos sp.": "bos_taurus",
}

# Genetically exact, complete C57BL/6 H2-b haplotypes (no omitted loci):
# - H2-b Class I: expresses only H2-Kb and H2-Db (no H2-L locus in H2-b).
# - H2-b Class II: expresses only I-Ab (H2-Ea has a 627-bp promoter/exon-1
#   deletion pseudogene in H2-b mice).
EXACT_HAPLOTYPE_CANDIDATES: dict[tuple[str, str, str], tuple[str, ...]] = {
    ("mus_musculus", "I", "h2-b class i"): ("H2-K*b", "H2-D*b"),
    ("mus_musculus", "II", "h2-b class ii"): ("H2-AA*b/AB*b",),
}

_ENSEMBL_GENE_RE = re.compile(r"\bgene:([^\s]+)")
_UNIPROT_GN_RE = re.compile(r"\bGN=([^\s]+)")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def canonical_species_slug(species: Any) -> str | None:
    """Normalize a hitlist `mhc_species` string to a canonical proteome slug."""
    if not isinstance(species, str):
        return None
    key = species.strip().lower()
    if not key or key == "homo sapiens":
        return None
    return SPECIES_CANONICAL_MAP.get(key)


def _allele_slug(tag: str) -> str:
    return _SLUG_RE.sub("_", tag.strip().lower()).strip("_") or "unknown"


def read_proteome_fasta_with_reviewed(path: Path) -> tuple[dict[str, Protein], set[str]]:
    """Read an Ensembl or UniProt proteome FASTA, also returning Swiss-Prot (`sp|`) IDs."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Proteome FASTA not found: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    proteins: dict[str, Protein] = {}
    reviewed_ids: set[str] = set()
    header: str | None = None
    chunks: list[str] = []

    def _finish() -> None:
        if header is None:
            return
        seq = "".join(chunks).strip().upper().rstrip("*")
        if not seq:
            return
        first_tok = header.split()[0]
        is_reviewed = False
        if "|" in first_tok:
            parts = first_tok.split("|")
            is_reviewed = parts[0].strip().lower() == "sp"
            prot_id = parts[1].strip() if len(parts) >= 2 and parts[1].strip() else first_tok
        else:
            prot_id = first_tok
        m_ens = _ENSEMBL_GENE_RE.search(header)
        m_gn = _UNIPROT_GN_RE.search(header)
        if m_ens:
            gene = m_ens.group(1)
        elif m_gn:
            gene = m_gn.group(1)
        else:
            gene = prot_id
        # Keep first deterministic occurrence if duplicate accession appears
        if prot_id not in proteins:
            proteins[prot_id] = Protein(prot_id, gene, seq)
            if is_reviewed:
                reviewed_ids.add(prot_id)

    with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                _finish()
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
    _finish()
    if not proteins:
        raise ValueError(f"Empty proteome FASTA: {path}")
    return proteins, reviewed_ids


def read_proteome_fasta(path: Path) -> dict[str, Protein]:
    """Read an Ensembl or UniProt reference proteome FASTA (plain or .gz)."""
    proteins, _ = read_proteome_fasta_with_reviewed(path)
    return proteins


def load_canonical_protein_ids(peptide_mappings_parquet: Path) -> dict[str, set[str]]:
    """Extract per-species canonical protein IDs (with & without version) from peptide_mappings."""
    import pandas as pd

    df = pd.read_parquet(
        peptide_mappings_parquet,
        columns=["proteome", "protein_id", "is_canonical_transcript"],
    )
    canon = df[df["is_canonical_transcript"] == True]  # noqa: E712
    by_species: dict[str, set[str]] = defaultdict(set)
    for prot_sp, pid in zip(canon["proteome"].tolist(), canon["protein_id"].tolist()):
        slug = canonical_species_slug(prot_sp)
        if slug is None or not isinstance(pid, str) or not pid.strip():
            continue
        tok = pid.strip()
        by_species[slug].add(tok)
        by_species[slug].add(tok.split(".", 1)[0])
    return dict(by_species)


def resolve_source_junction(
    peptide: str,
    occurrences: list[tuple[str, int]],
    proteins: Mapping[str, Protein],
    *,
    canonical_ids: set[str] | frozenset[str] | None = None,
    reviewed_ids: set[str] | frozenset[str] | None = None,
    flank_length: int = 15,
) -> tuple[dict[str, Any], str, list[tuple[str, int]]]:
    """Resolve source flanks: unanimous -> within-gene canonical/reviewed -> masked.

    Never guesses across different genes (cross-gene paralogs stay conservatively
    masked unless resolved externally by expression). Returns
    ``(junction_context, policy_label, preferred_occurrences_for_background)``.
    """
    ctx, policy = source_junction(peptide, occurrences, proteins)
    if policy == "unanimous_reference_junction":
        return ctx, policy, occurrences

    masked = {
        "source_n": None,
        "source_c": None,
        "n_terminus": None,
        "c_terminus": None,
    }
    genes = {
        proteins[pid].gene.split(".", 1)[0]
        for pid, _ in occurrences
        if pid in proteins and proteins[pid].gene
    }
    if len(genes) != 1:
        return masked, "ambiguous_cross_gene_junction", occurrences

    # Single gene: try explicit canonical transcript first
    if canonical_ids:
        canon_occs = [
            (pid, st)
            for pid, st in occurrences
            if pid in canonical_ids or pid.split(".", 1)[0] in canonical_ids
        ]
        if canon_occs:
            juncs = [
                junction(proteins[pid].sequence, st, len(peptide), flank_length=flank_length)
                for pid, st in canon_occs
            ]
            if all(j == juncs[0] for j in juncs):
                return juncs[0], "within_gene_canonical_junction", canon_occs
            return masked, "ambiguous_within_gene_junction", occurrences

    # Single gene in UniProt proteome: prefer reviewed Swiss-Prot (`sp|`) over TrEMBL (`tr|`)
    if reviewed_ids:
        rev_occs = [(pid, st) for pid, st in occurrences if pid in reviewed_ids]
        if rev_occs:
            juncs = [
                junction(proteins[pid].sequence, st, len(peptide), flank_length=flank_length)
                for pid, st in rev_occs
            ]
            if all(j == juncs[0] for j in juncs):
                return juncs[0], "within_gene_reviewed_junction", rev_occs

    return masked, "ambiguous_within_gene_junction", occurrences


def collect_candidate_junctions(
    peptide: str,
    occurrences: list[tuple[str, int]],
    proteins: Mapping[str, Protein],
    *,
    max_candidates: int = 5,
    flank_length: int = 15,
) -> tuple[SourceJunction, ...]:
    """Return deduplicated valid SourceJunction options when 2 <= M <= max_candidates."""
    if max_candidates < 2:
        return ()
    seen: dict[tuple[str | None, str | None, bool | None, bool | None], SourceJunction] = {}
    for pid, st in occurrences:
        j = junction(proteins[pid].sequence, st, len(peptide), flank_length=flank_length)
        if j["source_n"] is None or j["source_c"] is None:
            continue
        key = (j["source_n"], j["source_c"], j["n_terminus"], j["c_terminus"])
        if key not in seen:
            seen[key] = SourceJunction(
                source_n=j["source_n"],
                source_c=j["source_c"],
                n_terminus=j["n_terminus"],
                c_terminus=j["c_terminus"],
            )
            if len(seen) > max_candidates:
                return ()
    if len(seen) < 2:
        return ()
    return tuple(
        seen[k]
        for k in sorted(seen, key=lambda t: (t[0] or "", t[1] or "", bool(t[2]), bool(t[3])))
    )


def discover_local_proteomes(
    hitlist_dir: Path,
    pyensembl_dir: Path | None = None,
) -> dict[str, Path]:
    """Locate available non-human reference proteomes from local hitlist & pyensembl caches."""
    out: dict[str, Path] = {}
    prot_dir = Path(hitlist_dir) / "proteomes"
    if prot_dir.exists():
        for slug, fname in (
            ("canis_lupus", "canis_lupus.fasta"),
            ("macaca_mulatta", "macaca_mulatta.fasta"),
            ("equus_caballus", "equus_caballus.fasta"),
            ("gallus_gallus", "gallus_gallus.fasta"),
            ("sus_scrofa", "sus_scrofa.fasta"),
            ("bos_taurus", "bos_taurus.fasta"),
            ("mus_musculus", "mus_musculus.fasta"),
            ("mus_musculus", "Mus_musculus.GRCm39.pep.all.fa.gz"),
        ):
            cand = prot_dir / fname
            if cand.exists() and slug not in out:
                out[slug] = cand
    if "mus_musculus" not in out:
        candidates = []
        if pyensembl_dir is not None:
            candidates.append(Path(pyensembl_dir))
        candidates.extend(
            [
                Path.home() / "Library/Caches/pyensembl/GRCm39/ensembl112",
                Path.home() / ".cache/pyensembl/GRCm39/ensembl112",
            ]
        )
        for d in candidates:
            cand = d / "Mus_musculus.GRCm39.pep.all.fa.gz"
            if cand.exists():
                out["mus_musculus"] = cand
                break
    return out


def resolve_nonhuman_mhc_candidates(
    raw_allele: str,
    species_slug: str,
    raw_species: str,
    mhc_class: str,
    resolver: FastGrooveResolver,
    *,
    allow_exact_haplotypes: bool = True,
) -> tuple[tuple[MHCInput, ...], str, str] | None:
    """Resolve an allele string or exact haplotype to validated MHCInput candidates.

    Returns ``(mhc_tuple, canonical_tag, resolution_mode)`` or ``None`` if any
    candidate fails exact groove resolution (never drops part of a multiallelic bag).
    """
    al_str = str(raw_allele or "").strip()
    if not al_str or mhc_class not in ("I", "II"):
        return None
    low = al_str.lower()
    if "mutant" in low:
        return None

    hap_key = (species_slug, mhc_class, low)
    if allow_exact_haplotypes and hap_key in EXACT_HAPLOTYPE_CANDIDATES:
        tokens = EXACT_HAPLOTYPE_CANDIDATES[hap_key]
        mode = "exact_haplotype"
    else:
        if "class i" in low or "class ii" in low:
            return None
        sep = ";" if ";" in al_str else ","
        tokens = tuple(dict.fromkeys(part.strip() for part in al_str.split(sep) if part.strip()))
        mode = "exact_allele"

    if not tokens:
        return None

    by_groove: dict[tuple[str, str], list[str]] = defaultdict(list)
    for tok in tokens:
        grooves = resolver.resolve(tok, mhc_class)
        if grooves is None:
            return None
        by_groove[grooves].append(tok)

    if not by_groove:
        return None

    mhc_items: list[MHCInput] = []
    canonical_tokens: list[str] = []
    for g1, g2 in sorted(by_groove):
        rep_toks = sorted(by_groove[(g1, g2)])
        canonical_tokens.append(rep_toks[0])
        mhc_items.append(
            MHCInput(
                groove1=g1,
                groove2=g2,
                mhc_class=mhc_class,
                provenance={
                    "reported_alleles": rep_toks,
                    "mhc_species": raw_species,
                    "species_slug": species_slug,
                    "resolution_mode": mode,
                },
            )
        )
    tag = ";".join(sorted(canonical_tokens))
    return tuple(mhc_items), tag, mode


def _obs_id(prefix: str, key_parts: tuple[Any, ...]) -> str:
    h = hashlib.sha256("|".join(str(x) for x in key_parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{h}"


def _write_obs_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def allocate_nonhuman_ms_strata(
    wmap: Mapping[tuple[str, str, str], float],
    stratum_counts: Mapping[tuple[str, str, str], int],
    *,
    nh_ms_draw_mass: float = 0.06,
    max_fraction_of_elution: float = 0.25,
) -> dict[tuple[str, str, str], float]:
    """Carve bounded sqrt-balanced mass for `nh_ms:*` purely from human elution.

    Guarantees ZERO drift on human affinity, APC/perturbation contrasts, bulk
    proteomics, stability, qualitative binding, or T-cell draw budgets, and
    caps non-human MS at ``max_fraction_of_elution`` of total elution mass.
    """
    out = dict(wmap)
    nh_keys = [k for k in out if k[0] == "elution_observation" and str(k[2]).startswith("nh_ms:")]
    hu_elution_keys = [
        k for k in out if k[0] == "elution_observation" and not str(k[2]).startswith("nh_ms:")
    ]
    if not nh_keys or not hu_elution_keys or nh_ms_draw_mass <= 0:
        for k in nh_keys:
            out[k] = 0.0
        return out

    total_elution = sum(out[k] for k in nh_keys) + sum(out[k] for k in hu_elution_keys)
    if total_elution <= 0:
        return out
    eff_nh_mass = min(float(nh_ms_draw_mass), total_elution * float(max_fraction_of_elution))
    eff_hu_mass = total_elution - eff_nh_mass
    hu_sum = sum(out[k] for k in hu_elution_keys) or 1.0
    for k in hu_elution_keys:
        out[k] = out[k] * (eff_hu_mass / hu_sum)

    sqrt_weights = {k: math.sqrt(max(1.0, float(stratum_counts.get(k, 1)))) for k in nh_keys}
    sqrt_sum = sum(sqrt_weights.values()) or 1.0
    for k in nh_keys:
        out[k] = eff_nh_mass * (sqrt_weights[k] / sqrt_sum)
    return out


def build_hitlist_nonhuman_ms_expansion(
    observations_parquet: Path,
    protected: ProtectedIdentityIndex,
    *,
    proteome_paths: Mapping[str, Path],
    out_path: Path,
    mhc_full_seqs_csv: Path | None = None,
    peptide_mappings_parquet: Path | None = None,
    canonical_protein_ids_by_species: Mapping[str, set[str] | frozenset[str]] | None = None,
    max_candidate_junctions: int = 5,
    neg_per_pos: int = 1,
    seed: int = 20261123,
    allow_exact_haplotypes: bool = True,
    extra_observed_peptides: set[str] | None = None,
) -> dict[str, Any]:
    """Build leak-free non-human MS elution training JSONL from observations.parquet."""
    import pandas as pd

    if neg_per_pos < 1:
        raise ValueError("neg_per_pos must be >= 1")

    # Normalize proteome_paths keys to canonical species slugs
    norm_proteomes: dict[str, Path] = {}
    for k, v in proteome_paths.items():
        slug = k.strip().lower()
        if slug in SPECIES_CANONICAL_MAP:
            slug = SPECIES_CANONICAL_MAP[slug]
        norm_proteomes[slug] = Path(v)

    canon_by_species: dict[str, set[str]] = {}
    if peptide_mappings_parquet is not None and Path(peptide_mappings_parquet).exists():
        for sp, ids in load_canonical_protein_ids(Path(peptide_mappings_parquet)).items():
            canon_by_species[sp] = set(ids)
    if canonical_protein_ids_by_species:
        for k, ids in canonical_protein_ids_by_species.items():
            slug = k.strip().lower()
            if slug in SPECIES_CANONICAL_MAP:
                slug = SPECIES_CANONICAL_MAP[slug]
            canon_by_species.setdefault(slug, set()).update(ids)

    resolver = FastGrooveResolver(mhc_full_seqs_csv=mhc_full_seqs_csv)

    # Pass 1: Collect ALL MS-observed peptides across ALL species (human + non-human)
    # so generated negatives can never collide with any known eluted ligand.
    all_obs_df = pd.read_parquet(
        observations_parquet,
        columns=["peptide", "response_measured", "has_ptm"],
    )
    all_observed_ms: set[str] = set()
    for pep_raw, resp, has_ptm in zip(
        all_obs_df["peptide"].tolist(),
        all_obs_df["response_measured"].tolist(),
        all_obs_df["has_ptm"].tolist(),
    ):
        if (not pd.isna(has_ptm) and bool(has_ptm)) or resp != "ligand presentation":
            continue
        if isinstance(pep_raw, str):
            p = pep_raw.strip().upper()
            if p and (set(p) <= AA_SET):
                all_observed_ms.add(p)
    if extra_observed_peptides:
        for p in extra_observed_peptides:
            s = str(p).strip().upper()
            if s:
                all_observed_ms.add(s)
    del all_obs_df

    # Pass 2: Read non-human rows with required curation & provenance columns
    cols = [
        "peptide",
        "mhc_species",
        "mhc_class",
        "mhc_restriction",
        "mhc_allele_set",
        "has_ptm",
        "is_non_peptide_ligand",
        "is_potential_contaminant",
        "restriction_evidence",
        "assay_method",
        "response_measured",
        "qualitative_measurement",
        "pmid",
        "assay_iri",
        "reference_iri",
    ]
    df = pd.read_parquet(observations_parquet, columns=cols)
    nh = df[df["mhc_species"].fillna("") != "Homo sapiens"]

    counts: Counter[str] = Counter()
    by_species_class: Counter[str] = Counter()
    by_allele: Counter[str] = Counter()
    by_resolution_mode: Counter[str] = Counter()
    by_junction_policy: Counter[str] = Counter()

    # Deduplicate admitted positives by (peptide, canonical_mhc_key, pmid)
    seen_pos_keys: set[tuple[str, str, str]] = set()
    candidates_by_species: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def _is_na(val: Any) -> bool:
        return val is None or bool(pd.isna(val))

    def _safe_text(val: Any) -> str:
        return "" if _is_na(val) else str(val).strip()

    def _safe_pmid(val: Any) -> str:
        if _is_na(val):
            return "unknown"
        if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
            return str(int(val))
        s = str(val).strip()
        return s[:-2] if s.endswith(".0") else (s or "unknown")

    for row in nh.itertuples(index=False):
        counts["nonhuman_raw_rows"] += 1
        if row.response_measured != "ligand presentation":
            counts["drop_not_ligand_presentation"] += 1
            continue
        am = _safe_text(row.assay_method).lower()
        if "mass spectrometry" not in am:
            counts["drop_not_ms"] += 1
            continue
        if row.qualitative_measurement != "Positive":
            counts["drop_not_positive"] += 1
            continue
        if _is_na(row.has_ptm) or bool(row.has_ptm) is not False:
            counts["drop_ptm"] += 1
            continue
        if _is_na(row.is_non_peptide_ligand) or bool(row.is_non_peptide_ligand) is not False:
            counts["drop_non_peptide"] += 1
            continue
        if not _is_na(row.is_potential_contaminant) and bool(row.is_potential_contaminant):
            counts["drop_contaminant"] += 1
            continue
        if _safe_text(row.restriction_evidence).lower() == "predicted":
            counts["drop_predicted_restriction"] += 1
            continue
        cls = _safe_text(row.mhc_class)
        if cls not in ("I", "II"):
            counts["drop_mhc_class"] += 1
            continue
        pep = _clean_peptide(None if _is_na(row.peptide) else row.peptide, cls)
        if pep is None:
            counts["drop_peptide_invalid"] += 1
            continue

        raw_sp = _safe_text(row.mhc_species)
        sp_slug = canonical_species_slug(raw_sp)
        if sp_slug is None:
            counts["drop_unsupported_species"] += 1
            continue

        raw_allele = _safe_text(row.mhc_allele_set) or _safe_text(row.mhc_restriction)
        resolved = resolve_nonhuman_mhc_candidates(
            raw_allele,
            sp_slug,
            raw_sp,
            cls,
            resolver,
            allow_exact_haplotypes=allow_exact_haplotypes,
        )
        if resolved is None:
            counts["drop_unresolved_allele"] += 1
            continue
        mhc_tuple, allele_tag, res_mode = resolved

        emb_reason = protected.reason(pep)
        if emb_reason is not None:
            counts[f"drop_embargo_{emb_reason}"] += 1
            counts["drop_embargo_total"] += 1
            continue

        pmid_s = _safe_pmid(row.pmid)
        mhc_key = "|".join(f"{m.groove1}:{m.groove2}" for m in mhc_tuple)
        dedup_key = (pep, mhc_key, pmid_s)
        if dedup_key in seen_pos_keys:
            counts["drop_duplicate_pep_mhc_pmid"] += 1
            continue
        seen_pos_keys.add(dedup_key)

        candidates_by_species[sp_slug].append(
            {
                "peptide": pep,
                "mhc": mhc_tuple,
                "mhc_class": cls,
                "allele_tag": allele_tag,
                "resolution_mode": res_mode,
                "raw_species": raw_sp,
                "species_slug": sp_slug,
                "pmid": pmid_s,
                "assay_iri": _safe_text(row.assay_iri),
                "reference_iri": _safe_text(row.reference_iri),
                "assay_method": _safe_text(row.assay_method),
            }
        )

    del df
    del nh

    # Pass 3: Per-species proteome mapping, flank extraction, and same-protein negatives
    out_rows: list[dict[str, Any]] = []
    pos_admitted = 0
    neg_admitted = 0

    for sp_slug in sorted(candidates_by_species):
        cands = candidates_by_species[sp_slug]
        prot_path = norm_proteomes.get(sp_slug)
        if prot_path is None or not prot_path.exists():
            counts["drop_no_proteome"] += len(cands)
            continue

        proteins, reviewed_ids = read_proteome_fasta_with_reviewed(prot_path)
        sp_canon_ids = canon_by_species.get(sp_slug)
        uniq_peps = sorted({c["peptide"] for c in cands})
        hits = map_peptides(uniq_peps, proteins)
        families = protein_families(hits)

        # Sort candidates deterministically for 100% reproducible pairing
        cands.sort(key=lambda c: (c["allele_tag"], c["pmid"], c["peptide"], c["assay_iri"]))

        for cand in cands:
            pep = cand["peptide"]
            occurrences = hits.get(pep)
            if not occurrences:
                counts["drop_unmapped_to_proteome"] += 1
                continue

            family = f"nh_ms:{sp_slug}:{families[occurrences[0][0]]}"
            pos_junction, junction_policy, preferred_occs = resolve_source_junction(
                pep,
                occurrences,
                proteins,
                canonical_ids=sp_canon_ids,
                reviewed_ids=reviewed_ids,
            )
            study_id = f"nh_ms:{sp_slug}:{_allele_slug(cand['allele_tag'])}"
            pos_id = _obs_id(
                "nh_ms_pos",
                (
                    sp_slug,
                    cand["mhc_class"],
                    cand["allele_tag"],
                    cand["pmid"],
                    pep,
                ),
            )
            pair_id = stable_digest(f"nh_ms_pair|{seed}|{pos_id}")[:32]

            # Sample `neg_per_pos` distinct length-matched same-protein negatives
            # Prefer canonical/reviewed isoform when resolved; fall back to all occurrences
            sampled_negs: list[tuple[str, str, int, dict[str, Any]]] = []
            local_excluded: set[str] = set()
            for rep in range(neg_per_pos):
                obs_guard = (
                    all_observed_ms if not local_excluded else (all_observed_ms | local_excluded)
                )
                neg_cand = background_candidate(
                    pep,
                    preferred_occs,
                    proteins,
                    observed=obs_guard,
                    protected=protected,
                    key=f"nh_ms_bg:{seed}:{pos_id}:{rep}",
                )
                if neg_cand is None and preferred_occs is not occurrences:
                    neg_cand = background_candidate(
                        pep,
                        occurrences,
                        proteins,
                        observed=obs_guard,
                        protected=protected,
                        key=f"nh_ms_bg:{seed}:{pos_id}:{rep}",
                    )
                if neg_cand is None:
                    break
                neg_seq, neg_prot, neg_pos, neg_junc = neg_cand
                local_excluded.add(neg_seq)
                sampled_negs.append((neg_seq, neg_prot, neg_pos, neg_junc))

            if len(sampled_negs) < neg_per_pos:
                counts["drop_no_qualified_background"] += 1
                continue

            base_prov = {
                "study_id": study_id,
                "pmid": cand["pmid"],
                "pair_id": pair_id,
                "mhc_species": cand["raw_species"],
                "species_slug": sp_slug,
                "canonical_allele": cand["allele_tag"],
                "resolution_mode": cand["resolution_mode"],
                "assay_method": cand["assay_method"],
                "assay_iri": cand["assay_iri"],
                "reference_iri": cand["reference_iri"],
                "source_family": family,
                "source_candidate_count": len(occurrences),
                "junction_policy": junction_policy,
                "test_accessed": False,
            }

            is_ambiguous = junction_policy.startswith("ambiguous_")
            pos_cand_juncs: tuple[SourceJunction, ...] = ()
            attach_sym_cands = False
            if is_ambiguous and max_candidate_junctions >= 2:
                cand_juncs = collect_candidate_junctions(
                    pep,
                    occurrences,
                    proteins,
                    max_candidates=max_candidate_junctions,
                )
                if cand_juncs and all(
                    nj["source_n"] is not None and nj["source_c"] is not None
                    for _, _, _, nj in sampled_negs
                ):
                    pos_cand_juncs = cand_juncs
                    attach_sym_cands = True
                    counts["ambiguous_with_candidate_junctions"] += 1

            pos_obs = Observation(
                observation_id=pos_id,
                source="hitlist_nonhuman_ms_expansion",
                endpoint="elution_observation",
                peptide=pep,
                mhc=cand["mhc"],
                value=1.0,
                unit="binary",
                split="train",
                group_id=family,
                context=Context(**pos_junction, candidate_junctions=pos_cand_juncs),
                provenance={
                    **base_prov,
                    "measurement_lineage": {
                        "namespace": "hitlist_nonhuman_ms",
                        "measurement_id": cand["assay_iri"] or pos_id,
                    },
                    "label_kind": "reported_positive_nonhuman_ms",
                },
            )
            pos_obs.validate()
            out_rows.append(pos_obs.to_dict())
            pos_admitted += 1

            for rep, (neg_seq, neg_prot, neg_pos, neg_junc) in enumerate(sampled_negs):
                # Symmetric missing-junction masking in "masked" mode; symmetric
                # candidate_junctions attachment for "sample" / "mil" modes.
                eff_neg_junc = pos_junction if is_ambiguous else neg_junc
                neg_cand_juncs = (SourceJunction(**neg_junc),) if attach_sym_cands else ()
                neg_id = _obs_id("nh_ms_neg", (pair_id, rep, neg_seq))
                neg_obs = Observation(
                    observation_id=neg_id,
                    source="hitlist_nonhuman_ms_background",
                    endpoint="elution_observation",
                    peptide=neg_seq,
                    mhc=cand["mhc"],
                    value=0.0,
                    unit="binary",
                    split="train",
                    group_id=family,
                    context=Context(**eff_neg_junc, candidate_junctions=neg_cand_juncs),
                    provenance={
                        **base_prov,
                        "measurement_lineage": {
                            "namespace": "hitlist_nonhuman_ms_background",
                            "measurement_id": neg_id,
                        },
                        "parent_positive_id": pos_id,
                        "replicate": rep,
                        "source_protein": neg_prot,
                        "generated_position": neg_pos,
                        "label_kind": "generated_same_protein_nonhuman_background",
                    },
                )
                neg_obs.validate()
                out_rows.append(neg_obs.to_dict())
                neg_admitted += 1

            by_species_class[f"{sp_slug}_{cand['mhc_class']}"] += 1
            by_allele[f"{sp_slug}:{cand['allele_tag']}"] += 1
            by_resolution_mode[cand["resolution_mode"]] += 1
            by_junction_policy[junction_policy] += 1

    _write_obs_jsonl(out_path, out_rows)
    return {
        "out_path": str(out_path),
        "test_accessed": False,
        "neg_per_pos": neg_per_pos,
        "allow_exact_haplotypes": allow_exact_haplotypes,
        "positives_admitted": pos_admitted,
        "negatives_admitted": neg_admitted,
        "total_rows_written": len(out_rows),
        "n_global_observed_ms_excluded": len(all_observed_ms),
        "counts": dict(sorted(counts.items())),
        "positives_by_species_class": dict(sorted(by_species_class.items())),
        "positives_by_allele": dict(sorted(by_allele.items(), key=lambda kv: (-kv[1], kv[0]))),
        "positives_by_resolution_mode": dict(sorted(by_resolution_mode.items())),
        "positives_by_junction_policy": dict(sorted(by_junction_policy.items())),
    }
