"""Leak-free multi-output expansion builders from `hitlist>=2.17.0` for Presto-Omni v2.3+.

Builds five independently embargo-verified (`ProtectedIdentityIndex`, `test_accessed=False`)
training expansions targeting the highest-leverage multi-output improvements:
  1. `HLA Ligand Atlas 2020.12` (benign primary human tissue Class I & II multiallelic bags,
     29 organs, 21 4-digit HLA-typed donors, `cell_type="healthy_tissue"`).
  2. `Bergseng 2015 HLA-DQ` (PMID 25502872, deep homozygous EBV-LCL HLA-DQ peptidomics).
  3. `DO:DM Ratio 35085787` (PMID 35085787, T2-DR4 graded `HLA-DM` / `HLA-DO` editing arms
     emitting exact `components={"HLA-DMA", "HLA-DMB", "HLA-DOA", "HLA-DOB"}` states).
  4. Non-classical peptide-binding MHC-I (`HLA-E*`, `HLA-G*`, `HLA-F*` elution + qualitative
     binding; strictly excludes lipid/metabolite-presenting `CD1` and `MR1`).
  5. Expanded 4-protease `Bekker-Jensen_2017` `bulk_observation` MS detectability pairs
     (`chymotrypsin`, `gluc`, `lysc`, `trypsin_p` — scaling ~4–6x beyond v2.1's 2.6% sample).

All elution builders reuse `read_proteome_fasta_with_reviewed`, `map_peptides`,
`resolve_source_junction`, and `collect_candidate_junctions` for within-gene canonical
isoform flank resolution, symmetric `candidate_junctions` (`2 <= M <= 5`) for
`candidate_junction_mode="mil"`, same-protein length-matched negative sampling excluding
globally observed peptides + `ProtectedIdentityIndex`, and zero-regression stratum budgeting.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping

import pandas as pd

from .hitlist_binding_expansion import (
    FastGrooveResolver,
    ProtectedIdentityIndex,
)
from .hitlist_nonhuman_ms_expansion import (
    collect_candidate_junctions,
    read_proteome_fasta_with_reviewed,
    resolve_source_junction,
)
from .modular import (
    AssayMethod,
    Component,
    Context,
    MHCInput,
    Observation,
    SourceJunction,
)
from .presentation import Protein, junction, map_peptides

_CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def is_canonical_peptide(seq: str) -> bool:
    return bool(seq) and set(seq).issubset(_CANONICAL_AA)


def _is_na(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


ALLOWED_NONCLASSICAL_PREFIXES: tuple[str, ...] = ("HLA-E*", "HLA-G*", "HLA-F*")
REJECTED_NONCLASSICAL_TOKENS: tuple[str, ...] = ("CD1", "MR1", "MICA", "MICB", "AZGP1")

ENZYME_MAP: dict[str, str] = {
    "Chymotrypsin": "chymotrypsin",
    "GluC": "gluc",
    "LysC": "lysc",
    "Trypsin/P (cleaves K/R except before P)": "trypsin_p",
}
ENZYME_CUT_RE: dict[str, re.Pattern[str]] = {
    "chymotrypsin": re.compile(r"(?<=[FYWL])(?!P)"),
    "gluc": re.compile(r"(?<=E)(?!P)"),
    "lysc": re.compile(r"(?<=K)"),
    "trypsin_p": re.compile(r"(?<=[KR])(?!P)"),
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.name.endswith(".gz") else open
    n = 0
    with opener(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            n += 1
    return n


def normalize_human_hla_token(token: str) -> str:
    """Normalize HLA Ligand Atlas / hitlist allele tokens for `FastGrooveResolver`."""
    tok = str(token or "").strip()
    if not tok:
        return ""
    if "-" in tok and not tok.startswith("HLA-"):
        left, right = tok.split("-", 1)
        left_n = left if left.startswith("HLA-") else f"HLA-{left}"
        right_n = right if right.startswith("HLA-") else f"HLA-{right}"
        return f"{left_n}/{right_n}"
    if "/" in tok:
        left, right = tok.split("/", 1)
        left_n = left if left.startswith("HLA-") else f"HLA-{left}"
        right_n = right if right.startswith("HLA-") else f"HLA-{right}"
        return f"{left_n}/{right_n}"
    if not tok.startswith("HLA-"):
        return f"HLA-{tok}"
    return tok


def resolve_human_mhc_bag(
    tokens: Iterable[str],
    mhc_class: str,
    resolver: FastGrooveResolver,
) -> tuple[MHCInput, ...]:
    """Resolve a set of human HLA allele tokens into a deduplicated `MHCInput` tuple."""
    seen: set[tuple[str, str]] = set()
    bag: list[MHCInput] = []
    for raw in tokens:
        for part in str(raw or "").split(";"):
            norm = normalize_human_hla_token(part)
            if not norm:
                continue
            grooves = resolver.resolve(norm, mhc_class)
            if grooves is None:
                continue
            g1, g2 = grooves
            key = (g1, g2)
            if key in seen:
                continue
            seen.add(key)
            bag.append(
                MHCInput(
                    groove1=g1,
                    groove2=g2,
                    mhc_class=mhc_class,
                    species1="human",
                    species2="human",
                    provenance={"reported_restriction": norm},
                )
            )
    return tuple(bag)


def _sample_same_protein_negatives(
    *,
    protein_seq: str,
    length: int,
    pos_start: int,
    global_observed: set[str],
    protected: ProtectedIdentityIndex,
    rng: random.Random,
    k: int = 1,
    max_attempts: int = 64,
) -> list[tuple[str, int]]:
    """Sample up to `k` distinct, non-overlapping unobserved, non-embargoed same-protein windows."""
    n = len(protein_seq)
    if n < length + 2 or k <= 0:
        return []
    max_start = n - length
    min_sep = max(2, length // 2)
    picked: list[tuple[str, int]] = []
    seen_seqs: set[str] = set()
    attempts = max(max_attempts, k * 24)
    for _ in range(attempts):
        if len(picked) >= k:
            break
        s = rng.randint(0, max_start)
        if abs(s - pos_start) < min_sep:
            continue
        if any(abs(s - prev_s) < min_sep for _, prev_s in picked):
            continue
        cand = protein_seq[s : s + length]
        if cand in seen_seqs or not is_canonical_peptide(cand):
            continue
        if cand in global_observed:
            continue
        if protected.reason(cand) is not None:
            continue
        seen_seqs.add(cand)
        picked.append((cand, s))
    return picked


def _emit_mapped_elution_pairs(
    *,
    items: list[dict[str, Any]],
    proteins: Mapping[str, Protein],
    reviewed_ids: set[str],
    canonical_ids: set[str] | None,
    global_observed: set[str],
    protected: ProtectedIdentityIndex,
    source_name: str,
    seed: int,
    flank_len: int = 15,
    max_candidate_junctions: int = 5,
    max_pairs_per_stratum: int | None = None,
    neg_per_pos: int = 1,
    neg_per_pos_by_class: Mapping[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map positive peptides to human proteome and emit 1:K positive + negative rows."""
    rng = random.Random(seed)
    counts: Counter[str] = Counter()

    if max_pairs_per_stratum is not None:
        by_st: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for it in items:
            by_st[str(it["stratum"])].append(it)
        capped: list[dict[str, Any]] = []
        for st in sorted(by_st):
            bucket = by_st[st]
            if len(bucket) > max_pairs_per_stratum:
                rng_st = random.Random(seed ^ (hash(st) & 0xFFFFFFFF))
                rng_st.shuffle(bucket)
                counts[f"capped_dropped:{st}"] += len(bucket) - max_pairs_per_stratum
                bucket = bucket[:max_pairs_per_stratum]
            capped.extend(bucket)
        items = capped

    unique_peps = sorted({it["peptide"] for it in items})
    mappings = map_peptides(unique_peps, proteins) if unique_peps else {}

    out_rows: list[dict[str, Any]] = []
    by_stratum: Counter[str] = Counter()
    seen_lineage: set[str] = set()

    for idx, it in enumerate(items):
        pep = it["peptide"]
        hits = mappings.get(pep) or []
        if not hits:
            counts["dropped_unmapped"] += 1
            continue

        jctx, jpolicy, pref_hits = resolve_source_junction(
            pep,
            hits,
            proteins,
            canonical_ids=canonical_ids,
            reviewed_ids=reviewed_ids,
            flank_length=flank_len,
        )
        counts[f"junction:{jpolicy}"] += 1
        chosen_pid, chosen_start = pref_hits[0]

        cand_junctions: tuple[SourceJunction, ...] = ()
        if jctx["source_n"] is None:
            cand_junctions = collect_candidate_junctions(
                pep,
                hits,
                proteins,
                max_candidates=max_candidate_junctions,
                flank_length=flank_len,
            )
            if cand_junctions:
                counts["ambiguous_with_candidate_junctions"] += 1

        mhc_bag: tuple[MHCInput, ...] = it["mhc"]
        mhc_cls = mhc_bag[0].mhc_class if mhc_bag else "I"
        target_k = int((neg_per_pos_by_class or {}).get(mhc_cls, neg_per_pos))
        neg_picks = _sample_same_protein_negatives(
            protein_seq=proteins[chosen_pid].sequence,
            length=len(pep),
            pos_start=chosen_start,
            global_observed=global_observed,
            protected=protected,
            rng=rng,
            k=max(1, target_k),
        )
        if not neg_picks:
            counts["dropped_no_negative"] += 1
            continue

        lineage_base = str(it["lineage"])
        if lineage_base in seen_lineage:
            counts["dropped_duplicate_lineage"] += 1
            continue
        seen_lineage.add(lineage_base)

        pair_id = f"{source_name}:pair:{idx}"
        group_id = str(it["group_id"])
        cell_type: str | None = it.get("cell_type")
        components: dict[str, Component] = dict(it.get("components") or {})
        stratum = str(it["stratum"])
        base_prov = dict(it.get("provenance") or {})

        # Symmetric flank / candidate_junctions masking parity (zero flank-presence shortcut!)
        if jctx["source_n"] is not None:
            pos_j = jctx
            pos_cands: tuple[SourceJunction, ...] = ()
        else:
            pos_j = {"source_n": None, "source_c": None, "n_terminus": None, "c_terminus": None}
            pos_cands = cand_junctions

        emit_records: list[tuple[bool, str, str, dict[str, Any], tuple[SourceJunction, ...]]] = [
            (True, "pos", pep, pos_j, pos_cands)
        ]
        for neg_idx, (neg_pep, neg_start) in enumerate(neg_picks):
            role_tag = "neg" if neg_idx == 0 else f"neg{neg_idx}"
            if jctx["source_n"] is not None:
                neg_j = junction(
                    proteins[chosen_pid].sequence, neg_start, len(neg_pep), flank_length=flank_len
                )
                neg_cands: tuple[SourceJunction, ...] = ()
            else:
                neg_j = {"source_n": None, "source_c": None, "n_terminus": None, "c_terminus": None}
                if cand_junctions:
                    nj = junction(
                        proteins[chosen_pid].sequence,
                        neg_start,
                        len(neg_pep),
                        flank_length=flank_len,
                    )
                    neg_cands = tuple(
                        SourceJunction(
                            source_n=nj["source_n"],
                            source_c=nj["source_c"],
                            n_terminus=nj["n_terminus"],
                            c_terminus=nj["c_terminus"],
                        )
                        for _ in range(len(cand_junctions))
                    )
                else:
                    neg_cands = ()
            emit_records.append((False, role_tag, neg_pep, neg_j, neg_cands))

        for is_pos, role, seq, j_d, cands in emit_records:
            ctx = Context(
                source_n=j_d["source_n"],
                source_c=j_d["source_c"],
                n_terminus=j_d["n_terminus"],
                c_terminus=j_d["c_terminus"],
                cell_type=cell_type,
                components=components,
                candidate_junctions=cands,
            )
            prov = {
                **base_prov,
                "pair_id": pair_id,
                "protein_id": chosen_pid,
                "source_protein": chosen_pid,
                "stratum_override": stratum,
                "role": "observed_positive" if is_pos else "background_candidate",
                "measurement_lineage": f"{lineage_base}:{role}",
            }
            obs = Observation(
                observation_id=f"{source_name}:{idx}:{role}",
                source=source_name,
                split="train",
                group_id=group_id,
                endpoint="elution_observation",
                peptide=seq,
                mhc=mhc_bag,
                value=1.0 if is_pos else 0.0,
                unit="binary",
                qualifier="=",
                context=ctx,
                provenance=prov,
            )
            out_rows.append(obs.to_dict())

        by_stratum[stratum] += 1
        counts["pairs_admitted"] += 1
        counts["negatives_admitted"] += len(neg_picks)

    stats = {
        "source": source_name,
        "positives_admitted": counts["pairs_admitted"],
        "total_rows_written": len(out_rows),
        "by_stratum": dict(sorted(by_stratum.items())),
        "counts": dict(sorted(counts.items())),
    }
    return out_rows, stats


def build_hla_ligand_atlas_expansion(
    *,
    atlas_dir: Path,
    human_fasta: Path,
    mhc_full_seqs_csv: Path,
    protected: ProtectedIdentityIndex,
    global_observed: set[str],
    out_path: Path,
    canonical_ids: set[str] | None = None,
    max_pairs_per_class: int = 80_000,
    neg_per_pos_class1: int = 1,
    neg_per_pos_class2: int = 1,
    seed: int = 20261124,
) -> dict[str, Any]:
    """Build leak-free benign tissue Class I & II pairs from HLA Ligand Atlas 2020.12."""
    donors_df = pd.read_csv(next(atlas_dir.rglob("donors.tsv.gz")), sep="\t")
    peptides_df = pd.read_csv(next(atlas_dir.rglob("peptides.tsv.gz")), sep="\t")
    hits_df = pd.read_csv(next(atlas_dir.rglob("sample_hits.tsv.gz")), sep="\t")

    resolver = FastGrooveResolver(mhc_full_seqs_csv)
    donor_bags: dict[tuple[str, str], tuple[MHCInput, ...]] = {}
    for donor, grp in donors_df.groupby("donor"):
        alleles = grp["hla_allele"].dropna().astype(str).tolist()
        i_toks = [a for a in alleles if a.startswith(("A*", "B*", "C*"))]
        ii_toks = [a for a in alleles if a.startswith(("DR", "DP", "DQ"))]
        donor_bags[(str(donor), "I")] = resolve_human_mhc_bag(i_toks, "I", resolver)
        donor_bags[(str(donor), "II")] = resolve_human_mhc_bag(ii_toks, "II", resolver)

    pep_lookup = dict(
        zip(peptides_df["peptide_sequence_id"], peptides_df["peptide_sequence"].astype(str))
    )
    hits_df = hits_df.copy()
    hits_df["peptide"] = hits_df["peptide_sequence_id"].map(pep_lookup)
    hits_df["mhc_class"] = hits_df["hla_class"].map({"HLA-I": "I", "HLA-II": "II"})

    counts_pre: Counter[str] = Counter()
    items: list[dict[str, Any]] = []
    grouped = hits_df.groupby(["peptide", "donor", "mhc_class"], sort=False)["tissue"].agg(
        lambda s: sorted(set(str(x) for x in s.dropna()))
    )
    for (pep, donor, mhc_class), tissues in grouped.items():
        pep_s = str(pep or "").strip().upper()
        if not is_canonical_peptide(pep_s):
            counts_pre["dropped_noncanonical"] += 1
            continue
        if mhc_class == "I" and not (8 <= len(pep_s) <= 15):
            counts_pre["dropped_length"] += 1
            continue
        if mhc_class == "II" and not (9 <= len(pep_s) <= 30):
            counts_pre["dropped_length"] += 1
            continue
        if protected.reason(pep_s) is not None:
            counts_pre["dropped_embargo"] += 1
            continue
        bag = donor_bags.get((str(donor), str(mhc_class)), ())
        if not bag:
            counts_pre["dropped_no_donor_mhc"] += 1
            continue
        items.append(
            {
                "peptide": pep_s,
                "mhc": bag,
                "stratum": f"v23_atlas:{mhc_class}",
                "group_id": f"hla_ligand_atlas:{donor}:{mhc_class}",
                "cell_type": "healthy_tissue",
                "lineage": f"hla_atlas_2020:{donor}:{mhc_class}:{pep_s}",
                "provenance": {
                    "study_id": "33737343:hla_ligand_atlas",
                    "pmid": "33737343",
                    "donor": str(donor),
                    "tissues": tissues[:8],
                    "n_tissues": len(tissues),
                },
            }
        )

    proteins, reviewed_ids = read_proteome_fasta_with_reviewed(human_fasta)
    all_obs = set(global_observed) | {it["peptide"] for it in items}

    rows, stats = _emit_mapped_elution_pairs(
        items=items,
        proteins=proteins,
        reviewed_ids=reviewed_ids,
        canonical_ids=canonical_ids,
        global_observed=all_obs,
        protected=protected,
        source_name="hitlist217_hla_ligand_atlas",
        seed=seed,
        max_pairs_per_stratum=max_pairs_per_class,
        neg_per_pos_by_class={"I": max(1, neg_per_pos_class1), "II": max(1, neg_per_pos_class2)},
    )
    _write_jsonl(out_path, rows)
    stats["pre_filter_counts"] = dict(sorted(counts_pre.items()))
    stats["out_path"] = str(out_path)
    stats["sha256"] = _sha256_file(out_path)
    return stats


def build_bergseng_dq_expansion(
    *,
    workbook_path: Path,
    human_fasta: Path,
    mhc_full_seqs_csv: Path,
    protected: ProtectedIdentityIndex,
    global_observed: set[str],
    out_path: Path,
    canonical_ids: set[str] | None = None,
    seed: int = 20261125,
) -> dict[str, Any]:
    """Build leak-free deep HLA-DQ elution pairs from Bergseng 2015 (PMID 25502872)."""
    import hitlist

    parsed = hitlist.parse_bergseng_2015_workbook(workbook_path)
    runs_df = parsed["runs"]
    meas_df = parsed["measurements"]

    resolver = FastGrooveResolver(mhc_full_seqs_csv)
    run_to_mhc: dict[str, tuple[MHCInput, ...]] = {}
    run_to_meta: dict[str, dict[str, Any]] = {}
    for _, r in runs_df.iterrows():
        rid = str(r["run_id"])
        aset = str(r.get("mhc_allele_set") or "").strip()
        if not aset:
            continue
        bag = resolve_human_mhc_bag([aset], "II", resolver)
        if bag:
            run_to_mhc[rid] = bag
            run_to_meta[rid] = {
                "dq_serotype": str(r.get("dq_serotype") or ""),
                "cell_line_name": str(r.get("cell_line_name") or ""),
                "mhc_allele_set": aset,
            }

    pos = meas_df[
        (meas_df["raw_ion_peak_volume"].fillna(0) > 0)
        & (~meas_df["exceeds_methods_pep_0_1"].fillna(False).astype(bool))
        & (meas_df["run_id"].isin(run_to_mhc))
    ]

    counts_pre: Counter[str] = Counter()
    seen_key: set[tuple[str, str]] = set()
    items: list[dict[str, Any]] = []
    for pep, rid in zip(pos["peptide"], pos["run_id"]):
        pep_s = str(pep or "").strip().upper()
        if not is_canonical_peptide(pep_s) or not (9 <= len(pep_s) <= 30):
            counts_pre["dropped_noncanonical_or_len"] += 1
            continue
        if protected.reason(pep_s) is not None:
            counts_pre["dropped_embargo"] += 1
            continue
        meta = run_to_meta[str(rid)]
        dedup_k = (pep_s, meta["mhc_allele_set"])
        if dedup_k in seen_key:
            continue
        seen_key.add(dedup_k)
        items.append(
            {
                "peptide": pep_s,
                "mhc": run_to_mhc[str(rid)],
                "stratum": "v23_dq:II",
                "group_id": f"bergseng_2015_dq:{meta['dq_serotype']}",
                "cell_type": "ebv_lcl",
                "lineage": f"bergseng2015:{meta['mhc_allele_set']}:{pep_s}",
                "provenance": {
                    "study_id": "25502872:bergseng_dq",
                    "pmid": "25502872",
                    **meta,
                },
            }
        )

    proteins, reviewed_ids = read_proteome_fasta_with_reviewed(human_fasta)
    all_obs = set(global_observed) | {it["peptide"] for it in items}

    rows, stats = _emit_mapped_elution_pairs(
        items=items,
        proteins=proteins,
        reviewed_ids=reviewed_ids,
        canonical_ids=canonical_ids,
        global_observed=all_obs,
        protected=protected,
        source_name="hitlist217_bergseng_dq",
        seed=seed,
    )
    _write_jsonl(out_path, rows)
    stats["pre_filter_counts"] = dict(sorted(counts_pre.items()))
    stats["out_path"] = str(out_path)
    stats["sha256"] = _sha256_file(out_path)
    return stats


def build_dodm_ratio_expansion(
    *,
    workbook_path: Path,
    human_fasta: Path,
    mhc_full_seqs_csv: Path,
    protected: ProtectedIdentityIndex,
    global_observed: set[str],
    out_path: Path,
    canonical_ids: set[str] | None = None,
    seed: int = 20261126,
) -> dict[str, Any]:
    """Build leak-free HLA-DO:HLA-DM Class II editing elution pairs from PMID 35085787."""
    import hitlist

    parsed = hitlist.parse_dodm_ratio_workbook(workbook_path)
    runs_df = parsed["runs"]
    ids_df = parsed["identifications"]

    resolver = FastGrooveResolver(mhc_full_seqs_csv)
    dr4_bag = resolve_human_mhc_bag(["HLA-DRB1*04:01"], "II", resolver)
    if not dr4_bag:
        raise RuntimeError("Failed to resolve HLA-DRB1*04:01 for DO:DM ratio expansion")

    clone_meta: dict[str, dict[str, Any]] = {}
    for _, r in runs_df.iterrows():
        clone = str(r["clone_name"])
        if clone not in clone_meta:
            clone_meta[clone] = {
                "condition_key": str(r["condition_key"]),
                "dm_present": bool(r["dm_present"]),
                "do_present": bool(r["do_present"]),
                "do_dm_level": str(r.get("do_dm_level") or ""),
            }

    def _match_clone(sample_id: str) -> dict[str, Any] | None:
        s = str(sample_id or "")
        for clone in sorted(clone_meta, key=len, reverse=True):
            if s.startswith(clone):
                return {"clone_name": clone, **clone_meta[clone]}
        return None

    counts_pre: Counter[str] = Counter()
    seen_key: set[tuple[str, str]] = set()
    items: list[dict[str, Any]] = []
    for sid, pep, has_ptm in zip(ids_df["sample_id"], ids_df["peptide"], ids_df["has_ptm"]):
        if not _is_na(has_ptm) and bool(has_ptm):
            counts_pre["dropped_ptm"] += 1
            continue
        pep_s = str(pep or "").strip().upper()
        if not is_canonical_peptide(pep_s) or not (9 <= len(pep_s) <= 30):
            counts_pre["dropped_noncanonical_or_len"] += 1
            continue
        if protected.reason(pep_s) is not None:
            counts_pre["dropped_embargo"] += 1
            continue
        cm = _match_clone(str(sid))
        if cm is None:
            counts_pre["dropped_unknown_clone"] += 1
            continue
        ck = cm["condition_key"]
        if (pep_s, ck) in seen_key:
            continue
        seen_key.add((pep_s, ck))

        dm_state = "intact" if cm["dm_present"] else "lost"
        do_state = "intact" if cm["do_present"] else "lost"
        comps = {
            "HLA-DMA": Component(state=dm_state),
            "HLA-DMB": Component(state=dm_state),
            "HLA-DOA": Component(state=do_state),
            "HLA-DOB": Component(state=do_state),
        }
        items.append(
            {
                "peptide": pep_s,
                "mhc": dr4_bag,
                "stratum": "v23_dodm:II",
                "group_id": f"dodm_35085787:{ck}",
                "cell_type": "monoallelic_host",
                "components": comps,
                "lineage": f"dodm35085787:{ck}:{pep_s}",
                "provenance": {
                    "study_id": "35085787:dodm_ratio",
                    "pmid": "35085787",
                    **cm,
                },
            }
        )

    proteins, reviewed_ids = read_proteome_fasta_with_reviewed(human_fasta)
    all_obs = set(global_observed) | {it["peptide"] for it in items}

    rows, stats = _emit_mapped_elution_pairs(
        items=items,
        proteins=proteins,
        reviewed_ids=reviewed_ids,
        canonical_ids=canonical_ids,
        global_observed=all_obs,
        protected=protected,
        source_name="hitlist217_dodm_ratio",
        seed=seed,
    )
    _write_jsonl(out_path, rows)
    stats["pre_filter_counts"] = dict(sorted(counts_pre.items()))
    stats["out_path"] = str(out_path)
    stats["sha256"] = _sha256_file(out_path)
    return stats


def build_nonclassical_mhc_expansion(
    *,
    observations_parquet: Path,
    binding_parquet: Path,
    human_fasta: Path,
    mhc_full_seqs_csv: Path,
    protected: ProtectedIdentityIndex,
    global_observed: set[str],
    out_elution_path: Path,
    out_binding_path: Path,
    canonical_ids: set[str] | None = None,
    seed: int = 20261127,
) -> dict[str, Any]:
    """Build leak-free non-classical peptide-binding MHC-I (`HLA-E*`, `HLA-G*`, `HLA-F*`) rows."""
    resolver = FastGrooveResolver(mhc_full_seqs_csv)

    def _is_allowed_nonclassical(allele: str) -> bool:
        a = str(allele or "").strip()
        if any(bad in a.upper() for bad in REJECTED_NONCLASSICAL_TOKENS):
            return False
        return a.startswith(ALLOWED_NONCLASSICAL_PREFIXES)

    odf = pd.read_parquet(
        observations_parquet,
        columns=["peptide", "mhc_class", "mhc_restriction", "has_ptm", "is_binding_assay", "pmid"],
    )
    odf = odf[(odf["mhc_class"] == "non-classical") & (~odf["is_binding_assay"].fillna(False))]
    counts_elut: Counter[str] = Counter()
    seen_elut: set[tuple[str, str]] = set()
    elut_items: list[dict[str, Any]] = []
    for pep, allele, has_ptm, pmid in zip(
        odf["peptide"], odf["mhc_restriction"], odf["has_ptm"], odf["pmid"]
    ):
        if not _is_na(has_ptm) and bool(has_ptm):
            counts_elut["dropped_ptm"] += 1
            continue
        allele_s = str(allele or "").strip()
        if not _is_allowed_nonclassical(allele_s):
            counts_elut["dropped_non_peptide_mhc"] += 1
            continue
        pep_s = str(pep or "").strip().upper()
        if not is_canonical_peptide(pep_s) or not (8 <= len(pep_s) <= 15):
            counts_elut["dropped_noncanonical_or_len"] += 1
            continue
        if protected.reason(pep_s) is not None:
            counts_elut["dropped_embargo"] += 1
            continue
        m_bag = resolve_human_mhc_bag([allele_s], "I", resolver)
        if not m_bag:
            counts_elut["dropped_unresolved_groove"] += 1
            continue
        if (pep_s, allele_s) in seen_elut:
            continue
        seen_elut.add((pep_s, allele_s))
        pmid_s = "" if _is_na(pmid) else str(int(pmid))
        elut_items.append(
            {
                "peptide": pep_s,
                "mhc": m_bag,
                "stratum": "v23_nonclassical:I",
                "group_id": f"nonclassical:{allele_s}:{pmid_s or 'unknown'}",
                "cell_type": None,
                "lineage": f"nc_ms:{allele_s}:{pep_s}",
                "provenance": {
                    "study_id": f"nonclassical:{pmid_s or 'unknown'}",
                    "pmid": pmid_s or None,
                    "reported_restriction": allele_s,
                },
            }
        )

    proteins, reviewed_ids = read_proteome_fasta_with_reviewed(human_fasta)
    all_obs = set(global_observed) | {it["peptide"] for it in elut_items}
    elut_rows, elut_stats = _emit_mapped_elution_pairs(
        items=elut_items,
        proteins=proteins,
        reviewed_ids=reviewed_ids,
        canonical_ids=canonical_ids,
        global_observed=all_obs,
        protected=protected,
        source_name="hitlist217_nonclassical_mhc",
        seed=seed,
    )
    _write_jsonl(out_elution_path, elut_rows)
    elut_stats["pre_filter_counts"] = dict(sorted(counts_elut.items()))
    elut_stats["out_path"] = str(out_elution_path)
    elut_stats["sha256"] = _sha256_file(out_elution_path)

    bdf = pd.read_parquet(
        binding_parquet,
        columns=[
            "peptide",
            "mhc_class",
            "mhc_restriction",
            "has_ptm",
            "response_measured",
            "qualitative_measurement",
            "pmid",
        ],
    )
    bdf = bdf[
        (bdf["mhc_class"] == "non-classical") & (bdf["response_measured"] == "qualitative binding")
    ]
    bind_rows: list[dict[str, Any]] = []
    counts_bind: Counter[str] = Counter()
    seen_bind: set[tuple[str, str]] = set()
    for idx, (pep, allele, has_ptm, qual, pmid) in enumerate(
        zip(
            bdf["peptide"],
            bdf["mhc_restriction"],
            bdf["has_ptm"],
            bdf["qualitative_measurement"],
            bdf["pmid"],
        )
    ):
        if not _is_na(has_ptm) and bool(has_ptm):
            counts_bind["dropped_ptm"] += 1
            continue
        allele_s = str(allele or "").strip()
        if not _is_allowed_nonclassical(allele_s):
            counts_bind["dropped_non_peptide_mhc"] += 1
            continue
        pep_s = str(pep or "").strip().upper()
        if not is_canonical_peptide(pep_s) or not (8 <= len(pep_s) <= 15):
            counts_bind["dropped_noncanonical_or_len"] += 1
            continue
        if protected.reason(pep_s) is not None:
            counts_bind["dropped_embargo"] += 1
            continue
        q_s = str(qual or "").strip()
        if q_s.startswith("Positive"):
            val = 1.0
        elif q_s == "Negative":
            val = 0.0
        else:
            counts_bind["dropped_unknown_qual"] += 1
            continue
        m_bag = resolve_human_mhc_bag([allele_s], "I", resolver)
        if not m_bag:
            counts_bind["dropped_unresolved_groove"] += 1
            continue
        if (pep_s, allele_s) in seen_bind:
            continue
        seen_bind.add((pep_s, allele_s))
        pmid_s = "" if _is_na(pmid) else str(int(pmid))
        obs = Observation(
            observation_id=f"hitlist217_nonclassical_bind:{idx}",
            source="hitlist217_nonclassical_mhc",
            split="train",
            group_id=f"nc_bind:{allele_s}:{pmid_s or 'unknown'}",
            endpoint="qualitative_binding",
            peptide=pep_s,
            mhc=m_bag,
            value=val,
            unit="binary",
            qualifier="=",
            assay_method=AssayMethod(
                preparation="unknown",
                mechanism="unknown",
                detector="unknown",
                review_status="protocol_citation_conflict",
            ),
            provenance={
                "study_id": f"nc_bind:{pmid_s or 'unknown'}",
                "pmid": pmid_s or None,
                "reported_restriction": allele_s,
                "qualitative_measurement": q_s,
                "measurement_lineage": f"nc_bind:{allele_s}:{pep_s}",
            },
        )
        bind_rows.append(obs.to_dict())
        counts_bind["admitted"] += 1

    _write_jsonl(out_binding_path, bind_rows)
    return {
        "elution": elut_stats,
        "binding": {
            "admitted": len(bind_rows),
            "counts": dict(sorted(counts_bind.items())),
            "out_path": str(out_binding_path),
            "sha256": _sha256_file(out_binding_path),
        },
    }


def build_expanded_bekker_jensen_bulk(
    *,
    bulk_proteomics_parquet: Path,
    human_fasta: Path,
    protected: ProtectedIdentityIndex,
    out_path: Path,
    max_pairs_per_enzyme: int = 40_000,
    seed: int = 20261128,
) -> dict[str, Any]:
    """Build expanded 4-protease Bekker-Jensen 2017 `bulk_observation` training pairs."""
    rng = random.Random(seed)
    bdf = pd.read_parquet(
        bulk_proteomics_parquet,
        columns=["source", "granularity", "digestion_enzyme", "peptide"],
    )
    bdf = bdf[
        (bdf["source"] == "Bekker-Jensen_2017")
        & (bdf["granularity"] == "peptide")
        & (bdf["digestion_enzyme"].isin(ENZYME_MAP))
    ]

    det_by_enz: dict[str, set[str]] = defaultdict(set)
    for enz_raw, pep in zip(bdf["digestion_enzyme"], bdf["peptide"]):
        pep_s = str(pep or "").strip().upper()
        if is_canonical_peptide(pep_s) and 7 <= len(pep_s) <= 30:
            det_by_enz[ENZYME_MAP[str(enz_raw)]].add(pep_s)

    proteins, _ = read_proteome_fasta_with_reviewed(human_fasta)
    all_detected = sorted({p for s in det_by_enz.values() for p in s})
    mappings = map_peptides(all_detected, proteins)

    prot_peps_by_enz: dict[str, dict[str, list[tuple[str, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for enz_key, peps in det_by_enz.items():
        for pep in sorted(peps):
            if protected.reason(pep) is not None:
                continue
            hits = mappings.get(pep) or []
            if not hits:
                continue
            pid, start = hits[0]
            prot_peps_by_enz[enz_key][pid].append((pep, start))

    rows: list[dict[str, Any]] = []
    by_enzyme: dict[str, int] = {}
    obs_idx = 0
    for enz_key in ("chymotrypsin", "gluc", "lysc", "trypsin_p"):
        cut_re = ENZYME_CUT_RE[enz_key]
        enz_detected = det_by_enz[enz_key]
        pair_candidates: list[tuple[str, str, int, str, int]] = []

        for pid, pos_list in sorted(prot_peps_by_enz[enz_key].items()):
            seq = proteins[pid].sequence
            cuts = [0] + [m.start() for m in cut_re.finditer(seq)] + [len(seq)]
            undets: list[tuple[str, int]] = []
            for a, b in zip(cuts[:-1], cuts[1:]):
                frag = seq[a:b]
                if 7 <= len(frag) <= 30 and is_canonical_peptide(frag):
                    if frag not in enz_detected and protected.reason(frag) is None:
                        undets.append((frag, a))
            if not undets:
                continue
            by_len: dict[int, list[tuple[str, int]]] = defaultdict(list)
            for frag, st in undets:
                by_len[len(frag)].append((frag, st))
            used_neg: set[str] = set()
            for pos_pep, pos_st in pos_list:
                pool = []
                for dl in (0, -1, 1, -2, 2):
                    for cand in by_len.get(len(pos_pep) + dl, ()):
                        if cand[0] not in used_neg:
                            pool.append(cand)
                    if pool:
                        break
                if not pool:
                    continue
                neg_pep, neg_st = rng.choice(pool)
                used_neg.add(neg_pep)
                pair_candidates.append((pid, pos_pep, pos_st, neg_pep, neg_st))

        if len(pair_candidates) > max_pairs_per_enzyme:
            rng.shuffle(pair_candidates)
            pair_candidates = pair_candidates[:max_pairs_per_enzyme]

        by_enzyme[enz_key] = len(pair_candidates)
        for pid, pos_pep, pos_st, neg_pep, neg_st in pair_candidates:
            seq = proteins[pid].sequence
            pair_id = f"bj2017_v23:{enz_key}:{obs_idx}"
            for is_pos, pep_s, st in ((True, pos_pep, pos_st), (False, neg_pep, neg_st)):
                j_d = junction(seq, st, len(pep_s), flank_length=15)
                role = "pos" if is_pos else "neg"
                obs = Observation(
                    observation_id=f"bj2017_v23:{enz_key}:{obs_idx}:{role}",
                    source="hitlist217_bulk_bekker_jensen",
                    split="train",
                    group_id=f"bj2017_v23:{enz_key}:{pid}",
                    endpoint="bulk_observation",
                    peptide=pep_s,
                    mhc=(),
                    value=1.0 if is_pos else 0.0,
                    unit="binary",
                    qualifier="=",
                    context=Context(
                        source_n=j_d["source_n"],
                        source_c=j_d["source_c"],
                        n_terminus=j_d["n_terminus"],
                        c_terminus=j_d["c_terminus"],
                        enzyme=enz_key,
                    ),
                    provenance={
                        "study_id": f"28561026:{enz_key}",
                        "pmid": "28561026",
                        "pair_id": pair_id,
                        "enzyme": enz_key,
                        "protein_id": pid,
                        "measurement_lineage": f"bj2017_v23:{enz_key}:{pid}:{pep_s}:{role}",
                    },
                )
                rows.append(obs.to_dict())
            obs_idx += 1

    _write_jsonl(out_path, rows)
    return {
        "source": "hitlist217_bulk_bekker_jensen",
        "pairs_by_enzyme": by_enzyme,
        "total_pairs": sum(by_enzyme.values()),
        "total_rows_written": len(rows),
        "out_path": str(out_path),
        "sha256": _sha256_file(out_path),
    }


def allocate_v23_elution_strata(
    stratum_weights: dict[tuple[str, ...], float],
    stratum_counts: dict[tuple[str, ...], int],
    *,
    budgets: dict[str, float],
    max_total_fraction_of_elution: float = 0.30,
) -> dict[tuple[str, ...], float]:
    """Carve bounded mass for v2.3 elution strata purely from existing elution strata.

    Non-elution strata (affinity, bulk, stability, T-cell, qualitative binding) remain
    100% unchanged to the last floating-point digit.
    """
    out = dict(stratum_weights)
    active_groups: dict[str, list[tuple[str, ...]]] = {}
    for prefix, target_mass in budgets.items():
        if target_mass <= 0:
            continue
        keys = [
            k
            for k, n in stratum_counts.items()
            if n > 0 and len(k) >= 2 and any(str(part).startswith(prefix) for part in k[1:])
        ]
        if keys:
            active_groups[prefix] = keys

    if not active_groups:
        return out

    new_keys_all = {k for ks in active_groups.values() for k in ks}
    elut_existing = [
        k
        for k, w in out.items()
        if w > 0 and len(k) >= 1 and k[0] == "elution_observation" and k not in new_keys_all
    ]
    elut_total = sum(out[k] for k in elut_existing)
    if elut_total <= 0:
        return out

    requested = sum(budgets[p] for p in active_groups)
    cap = elut_total * max_total_fraction_of_elution
    scale_req = min(1.0, cap / requested) if requested > 0 else 0.0
    actual_carve = requested * scale_req

    keep_factor = (elut_total - actual_carve) / elut_total
    for k in elut_existing:
        out[k] = out[k] * keep_factor

    for prefix, keys in active_groups.items():
        mass = budgets[prefix] * scale_req
        sqrt_tot = sum(stratum_counts[k] ** 0.5 for k in keys)
        for k in keys:
            out[k] = mass * ((stratum_counts[k] ** 0.5) / sqrt_tot)
    return out
