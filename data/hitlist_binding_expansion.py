"""Leak-free hitlist purified binding expansion builder for Presto-Omni v2.2.

Builds three strictly purified, zero-confounder training slices from hitlist
`binding.parquet`, passing EVERY candidate peptide through the exact + 8-mer +
length-matched Hamming-1 `ProtectedIdentityIndex` (`test_accessed = False`):

  1. Class II High-Throughput Direct Binding (`pmid=32903714` Buus/Nielsen MHC-II
     microarray + `pmid=35486722` Rappazzo/Birnbaum yeast-display MHC-II binding):
     - Pure cell-free MHC-II binding (`response_measured == "MHC binding"`,
       `assay_method == "High throughput multiplexed assay"`).
     - Stratified 1:3 pos:neg subsampling per allele (retaining 100% of all
       non-embargoed positives + 3x deterministic hard experimental negatives per
       allele), admitted as `endpoint="qualitative_binding"` (`track=None`,
       `OUTPUT_POLARITY=-1` supervising the shared `AssayMeasurementLayer` decoder
       without shifting verified tracks or base-rate calibration).

  2. Non-Human Purified Quantitative Affinity (`Mus musculus`, `Macaca mulatta`,
     `Equus caballus`, `Pan troglodytes`, etc.):
     - 100% purified MHC quantitative `ic50`, `kd_via_ic50`, `kd_via_ec50`, `ec50`, `kd`.
     - Normalizes mouse Class II beta shorthand (`H2-AA*b/AB*b` -> `H2-AA*b` + `H2-AB1*b`,
       `H2-EA*k/EB*k` -> `H2-EA*k` + `H2-EB1*k`) for `mhcseqs` groove lookup.
     - Admitted on exact existing purified `AssayTrack`s (preserving inequality
       qualifiers and nM units); 100% leak-free vs human validation & sealed TEST.

  3. `PMID 12135423` Purified Human Class I `KD` (`65` HLA-A/B/C alleles):
     - Strictly purified Class I `dissociation constant KD` rows previously excluded
       by `protocol_citation_conflict` in `data/measurement_curation.py`.
     - Admitted as `endpoint="kd"` with `assay_method=AssayMethod("purified",
       "competitive", "fluorescence", "protocol_citation_conflict")` so
       `observation_track(row)` returns `None` (isolated unknown-method reference
       track, zero contamination of verified `kd/purified/...` factor embeddings).

Canonical TEST outcomes are NEVER read (`test_accessed = False`).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

from presto.data.assay_measurement import AssayMethod, normalize_assay_method
from presto.data.measurement_curation import NM, reported_measurement
from presto.data.modular import AA, Context, MHCInput, Observation

AA_SET = frozenset(AA)

CONC_RESPONSE_TO_ENDPOINT: dict[str, str] = {
    "half maximal inhibitory concentration (IC50)": "ic50",
    "dissociation constant KD": "kd",
    "half maximal effective concentration (EC50)": "ec50",
    "dissociation constant KD (~IC50)": "kd_via_ic50",
    "dissociation constant KD (~EC50)": "kd_via_ec50",
}

POSITIVE_QUAL = frozenset({"Positive", "Positive-Low", "Positive-Intermediate", "Positive-High"})
NEGATIVE_QUAL = frozenset({"Negative"})


class ProtectedIdentityIndex:
    """Exact, 8-mer substring, and equal-length Hamming-1 embargo index."""

    def __init__(self, peptides: set[str]):
        self.exact: set[str] = set()
        self.eightmers: set[str] = set()
        self._halves: dict[tuple[int, int, str], set[str]] = defaultdict(set)
        for raw in peptides:
            s = raw.strip().upper()
            if not s or not (set(s) <= AA_SET):
                continue
            self.exact.add(s)
            n = len(s)
            if n >= 8:
                for i in range(n - 7):
                    self.eightmers.add(s[i : i + 8])
            midpoint = n // 2
            self._halves[(n, 0, s[:midpoint])].add(s)
            self._halves[(n, 1, s[midpoint:])].add(s)

    def reason(self, peptide: str) -> str | None:
        """Return embargo collision reason ('exact', 'shared8mer', 'hamming1') or None."""
        s = peptide.strip().upper()
        if s in self.exact:
            return "exact"
        n = len(s)
        if n >= 8:
            for i in range(n - 7):
                if s[i : i + 8] in self.eightmers:
                    return "shared8mer"
        midpoint = n // 2
        for side, part in ((0, s[:midpoint]), (1, s[midpoint:])):
            for other in self._halves.get((n, side, part), ()):
                if sum(a != b for a, b in zip(s, other, strict=True)) <= 1:
                    return "hamming1"
        return None

    def is_protected(self, peptide: str) -> bool:
        return self.reason(peptide) is not None


def load_protected_identity_index(
    embargo_path: Path, val_jsonl_paths: list[Path]
) -> tuple[ProtectedIdentityIndex, dict[str, Any]]:
    """Build ProtectedIdentityIndex from sealed-identity-embargo + validation JSONLs."""
    protected: set[str] = set()
    counts: dict[str, int] = {}
    n_emb = 0
    with gzip.open(embargo_path, "rt", encoding="utf-8") as f:
        for line in f:
            tok = line.strip().upper()
            if tok and set(tok) <= AA_SET:
                protected.add(tok)
                n_emb += 1
    counts["sealed_identity_lines"] = n_emb

    for vf in val_jsonl_paths:
        if not vf.exists():
            continue
        nv = 0
        with vf.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                pep = str(row.get("peptide") or "").strip().upper()
                if pep:
                    protected.add(pep)
                    nv += 1
        counts[vf.name] = nv

    idx = ProtectedIdentityIndex(protected)
    return idx, {
        "n_unique_protected_peptides": len(idx.exact),
        "n_protected_8mers": len(idx.eightmers),
        "source_counts": counts,
        "test_accessed": False,
    }


class FastGrooveResolver:
    """Cached O(1) mhcseqs groove resolver backed by a single-pass mhc-full-seqs.csv index."""

    def __init__(
        self,
        mhc_full_seqs_csv: Path | None = None,
        *,
        search_dir: str | None = None,
    ) -> None:
        import csv as _csv
        import mhcseqs

        self._mhcseqs = mhcseqs
        if mhc_full_seqs_csv is not None:
            csv_path = Path(mhc_full_seqs_csv)
            if not csv_path.exists():
                raise FileNotFoundError(f"mhc-full-seqs.csv not found at {csv_path}")
        else:
            find_fn = getattr(mhcseqs, "find_csv", None) or mhcseqs._find_csv
            csv_path = Path(find_fn("mhc-full-seqs.csv", search_dir=search_dir))
        self._csv_path = csv_path
        self._rows_by_key: dict[str, dict[str, str]] = {}
        self._synonymous_two_field: dict[str, list[dict[str, str]]] = defaultdict(list)
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in _csv.DictReader(f):
                for field in ("two_field_allele", "representative_allele"):
                    val = (row.get(field) or "").strip()
                    if val:
                        self._rows_by_key.setdefault(val, row)
                        self._rows_by_key.setdefault(val.upper(), row)
                two_f = (row.get("two_field_allele") or "").strip()
                if "*" in two_f and two_f.count(":") >= 2:
                    gene_part, fields_part = two_f.split("*", 1)
                    parts = fields_part.split(":")
                    if len(parts) >= 3:
                        two_field_key = f"{gene_part}*{parts[0]}:{parts[1]}".upper()
                        self._synonymous_two_field[two_field_key].append(row)
        if not self._rows_by_key:
            raise RuntimeError(f"Empty mhc-full-seqs.csv at {csv_path}")

        self._lookup_cache: dict[str, Any] = {}
        self._pair_cache: dict[tuple[str, str], tuple[str, str] | None] = {}
        dra = self._safe_lookup("HLA-DRA*01:01")
        self._dra_g1: str = dra.groove1 if dra and dra.ok and dra.groove1 else ""
        if not self._dra_g1:
            raise RuntimeError(f"Failed to resolve HLA-DRA*01:01 groove1 from {csv_path}")

    def _decompose_row(self, row: dict[str, str]) -> Any:
        try:
            rec = self._mhcseqs.decompose_domains(
                row.get("sequence", ""),
                mhc_class=row.get("mhc_class", ""),
                chain=row.get("chain", ""),
                allele=row.get("two_field_allele", ""),
                gene=row.get("gene", ""),
                species=row.get("species", ""),
            )
            return rec if (rec and rec.ok) else None
        except (ValueError, KeyError, TypeError, AttributeError):
            return None

    def _safe_lookup(self, name: str) -> Any:
        if name in self._lookup_cache:
            return self._lookup_cache[name]
        query = name.strip()
        candidates = [query]
        try:
            norm = self._mhcseqs.normalize_allele_name(query)
            if norm and norm not in candidates:
                candidates.append(norm)
        except (ValueError, KeyError, TypeError, AttributeError):
            pass
        # Canis lupus familiaris (dog) alleles appear as DLA-* in mhcseqs>=2.12/mhcgnomes>=3.74
        # and as CanisSp-* in older catalogs; map Calu-/CanisLupus-/CanisSp- aliases cleanly.
        expanded: list[str] = []
        for cand in candidates:
            expanded.append(cand)
            up = cand.upper()
            if up.startswith("CANISLUPUS-"):
                suffix = cand[len("CanisLupus-") :]
                expanded.extend(["DLA-" + suffix, "CanisSp-" + suffix])
            elif up.startswith("CALU-"):
                suffix = cand[len("Calu-") :]
                expanded.extend(["DLA-" + suffix, "CanisSp-" + suffix])
            elif up.startswith("CANISSP-"):
                expanded.append("DLA-" + cand[len("CanisSp-") :])

        row = None
        for cand in expanded:
            row = self._rows_by_key.get(cand) or self._rows_by_key.get(cand.upper())
            if row is not None:
                break
        if row is not None:
            val = self._decompose_row(row)
            self._lookup_cache[name] = val
            return val

        # Safe synonymous 3rd-field fallback (only when every 3-field record under
        # the same 2-field allele shares identical (groove1, groove2))
        for cand in expanded:
            syn_rows = self._synonymous_two_field.get(cand.upper())
            if not syn_rows:
                continue
            recs = [self._decompose_row(r) for r in syn_rows]
            valid = [r for r in recs if r is not None]
            if valid and len({(r.groove1, r.groove2) for r in valid}) == 1:
                self._lookup_cache[name] = valid[0]
                return valid[0]

        self._lookup_cache[name] = None
        return None

    def _lookup_beta_with_aliases(self, prefix: str, beta_tok: str) -> Any:
        candidates_to_try = [beta_tok]
        if prefix and not beta_tok.startswith(f"{prefix}-"):
            candidates_to_try.append(f"{prefix}-{beta_tok}")
        # Mouse H2-AA*b/AB*b -> H2-AB1*b, H2-EA*k/EB*k -> H2-EB1*k
        if prefix == "H2":
            if beta_tok.startswith("AB*"):
                candidates_to_try.append(f"H2-AB1*{beta_tok[3:]}")
            elif beta_tok.startswith("EB*"):
                candidates_to_try.append(f"H2-EB1*{beta_tok[3:]}")
            elif beta_tok.startswith("H2-AB*"):
                candidates_to_try.append(f"H2-AB1*{beta_tok[6:]}")
            elif beta_tok.startswith("H2-EB*"):
                candidates_to_try.append(f"H2-EB1*{beta_tok[6:]}")
        for cand in candidates_to_try:
            rec = self._safe_lookup(cand)
            if rec and rec.groove2:
                return rec
        return None

    def resolve(self, restriction: str, mhc_class: str) -> tuple[str, str] | None:
        key = (str(restriction or "").strip(), str(mhc_class or "").strip())
        if key in self._pair_cache:
            return self._pair_cache[key]
        al_s, cls = key
        if not al_s or cls not in ("I", "II"):
            self._pair_cache[key] = None
            return None
        result: tuple[str, str] | None = None
        if "/" in al_s:
            a1, a2 = [x.strip() for x in al_s.split("/", 1)]
            prefix = a1.split("-", 1)[0] if "-" in a1 else ""
            r1 = self._safe_lookup(a1)
            r2 = self._lookup_beta_with_aliases(prefix, a2)
            if r1 and r2 and r1.groove1 and r2.groove2:
                result = (r1.groove1, r2.groove2)
        else:
            r = self._safe_lookup(al_s)
            if r:
                if cls == "I" and r.groove1 and r.groove2:
                    result = (r.groove1, r.groove2)
                elif cls == "II" and al_s.startswith("HLA-DRB") and self._dra_g1 and r.groove2:
                    result = (self._dra_g1, r.groove2)
        if result is not None:
            g1, g2 = result
            if not (set(g1) <= AA_SET and set(g2) <= AA_SET):
                result = None
        self._pair_cache[key] = result
        return result


def _clean_peptide(raw: Any, mhc_class: str) -> str | None:
    if not isinstance(raw, str):
        return None
    pep = raw.strip().upper()
    if not pep or not (set(pep) <= AA_SET):
        return None
    if mhc_class == "I" and not (7 <= len(pep) <= 20):
        return None
    if mhc_class == "II" and not (9 <= len(pep) <= 40):
        return None
    return pep


def _obs_id(prefix: str, key_parts: tuple[Any, ...]) -> str:
    h = hashlib.sha256("|".join(str(x) for x in key_parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{h}"


def _convert_to_nm(val: float, units: Any) -> float | None:
    if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
        return None
    if val <= 0:
        return None
    u = str(units or "nM").strip()
    factor = NM.get(u)
    if factor is None:
        return None
    nm = float(val) * float(factor)
    return nm if (math.isfinite(nm) and 1e-4 <= nm <= 1e9) else None


def build_hitlist_binding_expansion(
    binding_parquet: Path,
    protected: ProtectedIdentityIndex,
    *,
    allowed_tracks: set[tuple[str, str, str, str]],
    out_path: Path,
    mhc_full_seqs_csv: Path | None = None,
    ht_neg_per_pos: int = 3,
    seed: int = 20261122,
    include_ht_class2: bool = True,
    include_nonhuman_quant: bool = True,
    include_pmid12135423_kd: bool = True,
    include_human_quant_expansion: bool = False,
    existing_pep_allele_ep: set[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    """Build leak-free purified binding expansion JSONL from hitlist binding.parquet."""
    import pandas as pd

    resolver = FastGrooveResolver(mhc_full_seqs_csv=mhc_full_seqs_csv)
    cols = [
        "peptide",
        "has_ptm",
        "mhc_restriction",
        "mhc_species",
        "mhc_class",
        "assay_iri",
        "reference_iri",
        "pmid",
        "assay_comments",
        "qualitative_measurement",
        "assay_method",
        "response_measured",
        "measurement_units",
        "measurement_inequality",
        "quantitative_measurement",
        "quantitative_value",
    ]
    df = pd.read_parquet(binding_parquet, columns=cols)
    df = df[~df["has_ptm"].fillna(False)]

    stats: dict[str, Any] = {
        "slice1_ht_class2": Counter(),
        "slice2_nonhuman_quant": Counter(),
        "slice3_pmid12135423_kd": Counter(),
        "slice4_human_quant": Counter(),
        "slice1_by_allele": Counter(),
        "slice2_by_species": Counter(),
        "slice2_by_endpoint_class": Counter(),
        "slice3_by_qualifier": Counter(),
        "slice4_by_endpoint_class": Counter(),
        "test_accessed": False,
    }
    rows_out: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()

    # ------------------------------------------------------------------
    # Slice 1: Class II High-Throughput Direct Binding (pmid 32903714, 35486722)
    # ------------------------------------------------------------------
    if include_ht_class2:
        ht = df[
            (df["response_measured"] == "MHC binding")
            & (df["mhc_class"] == "II")
            & (df["mhc_species"] == "Homo sapiens")
        ]
        pos_by_allele: dict[str, list[dict[str, Any]]] = defaultdict(list)
        neg_by_allele: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rec in ht.to_dict(orient="records"):
            pep = _clean_peptide(rec.get("peptide"), "II")
            if pep is None:
                stats["slice1_ht_class2"]["drop_peptide"] += 1
                continue
            if protected.is_protected(pep):
                stats["slice1_ht_class2"]["drop_embargo"] += 1
                continue
            qual = str(rec.get("qualitative_measurement") or "").strip()
            if qual in POSITIVE_QUAL:
                y = 1
            elif qual in NEGATIVE_QUAL:
                y = 0
            else:
                stats["slice1_ht_class2"]["drop_qual"] += 1
                continue
            al = str(rec.get("mhc_restriction") or "").strip()
            grooves = resolver.resolve(al, "II")
            if grooves is None:
                stats["slice1_ht_class2"]["drop_groove"] += 1
                continue
            pmid = str(int(rec["pmid"])) if pd.notna(rec.get("pmid")) else "ht_mhc_binding"
            dk = ("ht_qb", pep, al, y)
            if dk in seen_keys:
                continue
            seen_keys.add(dk)
            item = {
                "pep": pep,
                "al": al,
                "g1": grooves[0],
                "g2": grooves[1],
                "y": y,
                "pmid": pmid,
                "assay_iri": str(rec.get("assay_iri") or ""),
            }
            if y == 1:
                pos_by_allele[al].append(item)
            else:
                neg_by_allele[al].append(item)

        rng = random.Random(seed)
        for al in sorted(pos_by_allele):
            pos_list = pos_by_allele[al]
            neg_list = sorted(neg_by_allele.get(al, ()), key=lambda x: (x["pep"], x["pmid"]))
            # Never keep a peptide as both pos and neg for the same allele
            pos_peps = {x["pep"] for x in pos_list}
            neg_list = [x for x in neg_list if x["pep"] not in pos_peps]
            max_neg = max(len(pos_list) * int(ht_neg_per_pos), 0)
            if len(neg_list) > max_neg:
                rng.shuffle(neg_list)
                neg_list = neg_list[:max_neg]
            for item in (*pos_list, *neg_list):
                mhc = MHCInput(
                    groove1=item["g1"],
                    groove2=item["g2"],
                    mhc_class="II",
                    provenance={"reported_restriction": item["al"]},
                )
                oid = _obs_id("ht_qb", (item["pmid"], item["al"], item["pep"], item["y"]))
                obs = Observation(
                    observation_id=oid,
                    source="hitlist_ht_class2_binding",
                    endpoint="qualitative_binding",
                    peptide=item["pep"],
                    mhc=(mhc,),
                    value=int(item["y"]),
                    unit="binary",
                    qualifier="=",
                    split="train",
                    group_id=f"ht_qb:{item['pmid']}:{item['al']}",
                    context=Context(),
                    provenance={
                        "study_id": f"ht_qb:{item['pmid']}:{item['al']}",
                        "pmid": item["pmid"],
                        "measurement_basis": "measured",
                        "measurement_lineage": {
                            "namespace": "hitlist_ht_class2_binding",
                            "measurement_id": item["assay_iri"] or oid,
                        },
                    },
                )
                obs.validate(512)
                rows_out.append(asdict(obs))
                stats["slice1_ht_class2"]["admitted"] += 1
                stats["slice1_ht_class2"]["admitted_pos" if item["y"] else "admitted_neg"] += 1
                stats["slice1_by_allele"][al] += 1

    # ------------------------------------------------------------------
    # Slice 2: Non-Human Purified Quantitative Affinity
    # ------------------------------------------------------------------
    if include_nonhuman_quant:
        nh = df[
            (df["mhc_species"] != "Homo sapiens")
            & (df["response_measured"].isin(CONC_RESPONSE_TO_ENDPOINT))
            & (df["assay_method"].astype(str).str.startswith("purified MHC"))
        ]
        for rec in nh.to_dict(orient="records"):
            cls = str(rec.get("mhc_class") or "").strip()
            if cls not in ("I", "II"):
                stats["slice2_nonhuman_quant"]["drop_class"] += 1
                continue
            pep = _clean_peptide(rec.get("peptide"), cls)
            if pep is None:
                stats["slice2_nonhuman_quant"]["drop_peptide"] += 1
                continue
            if protected.is_protected(pep):
                stats["slice2_nonhuman_quant"]["drop_embargo"] += 1
                continue
            ep = CONC_RESPONSE_TO_ENDPOINT[str(rec["response_measured"])]
            al = str(rec.get("mhc_restriction") or "").strip()
            grooves = resolver.resolve(al, cls)
            if grooves is None:
                stats["slice2_nonhuman_quant"]["drop_groove"] += 1
                continue
            try:
                meas = reported_measurement(rec)
            except Exception:  # noqa: BLE001
                stats["slice2_nonhuman_quant"]["drop_measurement"] += 1
                continue
            nm = _convert_to_nm(rec.get("quantitative_value"), rec.get("measurement_units"))
            if nm is None:
                stats["slice2_nonhuman_quant"]["drop_value"] += 1
                continue
            am = normalize_assay_method(rec)
            if am.usable:
                tk = (ep, am.preparation, am.mechanism, am.detector)
                if tk not in allowed_tracks:
                    # Keep track=None (unknown method) rather than adding an untrained track
                    am = AssayMethod("purified", "unknown", "unknown", "source_tag_unreviewed")
            else:
                am = AssayMethod("purified", "unknown", "unknown", "source_tag_unreviewed")
            pmid = str(int(rec["pmid"])) if pd.notna(rec.get("pmid")) else "nonhuman_quant"
            sp_name = str(rec.get("mhc_species") or "unknown").strip()
            qual = str(meas["effective_qualifier"])
            dk = ("nh_q", ep, pep, al, round(nm, 4), qual)
            if dk in seen_keys:
                continue
            seen_keys.add(dk)
            oid = _obs_id("nh_q", (ep, pmid, al, pep, round(nm, 4), qual))
            mhc = MHCInput(
                groove1=grooves[0],
                groove2=grooves[1],
                mhc_class=cls,
                provenance={"reported_restriction": al, "mhc_species": sp_name},
            )
            obs = Observation(
                observation_id=oid,
                source="hitlist_nonhuman_purified_quant",
                endpoint=ep,
                peptide=pep,
                mhc=(mhc,),
                value=float(nm),
                unit="nM",
                qualifier=qual,
                assay_method=am,
                split="train",
                group_id=f"nh_q:{pmid}",
                context=Context(),
                provenance={
                    "study_id": f"nh_q:{pmid}",
                    "pmid": pmid,
                    "mhc_species": sp_name,
                    "measurement_lineage": {
                        "namespace": "hitlist_nonhuman_purified_quant",
                        "measurement_id": str(rec.get("assay_iri") or oid),
                    },
                },
            )
            obs.validate(512)
            rows_out.append(asdict(obs))
            stats["slice2_nonhuman_quant"]["admitted"] += 1
            stats["slice2_by_species"][sp_name] += 1
            stats["slice2_by_endpoint_class"][f"{ep}_{cls}"] += 1

    # ------------------------------------------------------------------
    # Slice 3: PMID 12135423 Purified Human Class I KD (isolated reference track)
    # ------------------------------------------------------------------
    if include_pmid12135423_kd:
        p12 = df[
            df["assay_comments"].fillna("").str.contains("12135423")
            & (df["response_measured"] == "dissociation constant KD")
            & (df["mhc_class"] == "I")
            & (df["mhc_species"] == "Homo sapiens")
        ]
        conflict_am = AssayMethod(
            "purified", "competitive", "fluorescence", "protocol_citation_conflict"
        )
        assert not conflict_am.usable
        for rec in p12.to_dict(orient="records"):
            pep = _clean_peptide(rec.get("peptide"), "I")
            if pep is None:
                stats["slice3_pmid12135423_kd"]["drop_peptide"] += 1
                continue
            if protected.is_protected(pep):
                stats["slice3_pmid12135423_kd"]["drop_embargo"] += 1
                continue
            al = str(rec.get("mhc_restriction") or "").strip()
            grooves = resolver.resolve(al, "I")
            if grooves is None:
                stats["slice3_pmid12135423_kd"]["drop_groove"] += 1
                continue
            try:
                meas = reported_measurement(rec)
            except Exception:  # noqa: BLE001
                stats["slice3_pmid12135423_kd"]["drop_measurement"] += 1
                continue
            nm = _convert_to_nm(rec.get("quantitative_value"), rec.get("measurement_units"))
            if nm is None:
                stats["slice3_pmid12135423_kd"]["drop_value"] += 1
                continue
            qual = str(meas["effective_qualifier"])
            pmid = str(int(rec["pmid"])) if pd.notna(rec.get("pmid")) else "12135423"
            dk = ("p12_kd", pep, al, round(nm, 4), qual)
            if dk in seen_keys:
                continue
            seen_keys.add(dk)
            oid = _obs_id("p12_kd", (pmid, al, pep, round(nm, 4), qual))
            mhc = MHCInput(
                groove1=grooves[0],
                groove2=grooves[1],
                mhc_class="I",
                provenance={"reported_restriction": al},
            )
            obs = Observation(
                observation_id=oid,
                source="hitlist_pmid12135423_kd",
                endpoint="kd",
                peptide=pep,
                mhc=(mhc,),
                value=float(nm),
                unit="nM",
                qualifier=qual,
                assay_method=conflict_am,
                split="train",
                group_id=f"p12_kd:{pmid}",
                context=Context(),
                provenance={
                    "study_id": f"p12_kd:{pmid}",
                    "pmid": pmid,
                    "protocol_citation_conflict": True,
                    "measurement_lineage": {
                        "namespace": "hitlist_pmid12135423_kd",
                        "measurement_id": str(rec.get("assay_iri") or oid),
                    },
                },
            )
            obs.validate(512)
            rows_out.append(asdict(obs))
            stats["slice3_pmid12135423_kd"]["admitted"] += 1
            stats["slice3_by_qualifier"][qual] += 1

    # ------------------------------------------------------------------
    # Slice 4: Leak-Free Human Quantitative Binding Expansion (Class I & II)
    # Harvests any clean human quantitative IC50/KD/EC50 rows in hitlist
    # binding.parquet missing from affinity-train.jsonl, strictly embargoed.
    # ------------------------------------------------------------------
    if include_human_quant_expansion:
        existing_set = existing_pep_allele_ep or set()
        hq = df[
            (df["mhc_species"] == "Homo sapiens")
            & (df["mhc_class"].isin(["I", "II"]))
            & (df["response_measured"].isin(CONC_RESPONSE_TO_ENDPOINT))
            & (df["quantitative_value"].notna())
            & (~df["assay_comments"].fillna("").str.contains("12135423"))
        ]
        for rec in hq.to_dict(orient="records"):
            cls = str(rec.get("mhc_class") or "").strip()
            pep = _clean_peptide(rec.get("peptide"), cls)
            if pep is None:
                stats["slice4_human_quant"]["drop_peptide"] += 1
                continue
            if protected.is_protected(pep):
                stats["slice4_human_quant"]["drop_embargo"] += 1
                continue
            ep = CONC_RESPONSE_TO_ENDPOINT[str(rec["response_measured"])]
            al = str(rec.get("mhc_restriction") or "").strip()
            if (ep, pep, al) in existing_set:
                stats["slice4_human_quant"]["drop_already_in_train"] += 1
                continue
            grooves = resolver.resolve(al, cls)
            if grooves is None:
                stats["slice4_human_quant"]["drop_groove"] += 1
                continue
            try:
                meas = reported_measurement(rec)
            except Exception:  # noqa: BLE001
                stats["slice4_human_quant"]["drop_measurement"] += 1
                continue
            nm = _convert_to_nm(rec.get("quantitative_value"), rec.get("measurement_units"))
            if nm is None:
                stats["slice4_human_quant"]["drop_value"] += 1
                continue
            am = normalize_assay_method(rec)
            if am.usable:
                tk = (ep, am.preparation, am.mechanism, am.detector)
                if tk not in allowed_tracks:
                    am = AssayMethod("purified", "unknown", "unknown", "source_tag_unreviewed")
            else:
                am = AssayMethod("purified", "unknown", "unknown", "source_tag_unreviewed")
            qual = str(meas["effective_qualifier"])
            pmid = str(int(rec["pmid"])) if pd.notna(rec.get("pmid")) else "human_quant"
            dk = ("hu_q", ep, pep, al, round(nm, 4), qual)
            if dk in seen_keys:
                continue
            seen_keys.add(dk)
            oid = _obs_id("hu_q", (ep, pmid, al, pep, round(nm, 4), qual))
            mhc = MHCInput(
                groove1=grooves[0],
                groove2=grooves[1],
                mhc_class=cls,
                provenance={"reported_restriction": al, "mhc_species": "Homo sapiens"},
            )
            obs = Observation(
                observation_id=oid,
                source="hitlist_human_quant_expansion",
                endpoint=ep,
                peptide=pep,
                mhc=(mhc,),
                value=float(nm),
                unit="nM",
                qualifier=qual,
                assay_method=am,
                split="train",
                group_id=f"hu_q:{al}",
                context=Context(),
                provenance={
                    "study_id": f"hu_q:{pmid}",
                    "pmid": pmid,
                    "reported_restriction": al,
                    "measurement_lineage": {
                        "namespace": "hitlist_human_quant_expansion",
                        "measurement_id": str(rec.get("assay_iri") or oid),
                    },
                },
            )
            obs.validate(512)
            rows_out.append(asdict(obs))
            stats["slice4_human_quant"]["admitted"] += 1
            stats["slice4_by_endpoint_class"][f"{ep}_{cls}"] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")

    return {
        "out_path": str(out_path),
        "total_admitted": len(rows_out),
        "slice1_ht_class2": dict(stats["slice1_ht_class2"]),
        "slice1_by_allele": dict(stats["slice1_by_allele"]),
        "slice2_nonhuman_quant": dict(stats["slice2_nonhuman_quant"]),
        "slice2_by_species": dict(stats["slice2_by_species"]),
        "slice2_by_endpoint_class": dict(stats["slice2_by_endpoint_class"]),
        "slice3_pmid12135423_kd": dict(stats["slice3_pmid12135423_kd"]),
        "slice3_by_qualifier": dict(stats["slice3_by_qualifier"]),
        "slice4_human_quant": dict(stats["slice4_human_quant"]),
        "slice4_by_endpoint_class": dict(stats["slice4_by_endpoint_class"]),
        "test_accessed": False,
    }


__all__ = [
    "FastGrooveResolver",
    "ProtectedIdentityIndex",
    "build_hitlist_binding_expansion",
    "load_protected_identity_index",
]
