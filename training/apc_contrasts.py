"""Source-backed TRAIN-only cross-condition APC selectivity, pair calibration, and yield ordering.

Combines three complementary signals strictly from admitted TRAIN elution observations:
1. Within-study multi-arm repertoire selectivity & BA-matched pair condition contrasts:
       [(logit(x_A | C_1) - logit(x_A | C_2)) - (logit(x_B | C_1) - logit(x_B | C_2))] > 0
   Including:
   - exclusive positive peptides across sibling arms of multi-arm studies,
   - BA-matched (positive, background) pairs under original vs sibling-arm APC (`apc-shuffle`),
   - BA-matched (positive, background) pairs under original vs stripped APC (`apc-unknown`).
   Intrinsic binding and pure condition intercepts cancel identically.
2. Calibrated two-sided hinged counterfactual ordering on configured host APM/HLA gene
   perturbations and cytokines (KO < KD < WT < KO+rescue < OE; IFNG absent < unknown < present).
3. Cross-locus HLA orthogonality (`signed_dir = 0.0`): HLA-A/B/C perturbations have zero effect
   on non-matching Class-I loci (preventing endogenous HLA-A/C KO from penalizing HLA-B rescue).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import hashlib

import torch
from torch.nn import functional as F

from ..data.modular import ContextAssertion, Exposure
from ..data.modular_context import (
    HLA_LOCUS_GENES,
    parse_gene_perturbation,
    supported_perturbation_genes,
)
from ..models.apc_state import factor_classes
from .modular_controls import study_of
from .processing_pairs import PAIR_KEY

QUANT_CONTRAST_KEY = "apc_quantitative_contrast"


def _active_apc_signature(
    context,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    """Canonical tuple of active (`present` or `absent`) interventions and cytokines."""
    interv = {}
    for k, v in context.interventions.items():
        if v.state == "present":
            interv[k] = "present"
    for k, v in context.intervention_overrides.items():
        if v.state == "present":
            interv[k] = "present"
        else:
            interv.pop(k, None)

    cyto = {}
    for k, v in context.cytokines.items():
        if v.state in ("present", "absent"):
            cyto[k] = v.state
    for k, v in context.cytokine_overrides.items():
        if v.state in ("present", "absent"):
            cyto[k] = v.state
        else:
            cyto.pop(k, None)
    return tuple(sorted(interv.items())), tuple(sorted(cyto.items()))


def _present_only_signature(context) -> tuple[tuple[str, ...], tuple[str, ...]]:
    interv, cyto = _active_apc_signature(context)
    present_i = tuple(k for k, s in interv if s == "present")
    present_c = tuple(k for k, s in cyto if s == "present")
    return present_i, present_c


def _has_any_apc_context(context) -> bool:
    return bool(
        context.cell_type is not None
        or context.cell_type_override is not None
        or context.components
        or context.component_overrides
        or context.cytokines
        or context.cytokine_overrides
        or context.interventions
        or context.intervention_overrides
        or context.processing_pathways
        or context.pathway_overrides
    )


def _strip_all_apc(row):
    return replace(
        row,
        context=replace(
            row.context,
            components={},
            component_overrides={},
            cell_type=None,
            cell_type_override=None,
            cytokines={},
            cytokine_overrides={},
            interventions={},
            intervention_overrides={},
            processing_pathways={},
            pathway_overrides={},
        ),
    )


def _copy_full_apc(row, donor_context):
    return replace(
        row,
        context=replace(
            row.context,
            components=dict(donor_context.components),
            component_overrides=dict(donor_context.component_overrides),
            cell_type=donor_context.cell_type,
            cell_type_override=donor_context.cell_type_override,
            cytokines=dict(donor_context.cytokines),
            cytokine_overrides=dict(donor_context.cytokine_overrides),
            interventions=dict(donor_context.interventions),
            intervention_overrides=dict(donor_context.intervention_overrides),
            processing_pathways=dict(donor_context.processing_pathways),
            pathway_overrides=dict(donor_context.pathway_overrides),
        ),
    )


def _mhc_groove_signature(row) -> tuple[str, str] | None:
    classes = {m.class_override or m.mhc_class for m in row.mhc}
    if len(classes) != 1:
        return None
    cls = next(iter(classes))
    if cls not in ("I", "II"):
        return None
    grooves = sorted({f"{m.groove1}|{m.groove2}" for m in row.mhc})
    if not grooves:
        return None
    digest = hashlib.sha256(";".join(grooves).encode()).hexdigest()[:16]
    return cls, digest


def _single_hla_locus(row) -> str | None:
    """Return `HLA-A`, `HLA-B`, or `HLA-C` if all candidates belong to one Class-I locus."""
    loci = set()
    for m in row.mhc:
        if (m.class_override or m.mhc_class) != "I":
            return None
        alleles = (m.provenance or {}).get("catalog_alleles") or ()
        if not alleles:
            return None
        for allele in alleles:
            for locus in ("HLA-A", "HLA-B", "HLA-C"):
                if allele.startswith(locus + "*"):
                    loci.add(locus)
    return next(iter(loci)) if len(loci) == 1 else None


def _apply_apc_condition(
    row,
    interventions: dict[str, str],
    cytokines: dict[str, str],
):
    return replace(
        row,
        context=replace(
            row.context,
            interventions={k: ContextAssertion(v) for k, v in interventions.items()},
            intervention_overrides={},
            cytokines={k: Exposure(v) for k, v in cytokines.items()},
            cytokine_overrides={},
        ),
    )


class APCContrastPool:
    """Deterministic TRAIN-only pool for cross-condition selectivity and yield ordering."""

    def __init__(self, index, config):
        self.selectivity_buckets: dict[str, list] = defaultdict(list)
        self.yield_buckets: dict[str, list] = defaultdict(list)
        self.quant_buckets: dict[str, list] = defaultdict(list)
        self.auxiliary_indices: set[int] = set()

        by_study: dict[tuple[str, str], dict] = defaultdict(lambda: defaultdict(list))
        by_groove: dict[tuple[str, str], dict] = defaultdict(lambda: defaultdict(list))
        by_class_unperturbed: dict[str, list[int]] = defaultdict(list)
        by_hla_locus: dict[str, list[int]] = defaultdict(list)
        by_present_claim: dict[tuple[str, str], list[int]] = defaultdict(list)
        train_pos_by_id: dict[str, int] = {}
        study_arm_donor_idx: dict[tuple[str, str], dict[tuple, int]] = defaultdict(dict)
        aux_pairs: list[tuple[int, int, str, str | None, tuple]] = []

        for idx, row in enumerate(index):
            if row.split != "train":
                continue
            if QUANT_CONTRAST_KEY in row.provenance:
                self.auxiliary_indices.add(idx)
                qmeta = row.provenance[QUANT_CONTRAST_KEY]
                plex_id = str(qmeta.get("plex_id") or "default_plex")
                fc = float(qmeta["target_log2_fc"])
                ctrl_i = dict(qmeta.get("control_interventions") or {})
                ctrl_c = dict(qmeta.get("control_cytokines") or {})
                self.quant_buckets[plex_id].append((idx, fc, ctrl_i, ctrl_c))
                continue
            if PAIR_KEY in row.provenance:
                classes = {m.class_override or m.mhc_class for m in row.mhc}
                if len(classes) == 1 and next(iter(classes)) in ("I", "II"):
                    cls = next(iter(classes))
                    study = study_of(row.provenance)
                    cond_key = _active_apc_signature(row.context)
                    aux_pairs.append(
                        (idx, -1, cls, str(study) if study is not None else None, cond_key)
                    )
                continue
            if row.endpoint != "elution_observation" or row.value != 1.0 or row.qualifier != "=":
                continue
            train_pos_by_id[row.observation_id] = idx
            gsig = _mhc_groove_signature(row)
            if gsig is None:
                continue
            cls, _ = gsig
            present_i, present_c = _present_only_signature(row.context)
            cond_key = (present_i, present_c)

            study = study_of(row.provenance)
            if study is not None:
                skey = (cls, str(study))
                by_study[skey][cond_key].append(idx)
                active_sig = _active_apc_signature(row.context)
                study_arm_donor_idx[skey].setdefault(active_sig, idx)
            by_groove[gsig][cond_key].append(idx)

            if not present_i and not present_c:
                by_class_unperturbed[cls].append(idx)
                locus = _single_hla_locus(row)
                if locus is not None:
                    by_hla_locus[locus].append(idx)

            for claim in present_i:
                by_present_claim[(cls, f"intervention:{claim}")].append(idx)
            for cyto in present_c:
                by_present_claim[(cls, f"cytokine:{cyto}")].append(idx)

        def _cond_dicts(cond_key):
            pi, pc = cond_key
            return {k: "present" for k in pi}, {k: "present" for k in pc}

        def _register_arm_contrasts(arms_map, *, max_pairs_per_arm: int, only_missing: bool):
            conds = sorted(arms_map.keys())
            if len(conds) < 2:
                return
            pep_sets = {c: {index[i].peptide for i in arms_map[c]} for c in conds}
            for i_a in range(len(conds)):
                for i_b in range(i_a + 1, len(conds)):
                    ca, cb = conds[i_a], conds[i_b]
                    diff_factors = (set(ca[0]) ^ set(cb[0])) | {
                        f"cytokine:{c}" for c in (set(ca[1]) ^ set(cb[1]))
                    }
                    if not diff_factors:
                        continue
                    target_factors = [
                        f
                        for f in sorted(diff_factors)
                        if not (only_missing and self.selectivity_buckets.get(f))
                    ]
                    if not target_factors:
                        continue
                    excl_a = [idx for idx in arms_map[ca] if index[idx].peptide not in pep_sets[cb]]
                    excl_b = [idx for idx in arms_map[cb] if index[idx].peptide not in pep_sets[ca]]
                    if not excl_a or not excl_b:
                        continue
                    ia_dict, ca_dict = _cond_dicts(ca)
                    ib_dict, cb_dict = _cond_dicts(cb)
                    n_pairs = min(max(len(excl_a), len(excl_b)), max_pairs_per_arm)
                    pairs = [
                        (
                            "arm_swap",
                            excl_a[k % len(excl_a)],
                            excl_b[k % len(excl_b)],
                            ia_dict,
                            ca_dict,
                            ib_dict,
                            cb_dict,
                        )
                        for k in range(n_pairs)
                    ]
                    for factor in target_factors:
                        self.selectivity_buckets[factor].extend(pairs)

        # Primary: strictly within-study multi-arm contrasts (zero cross-study confounding)
        for _, arms_map in sorted(by_study.items()):
            _register_arm_contrasts(arms_map, max_pairs_per_arm=256, only_missing=False)

        # Fallback ONLY for single-arm factors absent from multi-arm studies (e.g. HLA-DM)
        for _, arms_map in sorted(by_groove.items()):
            unperturbed = ((), ())
            if unperturbed not in arms_map or len(arms_map) < 2:
                continue
            sub_map = {k: v for k, v in arms_map.items() if k == unperturbed or k[0] or k[1]}
            _register_arm_contrasts(sub_map, max_pairs_per_arm=64, only_missing=True)

        # BA-matched TRAIN pair condition contrasts: original > apc-shuffle & original > apc-unknown
        unknown_by_study: dict[tuple[str, str], list] = defaultdict(list)
        shuffle_by_study: dict[tuple[str, str], list] = defaultdict(list)
        for bg_idx, _, cls, study_str, _ in aux_pairs:
            bg_row = index[bg_idx]
            pos_idx = train_pos_by_id.get(bg_row.provenance[PAIR_KEY])
            if pos_idx is None:
                continue
            pos_row = index[pos_idx]
            skey = (cls, study_str or "nostudy")
            if _has_any_apc_context(pos_row.context):
                if len(unknown_by_study[skey]) < 128:
                    unknown_by_study[skey].append(("pair_vs_unknown", pos_idx, bg_idx))
            if study_str is not None:
                arms = study_arm_donor_idx.get((cls, study_str), {})
                if len(arms) >= 2:
                    pos_sig = _active_apc_signature(pos_row.context)
                    for other_sig, donor_idx in sorted(arms.items()):
                        if other_sig != pos_sig and len(shuffle_by_study[skey]) < 192:
                            shuffle_by_study[skey].append(
                                ("pair_vs_sibling", pos_idx, bg_idx, donor_idx)
                            )
        for cls in ("I", "II"):
            unk_items = [
                item
                for (c, _), items in sorted(unknown_by_study.items())
                if c == cls
                for item in items
            ]
            if unk_items:
                self.selectivity_buckets[f"pair_vs_unknown:{cls}"].extend(unk_items)
            shuf_items = [
                item
                for (c, _), items in sorted(shuffle_by_study.items())
                if c == cls
                for item in items
            ]
            if shuf_items:
                self.selectivity_buckets[f"pair_vs_sibling:{cls}"].extend(shuf_items)

        # Signed counterfactual yield buckets + HLA cross-locus orthogonality
        configured_interventions = set(config.interventions)
        for claim in config.interventions:
            parsed = parse_gene_perturbation(claim)
            if parsed is None:
                continue
            kind, gene = parsed
            sign = -1.0 if kind in ("knockout", "knockdown") else 1.0
            severity = 0.5 if kind == "knockdown" else 1.0
            if gene in HLA_LOCUS_GENES:
                pool = by_hla_locus.get(gene, [])
                if not pool and gene == "HLA-B":
                    pool = by_present_claim.get(("I", f"intervention:{claim}"), [])
                for idx in pool[:256]:
                    if kind in ("knockout", "knockdown"):
                        self.yield_buckets[f"gene:{gene}"].append(
                            (idx, {claim: "present"}, {}, {}, {}, -1.0 * severity)
                        )
                    else:
                        ko_claim = f"knockout:{gene}"
                        self.yield_buckets[f"gene:{gene}"].append(
                            (idx, {claim: "present"}, {}, {}, {}, 1.0)
                        )
                        if ko_claim in configured_interventions:
                            rescue_i = {ko_claim: "present", claim: "present"}
                            self.yield_buckets[f"gene:{gene}:rescue"].append(
                                (idx, rescue_i, {}, {}, {}, 0.6)
                            )
                            self.yield_buckets[f"gene:{gene}:oe_vs_rescue"].append(
                                (idx, {claim: "present"}, {}, rescue_i, {}, 0.4)
                            )
                # Cross-locus orthogonality: non-matching Class-I loci stay invariant
                for other_locus in sorted(HLA_LOCUS_GENES - {gene}):
                    for idx in by_hla_locus.get(other_locus, [])[:128]:
                        self.yield_buckets[f"hla_orth:{gene}_on_{other_locus}"].append(
                            (idx, {claim: "present"}, {}, {}, {}, 0.0)
                        )
                continue

            allowed_classes = factor_classes(gene)
            for cls in allowed_classes:
                if cls not in config.mhc_classes:
                    continue
                claim_rows = by_present_claim.get((cls, f"intervention:{claim}"), [])
                ref_rows = by_class_unperturbed.get(cls, [])
                candidates = (claim_rows[:128] + ref_rows[:128])[:256]
                for idx in candidates:
                    self.yield_buckets[f"gene:{gene}"].append(
                        (idx, {claim: "present"}, {}, {}, {}, sign * severity)
                    )

        # Complete adjacent KO < KD < WT < OE severity ladder inside each gene's yield bucket
        if getattr(config, "elution_shared_processing", False):
            for gene in supported_perturbation_genes(config.interventions):
                if gene in HLA_LOCUS_GENES:
                    pool = by_hla_locus.get(gene, [])[:128]
                    for idx in pool:
                        self.yield_buckets[f"gene:{gene}"].append(
                            (
                                idx,
                                {f"knockout:{gene}": "present"},
                                {},
                                {f"knockdown:{gene}": "present"},
                                {},
                                -0.5,
                            )
                        )
                        self.yield_buckets[f"gene:{gene}"].append(
                            (idx, {f"knockdown:{gene}": "present"}, {}, {}, {}, -0.5)
                        )
                        self.yield_buckets[f"gene:{gene}"].append(
                            (idx, {f"overexpression:{gene}": "present"}, {}, {}, {}, 0.8)
                        )
                    continue
                for cls in factor_classes(gene):
                    if cls not in config.mhc_classes:
                        continue
                    for idx in by_class_unperturbed.get(cls, [])[:128]:
                        self.yield_buckets[f"gene:{gene}"].append(
                            (
                                idx,
                                {f"knockout:{gene}": "present"},
                                {},
                                {f"knockdown:{gene}": "present"},
                                {},
                                -0.5,
                            )
                        )
                        self.yield_buckets[f"gene:{gene}"].append(
                            (idx, {f"knockdown:{gene}": "present"}, {}, {}, {}, -0.5)
                        )
                        self.yield_buckets[f"gene:{gene}"].append(
                            (idx, {f"overexpression:{gene}": "present"}, {}, {}, {}, 0.8)
                        )

        # Combined endogenous HLA-A+C KO orthogonality on HLA-B rows
        if {"knockout:HLA-A", "knockout:HLA-C"} <= configured_interventions:
            for idx in by_hla_locus.get("HLA-B", [])[:128]:
                self.yield_buckets["hla_orth:AC_on_HLA-B"].append(
                    (
                        idx,
                        {"knockout:HLA-A": "present", "knockout:HLA-C": "present"},
                        {},
                        {},
                        {},
                        0.0,
                    )
                )

        for cytokine in config.cytokines:
            for cls in config.mhc_classes:
                cyto_rows = by_present_claim.get((cls, f"cytokine:{cytokine}"), [])
                ref_rows = by_class_unperturbed.get(cls, [])
                candidates = (cyto_rows[:128] + ref_rows[:128])[:256]
                for idx in candidates:
                    self.yield_buckets[f"cytokine:{cytokine}"].append(
                        (idx, {}, {cytokine: "present"}, {}, {}, 0.65)
                    )
                    self.yield_buckets[f"cytokine:{cytokine}"].append(
                        (idx, {}, {}, {}, {cytokine: "absent"}, 0.35)
                    )

        self.selectivity_keys = tuple(sorted(k for k, v in self.selectivity_buckets.items() if v))
        self.yield_keys = tuple(sorted(k for k, v in self.yield_buckets.items() if v))
        self.quant_keys = tuple(sorted(k for k, v in self.quant_buckets.items() if v))
        self.counts = {
            "selectivity_factors": len(self.selectivity_keys),
            "selectivity_pairs": sum(
                len(self.selectivity_buckets[k]) for k in self.selectivity_keys
            ),
            "yield_factors": len(self.yield_keys),
            "yield_items": sum(len(self.yield_buckets[k]) for k in self.yield_keys),
            "quant_plexes": len(self.quant_keys),
            "quant_items": sum(len(self.quant_buckets[k]) for k in self.quant_keys),
        }

    def draw(
        self,
        index,
        rng,
        selectivity_per_step: int,
        yield_per_step: int,
        quant_per_step: int = 0,
    ):
        selectivity_draws = []
        if self.selectivity_keys and selectivity_per_step > 0:
            chosen_keys = [rng.choice(self.selectivity_keys) for _ in range(selectivity_per_step)]
            for key in chosen_keys:
                spec = rng.choice(self.selectivity_buckets[key])
                kind = spec[0]
                if kind == "arm_swap":
                    _, ia, ib, ia_d, ca_d, ib_d, cb_d = spec
                    row_a, row_b = index[ia], index[ib]
                    if row_a.split != "train" or row_b.split != "train":
                        raise ValueError("APC contrast draw must be TRAIN only")
                    selectivity_draws.append(
                        (
                            _apply_apc_condition(row_a, ia_d, ca_d),
                            _apply_apc_condition(row_a, ib_d, cb_d),
                            _apply_apc_condition(row_b, ia_d, ca_d),
                            _apply_apc_condition(row_b, ib_d, cb_d),
                        )
                    )
                elif kind == "pair_vs_unknown":
                    _, pos_idx, bg_idx = spec
                    pos_row, bg_row = index[pos_idx], index[bg_idx]
                    if pos_row.split != "train" or bg_row.split != "train":
                        raise ValueError("APC contrast draw must be TRAIN only")
                    selectivity_draws.append(
                        (
                            pos_row,
                            _strip_all_apc(pos_row),
                            bg_row,
                            _strip_all_apc(bg_row),
                        )
                    )
                elif kind == "pair_vs_sibling":
                    _, pos_idx, bg_idx, donor_idx = spec
                    pos_row, bg_row, donor_row = index[pos_idx], index[bg_idx], index[donor_idx]
                    if (
                        pos_row.split != "train"
                        or bg_row.split != "train"
                        or donor_row.split != "train"
                    ):
                        raise ValueError("APC contrast draw must be TRAIN only")
                    selectivity_draws.append(
                        (
                            pos_row,
                            _copy_full_apc(pos_row, donor_row.context),
                            bg_row,
                            _copy_full_apc(bg_row, donor_row.context),
                        )
                    )

        yield_draws = []
        if self.yield_keys and yield_per_step > 0:
            chosen_ykeys = [rng.choice(self.yield_keys) for _ in range(yield_per_step)]
            for key in chosen_ykeys:
                idx, pi, pc, ri, rc, signed_dir = rng.choice(self.yield_buckets[key])
                row = index[idx]
                if row.split != "train":
                    raise ValueError("APC yield draw must be TRAIN only")
                yield_draws.append(
                    (
                        _apply_apc_condition(row, pi, pc),
                        _apply_apc_condition(row, ri, rc),
                        signed_dir,
                    )
                )

        quant_draws = []
        if self.quant_keys and quant_per_step > 0:
            # Sample >= 2 rows per chosen plex so within-plex ranking contrasts are active
            plex_batch = max(1, quant_per_step // 2)
            per_plex = max(2, quant_per_step // plex_batch)
            for _ in range(plex_batch):
                if len(quant_draws) >= quant_per_step:
                    break
                pkey = rng.choice(self.quant_keys)
                bucket = self.quant_buckets[pkey]
                take = min(per_plex, quant_per_step - len(quant_draws))
                for _ in range(take):
                    idx, fc, ctrl_i, ctrl_c = rng.choice(bucket)
                    row = index[idx]
                    if row.split != "train":
                        raise ValueError("APC quantitative draw must be TRAIN only")
                    quant_draws.append(
                        (
                            row,
                            _apply_apc_condition(row, ctrl_i, ctrl_c),
                            fc,
                            pkey,
                        )
                    )
        return selectivity_draws, yield_draws, quant_draws


def _shared_directional_shift(output) -> torch.Tensor | None:
    """Extract mean `apc_yield_shift + hla_locus_shift` when shared processing is active."""
    candidates = output.diagnostics.get("candidates", ())
    if not candidates:
        return None
    vals = []
    for cand in candidates:
        terms = []
        for reg in cand.get("registers", ()):
            sp = reg.get("elution_terms", {}).get("shared_processing")
            if sp is None:
                return None
            terms.append(reg["class_probability"] * (sp["apc_yield_shift"] + sp["hla_locus_shift"]))
        if not terms:
            return None
        vals.append(torch.stack(terms).sum())
    return torch.stack(vals).mean() if vals else None


def apc_contrast_loss(
    model,
    selectivity_draws,
    yield_draws,
    *,
    quant_draws=(),
    yield_weight: float = 0.5,
    margin: float = 0.22,
    quant_weight: float = 0.5,
    quant_rank_weight: float = 0.5,
    quant_scale: float = 0.40,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Compute cross-condition selectivity, hinged yield, and quantitative fold-change loss."""
    flat_rows = []
    for quad in selectivity_draws:
        flat_rows.extend(quad)
    sel_split = len(flat_rows)
    for pert_row, ref_row, _ in yield_draws:
        flat_rows.append(pert_row)
        flat_rows.append(ref_row)
    yld_split = len(flat_rows)
    for trt_row, ctrl_row, _, _ in quant_draws:
        flat_rows.append(trt_row)
        flat_rows.append(ctrl_row)

    if not flat_rows:
        return None, {}

    outputs = model(flat_rows)
    parts: dict[str, float] = {}
    total: torch.Tensor | None = None

    if selectivity_draws:
        sel_losses = []
        for offset in range(0, sel_split, 4):
            a_ca, a_cb, b_ca, b_cb = (outputs[offset + k].value for k in range(4))
            double_diff = (a_ca - a_cb) - (b_ca - b_cb)
            sel_losses.append(F.softplus(0.20 - double_diff, beta=4.0))
        sel_term = torch.stack(sel_losses).mean()
        parts["apc_selectivity_loss"] = float(sel_term.detach())
        total = sel_term

    if yield_draws and yield_weight > 0:
        yield_losses = []
        cursor = sel_split
        for _, _, signed_dir in yield_draws:
            out_pert = outputs[cursor]
            out_ref = outputs[cursor + 1]
            cursor += 2
            pert_val = out_pert.value
            ref_val = out_ref.value
            dir_pert = _shared_directional_shift(out_pert)
            dir_ref = _shared_directional_shift(out_ref)
            if signed_dir == 0.0:
                # Cross-locus HLA orthogonality: zero shift on non-matching Class-I loci
                orth = 5.0 * (pert_val - ref_val).square()
                if dir_pert is not None and dir_ref is not None:
                    orth = orth + 5.0 * (dir_pert - dir_ref).square()
                yield_losses.append(orth)
            else:
                direction = 1.0 if signed_dir > 0 else -1.0
                target_margin = margin * abs(signed_dir)
                gap_total = direction * (pert_val - ref_val)
                item_loss = F.relu(target_margin - gap_total) + 0.10 * F.relu(
                    gap_total - 4.5 * target_margin
                )
                if dir_pert is not None and dir_ref is not None:
                    gap_dir = direction * (dir_pert - dir_ref)
                    dir_loss = F.relu(target_margin - gap_dir) + 0.15 * F.relu(
                        gap_dir - 3.5 * target_margin
                    )
                    item_loss = 0.5 * item_loss + 0.5 * dir_loss
                yield_losses.append(item_loss)
        yield_term = torch.stack(yield_losses).mean()
        parts["apc_yield_loss"] = float(yield_term.detach())
        total = yield_weight * yield_term if total is None else total + yield_weight * yield_term

    if quant_draws and quant_weight > 0:
        q_reg_losses = []
        by_plex: dict[str, list[tuple[torch.Tensor, float]]] = defaultdict(list)
        cursor = yld_split
        for _, _, fc, plex_id in quant_draws:
            delta_logit = outputs[cursor].value - outputs[cursor + 1].value
            cursor += 2
            clamped_fc = max(-4.0, min(4.0, float(fc)))
            target_t = delta_logit.new_tensor(quant_scale * clamped_fc)
            q_reg_losses.append(F.smooth_l1_loss(delta_logit, target_t, beta=0.35))
            by_plex[plex_id].append((delta_logit, float(fc)))
        q_term = torch.stack(q_reg_losses).mean()
        rank_losses = []
        if quant_rank_weight > 0:
            for items in by_plex.values():
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        d_i, fc_i = items[i]
                        d_j, fc_j = items[j]
                        if abs(fc_i - fc_j) < 0.5:
                            continue
                        sgn = 1.0 if fc_i > fc_j else -1.0
                        rank_losses.append(F.softplus(0.15 - sgn * (d_i - d_j), beta=4.0))
        if rank_losses:
            q_rank = torch.stack(rank_losses).mean()
            parts["apc_quant_rank_loss"] = float(q_rank.detach())
            q_term = q_term + quant_rank_weight * q_rank
        parts["apc_quant_loss"] = float(q_term.detach())
        total = quant_weight * q_term if total is None else total + quant_weight * q_term

    return total, parts
