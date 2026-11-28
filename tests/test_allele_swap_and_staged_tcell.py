"""Unit and integration tests for AlleleSwapPool and PRIME-style staged T-cell head."""

from __future__ import annotations

import random

import pytest
import torch

from presto.data.assay_measurement import AssayTrack
from presto.data.modular import Context, MHCInput, Observation, TargetScale
from presto.data.tcell_assay import TCellTrack
from presto.models.presto import Presto, PrestoConfig
from presto.models.tcell_assay import tcr_bulge_physicochem, tcr_facing_indices
from presto.training.allele_swap import AlleleSwapPool, allele_swap_loss
from presto.training.modular import initialize_from_checkpoint, save_checkpoint, train_step
from presto.training.modular_controls import tcell_ranking_loss

_G1_A = "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEA"
_G2_A = (
    "GSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQR"
)
_G1_B = "GSHSMRYFYTSVSRPGRGEPRFISVGYVDDTQFVRFDSDAASPREEPRAPWIEQEGPEYWDRNTQIYKAQAQTDRESLRNLRGYYNQSEA"
_G2_B = (
    "GSHTLQSMYGCDVGPDGRLLRGHDQYAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGECVEWLRRYLENGKDKLER"
)
_G1_DR1 = "IKEEHVIIQAEFYLNPDQSGEFMFDFDGDEIFHVDMAKKETVWRLEEFGRFASFEAQGALANIAVDKANLEIMTKRSNYTPIT"
_G2_DR1 = (
    "GDTRPRFLWQLKFECHFFNGTERVRLLERCIYNQEESVRFDSDVGEYRAVTELGRPDAEYWNSQKDLLEQRRAAVDTYCRHNYGVGESFTVQR"
)
_G1_DR4 = "IKEEHVIIQAEFYLNPDQSGEFMFDFDGDEIFHVDMAKKETVWRLEEFGRFASFEAQGALANIAVDKANLEIMTKRSNYTPIT"
_G2_DR4 = (
    "GDTRPRFLEQVKHECHFFNGTERVRFLDRYFYHQEEYVRFDSDVGEYRAVTELGRPDAEYWNSQKDLLEQKRAAVDTYCRHNYGVGESFTVQR"
)


def _make_config(*, tcell_staged: bool = False) -> PrestoConfig:
    t1 = TCellTrack(family="soluble_factor_release", response="ifng_release", method="elispot")
    t2 = TCellTrack(family="pmhc_multimer", response="multimer_binding", method="multimer")
    a1 = AssayTrack(
        endpoint="ic50",
        preparation="purified",
        mechanism="competitive",
        detector="fluorescence",
    )
    return PrestoConfig(
        endpoints=("ic50", "half_life", "elution_observation", "tcell_response"),
        assay_tracks=(a1,),
        tcell_tracks=(t1, t2),
        tcell_staged=tcell_staged,
        mhc_classes=("I", "II"),
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        interaction_mode="anchor_v3",
        elution_staging="affinity_downstream",
        elution_affinity_endpoint="ic50",
        elution_detach_affinity=False,
        elution_pmhc_residual=True,
        elution_shared_processing=True,
        source_context=True,
        state_bottleneck=16,
        state_mode="harmonized",
        components=("PSMB8", "TAP1"),
    )


def _elut_row(
    oid: str, pep: str, g1: str, g2: str, cls: str, val: float = 1.0, split: str = "train"
):
    return Observation(
        observation_id=oid,
        endpoint="elution_observation",
        source="test_ms",
        peptide=pep,
        mhc=(MHCInput(groove1=g1, groove2=g2, mhc_class=cls),),
        value=val,
        unit="binary",
        context=Context(source_n="AAAA", source_c="KKKK"),
        split=split,
    )


def _tcell_row(
    oid: str, pep: str, g1: str, g2: str, cls: str, track: TCellTrack, val: float, study: str = "s1"
):
    return Observation(
        observation_id=oid,
        endpoint="tcell_response",
        source="test_tcell",
        peptide=pep,
        mhc=(MHCInput(groove1=g1, groove2=g2, mhc_class=cls),),
        value=val,
        unit="binary",
        tcell_track=track,
        provenance={"study_id": study},
        split="train",
    )


def test_allele_swap_pool_disjoint_and_zero_processing_grad():
    cfg = _make_config(tcell_staged=False)
    model = Presto(cfg)
    pool = AlleleSwapPool(max_buffer_per_class=64)
    rows = [
        _elut_row("e1", "NLVPMVATV", _G1_A, _G2_A, "I", 1.0),
        _elut_row("e2", "RPHERNGFTV", _G1_B, _G2_B, "I", 1.0),
        _elut_row("e3", "AAAAAAAAA", _G1_A, _G2_A, "I", 0.0),  # negative ignored
        _elut_row("e4", "GILGFVFTL", _G1_A, _G2_A, "I", 1.0, split="validation"),  # val ignored
        _elut_row("e5", "PKYVKQNTLKLAT", _G1_DR1, _G2_DR1, "II", 1.0),
        _elut_row("e6", "AAYAAAKAAALAA", _G1_DR4, _G2_DR4, "II", 1.0),
    ]
    groups = pool.sample_groups(rows, random.Random(7), anchors_per_class=4, negatives_per_anchor=4)
    assert len(groups) == 4
    classes = {g.mhc_class for g in groups}
    assert classes == {"I", "II"}
    for g in groups:
        assert g.positive_peptide not in g.negative_peptides
        assert "AAAAAAAAA" not in g.negative_peptides
        assert "GILGFVFTL" not in g.negative_peptides

    loss, stats = allele_swap_loss(model, groups, temperature=0.5)
    assert loss is not None and torch.isfinite(loss)
    assert stats["allele_swap_groups_I"] == 2.0
    assert stats["allele_swap_groups_II"] == 2.0

    model.zero_grad()
    loss.backward()
    # Exact 0.0 gradient into processing & elution residual modules!
    for name, p in model.named_parameters():
        if name.startswith(("elution_shared.", "elution_pmhc_mlp.", "elution_processing.")):
            assert p.grad is None or float(p.grad.abs().max()) == 0.0, f"non-zero grad on {name}"
    # Non-zero gradient into binding assay_layer & encoder!
    assay_grad = sum(
        float(p.grad.abs().sum())
        for n, p in model.named_parameters()
        if n.startswith("assay_layer.") and p.grad is not None
    )
    assert assay_grad > 0.0


def test_tcr_bulge_physicochem_and_staged_tcell_invariants():
    assert tcr_facing_indices(9, "I") == (3, 4, 5, 6, 7)
    assert tcr_facing_indices(9, "II") == (1, 2, 4, 7)
    phys_arom = tcr_bulge_physicochem(
        [(0, "AAAWFWYWA", [0.0, 0.0, 2.3])], "I", device=torch.device("cpu"), dtype=torch.float32
    )
    phys_ala = tcr_bulge_physicochem(
        [(0, "AAAAAAAAA", [0.0, 0.0, 2.3])], "I", device=torch.device("cpu"), dtype=torch.float32
    )
    assert float(phys_arom[0, 0]) == pytest.approx(1.0)
    assert float(phys_ala[0, 0]) == pytest.approx(0.0)

    cfg_legacy = _make_config(tcell_staged=False)
    assert "tcell_staged" not in cfg_legacy.to_dict()
    cfg_staged = _make_config(tcell_staged=True)
    assert cfg_staged.to_dict()["tcell_staged"] is True

    model = Presto(cfg_staged)
    model.eval()
    t1, t2 = cfg_staged.tcell_tracks
    r1 = _tcell_row("tc1", "NLVPMVATV", _G1_A, _G2_A, "I", t1, 1.0)
    r2 = _tcell_row("tc2", "NLVPMVATV", _G1_A, _G2_A, "I", t2, 1.0)
    with torch.no_grad():
        out1, out2 = model([r1, r2])
    # Track-free recognition & bulge propensity are EXACTLY invariant to track selection,
    # and step-0 TCR bulge residual is exact zero (starts cleanly at binding+stability floor)!
    d_rec = (out1.components["tcell_recognition"] - out2.components["tcell_recognition"]).abs()
    d_bul = (
        out1.components["tcr_bulge_propensity"] - out2.components["tcr_bulge_propensity"]
    ).abs()
    assert float(d_rec) == 0.0
    assert float(d_bul) == 0.0
    assert float(out1.components["tcr_bulge_propensity"].abs()) == 0.0
    assert (
        float((out1.components["tcell_recognition"] - out1.components["tcell_pmhc_supply"]).abs())
        == 0.0
    )
    assert float((out1.reference_value - out2.reference_value).abs()) == 0.0

    # Verify exact 0.0 gradient from staged tcell_response into detached ic50 & half_life!
    model.train()
    batch = [
        _tcell_row("tc_p1", "NLVPMVATV", _G1_A, _G2_A, "I", t1, 1.0, study="st1"),
        _tcell_row("tc_n1", "AAAAAAAAA", _G1_A, _G2_A, "I", t1, 0.0, study="st1"),
        _tcell_row("tc_p2", "PKYVKQNTLKLAT", _G1_DR1, _G2_DR1, "II", t2, 1.0, study="st2"),
        _tcell_row("tc_n2", "AAYAAAKAAALAA", _G1_DR1, _G2_DR1, "II", t2, 0.0, study="st2"),
    ]
    outs = model(batch)
    rank_loss, n_pairs = tcell_ranking_loss(batch, outs)
    assert rank_loss is not None and n_pairs >= 2
    bce = sum(
        torch.nn.functional.binary_cross_entropy_with_logits(o.value, torch.tensor(r.value))
        for r, o in zip(batch, outs)
    )
    tot = bce + 0.5 * rank_loss
    model.zero_grad()
    tot.backward()
    for name, p in model.named_parameters():
        if name.startswith(("assay_layer.", "heads.half_life.", "elution_pmhc_mlp.")):
            assert p.grad is None or float(p.grad.abs().max()) == 0.0, f"leaked grad: {name}"
    tcr_grad = sum(
        float(p.grad.abs().sum())
        for n, p in model.named_parameters()
        if n.startswith("tcell_staged_head.") and p.grad is not None
    )
    assert tcr_grad > 0.0


def test_warm_start_upgrade_to_tcell_staged(tmp_path):
    cfg_old = _make_config(tcell_staged=False)
    old_model = Presto(cfg_old)
    scales = {
        "ic50": TargetScale(mean=2.0, scale=1.0, exact_count=10),
        "half_life": TargetScale(mean=1.5, scale=0.5, exact_count=10),
    }
    ckpt = tmp_path / "legacy.pt"
    save_checkpoint(ckpt, old_model, scales)

    cfg_new = _make_config(tcell_staged=True)
    new_model = Presto(cfg_new)
    with pytest.raises(ValueError, match="allow_tcell_staging_expansion"):
        initialize_from_checkpoint(new_model, ckpt, allow_tcell_staging_expansion=False)
    info = initialize_from_checkpoint(new_model, ckpt, allow_tcell_staging_expansion=True)
    assert "tcell_assay_layer.intensity_decoder.0.weight" in info["inherited_parameters"]
    assert not any(
        n.startswith("tcell_assay_layer.positivity_decoder.")
        for n in dict(new_model.named_parameters())
    )

    # End-to-end train_step combining allele_swap_groups + tcell_ranking_weight!
    opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    t1, t2 = cfg_new.tcell_tracks
    rows = [
        _elut_row("e1", "NLVPMVATV", _G1_A, _G2_A, "I", 1.0),
        _elut_row("e2", "RPHERNGFTV", _G1_B, _G2_B, "I", 1.0),
        _elut_row("e5", "PKYVKQNTLKLAT", _G1_DR1, _G2_DR1, "II", 1.0),
        _elut_row("e6", "AAYAAAKAAALAA", _G1_DR4, _G2_DR4, "II", 1.0),
        _tcell_row("tc_p1", "NLVPMVATV", _G1_A, _G2_A, "I", t1, 1.0, study="st1"),
        _tcell_row("tc_n1", "AAAAAAAAA", _G1_A, _G2_A, "I", t1, 0.0, study="st1"),
    ]
    pool = AlleleSwapPool()
    groups = pool.sample_groups(rows, random.Random(1), anchors_per_class=4, negatives_per_anchor=2)
    res = train_step(
        new_model,
        opt,
        rows,
        scales,
        allele_swap_groups=groups,
        allele_swap_weight=0.25,
        tcell_ranking_weight=0.35,
    )
    assert "allele_swap_loss" in res["loss_parts"]
    assert "tcell_ranking_loss" in res["loss_parts"]


def test_imgt_groove_embedding_and_pocket_cross_zero_perturbation_and_grad(tmp_path):
    from presto.models.imgt_groove import (
        IMGT_PAN_CLASS_NUM_SLOTS,
        IMGT_VOCAB_SIZE,
        imgt_groove_slot_indices,
        imgt_pan_class_aa_ids,
    )

    s1, s2 = imgt_groove_slot_indices(_G1_A, _G2_A, "I")
    assert len(s1) == len(_G1_A) and len(s2) == len(_G2_A)
    assert all(0 <= idx < IMGT_VOCAB_SIZE for idx in (*s1, *s2))

    # Real DRB1*04:01 vs DRB1*04:05 differ at beta57 (D->S) and beta71 (K->R):
    g2_dr0405 = (
        "GDTRPRFLEQVKHECHFFNGTERVRFLDRYFYHQEEYVRFDSDVGEY"
        "RAVTELGRPSAEYWNSQKDLLEQRRAAVDTYCRHNYGVGESFTVQR"
    )
    pan_0401 = imgt_pan_class_aa_ids(_G1_DR4, _G2_DR4, "II")
    pan_0405 = imgt_pan_class_aa_ids(_G1_DR4, g2_dr0405, "II")
    assert len(pan_0401) == IMGT_PAN_CLASS_NUM_SLOTS
    assert len(pan_0405) == IMGT_PAN_CLASS_NUM_SLOTS
    diffs = [i for i, (a, b) in enumerate(zip(pan_0401, pan_0405)) if a != b]
    assert len(diffs) == 2  # Exact unblurred 2-residue polymorphism (beta57 & beta71)

    cfg_old = _make_config(tcell_staged=True)
    assert "imgt_groove_embedding" not in cfg_old.to_dict()
    assert "imgt_pocket_cross" not in cfg_old.to_dict()
    old_model = Presto(cfg_old)
    scales = {
        "ic50": TargetScale(mean=2.0, scale=1.0, exact_count=10),
        "half_life": TargetScale(mean=1.5, scale=0.5, exact_count=10),
    }
    ckpt = tmp_path / "pre_imgt.pt"
    save_checkpoint(ckpt, old_model, scales)

    cfg_new = PrestoConfig.from_dict(
        {
            **cfg_old.to_dict(),
            "imgt_groove_embedding": True,
            "imgt_pocket_cross": True,
            "elution_mhc_abundance": True,
        }
    )
    new_model = Presto(cfg_new)
    initialize_from_checkpoint(new_model, ckpt, allow_interaction_expansion=True)
    old_model.eval()
    new_model.eval()
    row_i = _elut_row("e1", "NLVPMVATV", _G1_A, _G2_A, "I", 1.0)
    row_ii = _elut_row("e2", "PKYVKQNTLKLAT", _G1_DR4, _G2_DR4, "II", 1.0)
    with torch.no_grad():
        o1_i, o1_ii = old_model([row_i, row_ii])
        o2_i, o2_ii = new_model([row_i, row_ii])
    assert float((o1_i.value - o2_i.value).abs()) == 0.0
    assert float((o1_ii.value - o2_ii.value).abs()) == 0.0

    # Two steps wake up zero-init final projections (no scalar-gate saddle point!)
    # -> full gradient flow into 58-pocket direct MLP, 202-slot contact_bias, and MHC abundance!
    opt = torch.optim.AdamW(new_model.parameters(), lr=2e-3)
    batch = [
        row_i,
        row_ii,
        _elut_row("e3", "PKYVKQNTLKLAT", _G1_DR4, g2_dr0405, "II", 0.0),
    ]
    for _ in range(10):
        train_step(new_model, opt, batch, scales)
    new_model.eval()
    with torch.no_grad():
        o_0401, o_0405 = new_model(
            [row_ii, _elut_row("e3", "PKYVKQNTLKLAT", _G1_DR4, g2_dr0405, "II", 0.0)]
        )
    # Must cleanly separate DRB1*04:01 (pos) vs DRB1*04:05 (neg) in <=10 steps!
    assert float(o_0401.value - o_0405.value) > 0.15

    new_model.train()
    new_model.zero_grad()
    loss = sum(o.value for o in new_model(batch))
    loss.backward()
    assert float(new_model.imgt_slot_embedding.weight.grad.abs().sum()) > 0.0
    assert float(new_model.imgt_pocket_cross.contact_bias.grad.abs().sum()) > 0.0
    assert float(new_model.imgt_pocket_cross.groove_aa_emb.weight.grad.abs().sum()) > 0.0
    assert float(new_model.imgt_pocket_cross.direct_pocket_emb.weight.grad.abs().sum()) > 0.0
    assert float(new_model.imgt_pocket_cross.direct_mlp[-1].weight.grad.abs().sum()) > 0.0
    assert float(new_model.imgt_pocket_cross.out_proj.weight.grad.abs().sum()) > 0.0
    assert float(new_model.elution_mhc_abundance_mlp[-1].weight.grad.abs().sum()) > 0.0
    assert float(new_model.elution_mhc_abundance_mlp[1].weight.grad.abs().sum()) > 0.0
