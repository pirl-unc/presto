"""Unit tests for Presto-Omni v2.9 upgrades:
1. Within-peptide cross-allele comparative binding loss (`AlleleDiffPool` + `allele_diff_loss`),
2. Class I 15+mer overhang / nested 9-mer sub-registers (`class1_overhang_registers`),
3. Class-specific `TCellStagedHead v2` (`tcell_class_specific_tcr`, Class II stability mask),
4. Class-conditioned quantitative binding assay offsets (`class_conditioned_assay_offset`),
5. Warm-start migration from `v2.8` into `v2.9` with zero step-0 regression on 8-14mer Class I.
"""

from __future__ import annotations

import random

import torch

from presto.data.assay_measurement import AssayTrack
from presto.data.modular import Context, MHCInput, Observation, TargetScale
from presto.data.tcell_assay import TCellTrack
from presto.models.presto import Presto, PrestoConfig, candidates
from presto.training.allele_diff import AlleleDiffPool, allele_diff_loss
from presto.training.modular import initialize_from_checkpoint, save_checkpoint, train_step
from presto.training.modular_controls import validate_controls

_G1_I = "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEA"
_G2_I = (
    "GSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTD"
)
_G1_II = "IKEEHVIIQAEFYLNPDQSGEFMFDFDGDEIFHVDMAKKETVWRLEEFGRFASFEAQGALANIAVDKANLEIMTKRSNYTPITN"
_G2_DR0401 = (
    "GDTRPRFLEQVKHECHFFNGTERVRFLDRYFYHQEEYVRFDSDVGEYRAVTELGRPDAEYWNSQKDLLEQKRAAVDTYCRHNYGVGESFTVQRR"
)


def _make_config(
    *,
    class1_overhang_registers: bool = False,
    tcell_class_specific_tcr: bool = False,
    class_conditioned_assay_offset: bool = False,
) -> PrestoConfig:
    assay_tracks = (
        AssayTrack("ic50", "purified", "competitive", "radioactivity"),
        AssayTrack("kd_via_ic50", "purified", "competitive", "radioactivity"),
        AssayTrack("kd_via_ec50", "purified", "direct", "fluorescence"),
    )
    tcell_tracks = (
        TCellTrack("soluble_factor_release", "ifng_release", "elispot"),
        TCellTrack("pmhc_multimer", "qualitative_binding", "multimer_tetramer"),
    )
    return PrestoConfig(
        endpoints=(
            "ic50",
            "kd_via_ic50",
            "kd_via_ec50",
            "half_life",
            "elution_observation",
            "tcell_response",
        ),
        mhc_classes=("I", "II"),
        assay_tracks=assay_tracks,
        tcell_tracks=tcell_tracks,
        tcell_staged=True,
        tcell_ordinal_loss_weight=0.35,
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        elution_staging="affinity_downstream",
        elution_affinity_endpoint="ic50",
        elution_pmhc_residual=True,
        elution_mhc_abundance=True,
        interaction_mode="anchor_v3",
        groove_segment_embedding=True,
        imgt_groove_embedding=True,
        imgt_pocket_cross=True,
        class1_overhang_registers=class1_overhang_registers,
        tcell_class_specific_tcr=tcell_class_specific_tcr,
        class_conditioned_assay_offset=class_conditioned_assay_offset,
    )


def _mhc(cls: str, g1: str, g2: str) -> MHCInput:
    return MHCInput(groove1=g1, groove2=g2, mhc_class=cls)


def test_class1_overhang_registers_boundary_and_forward():
    # 8-14mers are strictly unchanged
    for length in (8, 9, 10, 11, 14):
        pep = "ACDEFGHIKLMNPQ"[:length]
        c_off = candidates(pep, "I", class1_overhang=False)
        c_on = candidates(pep, "I", class1_overhang=True)
        assert c_on == c_off
        assert len(c_on) == 1 and c_on[0][1] == pep

    # 15+mers scan sliding 9-mers with exact left/right flank lengths
    pep15 = "GILGFVFTLAAAAAA"
    c15 = candidates(pep15, "I", class1_overhang=True)
    assert len(c15) == 7
    assert all(len(core) == 9 for _, core, _ in c15)
    assert c15[0][1] == "GILGFVFTL"
    assert c15[0][0] == 0

    cfg = _make_config(class1_overhang_registers=True)
    model = Presto(cfg).eval()
    m1 = _mhc("I", _G1_I, _G2_I)
    row = Observation(
        observation_id="o15",
        endpoint="elution_observation",
        source="test",
        peptide=pep15,
        mhc=(m1,),
        context=Context(),
        split="predict",
    )
    out = model([row])[0]
    assert torch.isfinite(out.value)
    assert len(out.diagnostics["candidates"][0]["registers"][0]["starts"]) == 7


def test_allele_diff_pool_and_drb04_separation():
    # Synthetic DRB1*04:01 vs DRB1*04:05 differing at ONLY 2 residues (beta57, beta71)
    g2_0405 = list(_G2_DR0401)
    g2_0405[54] = "S"  # beta57 D->S
    g2_0405[66] = "R"  # beta71 K->R
    m_0401 = _mhc("II", _G1_II, _G2_DR0401)
    m_0405 = _mhc("II", _G1_II, "".join(g2_0405))

    rows = [
        Observation(
            observation_id="d1",
            endpoint="ic50",
            source="test",
            peptide="PKYVKQNTLKLAT",
            mhc=(m_0401,),
            value=10.0,
            unit="nM",
            qualifier="=",
            split="train",
        ),
        Observation(
            observation_id="d2",
            endpoint="ic50",
            source="test",
            peptide="PKYVKQNTLKLAT",
            mhc=(m_0405,),
            value=1500.0,
            unit="nM",
            qualifier="=",
            split="train",
        ),
        # Censored row must be ignored by AlleleDiffPool
        Observation(
            observation_id="d3",
            endpoint="ic50",
            source="test",
            peptide="PKYVKQNTLKLAT",
            mhc=(_mhc("II", _G1_II, "A" * len(_G2_DR0401)),),
            value=50000.0,
            unit="nM",
            qualifier=">",
            split="train",
        ),
    ]
    pool = AlleleDiffPool.from_observations(
        rows, min_diff_log10=0.15, close_identity_threshold=0.78
    )
    assert pool.counts["hard_close_II"] == 1
    assert pool.counts["general_II"] == 0

    cfg = _make_config(class_conditioned_assay_offset=True)
    model = Presto(cfg)
    scales = {"ic50": TargetScale(mean=2.5, scale=1.0, exact_count=100)}
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    pairs = pool.sample_pairs(random.Random(42), pairs_per_class=4)
    assert len(pairs) == 1

    pair_q = [("PKYVKQNTLKLAT", m_0401, "II"), ("PKYVKQNTLKLAT", m_0405, "II")]
    with torch.no_grad():
        s0 = model.score_binding_pairs(pair_q)
        diff0 = float((s0[0] - s0[1]).item())

    for _ in range(12):
        opt.zero_grad()
        loss, _ = allele_diff_loss(model, pairs, scales, beta=0.5)
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        opt.step()

    with torch.no_grad():
        s1 = model.score_binding_pairs(pair_q)
        diff1 = float((s1[0] - s1[1]).item())
    # Target diff is (1.0 - 3.2) = -2.2
    assert diff1 < diff0 - 0.15, f"expected DRB1*04 separation: {diff0:.3f} -> {diff1:.3f}"
    assert model.imgt_pocket_cross.direct_mlp[-1].weight.abs().max().item() > 0.0


def test_tcell_class_specific_tcr_masks_class2_stability_and_warm_starts(tmp_path):
    torch.manual_seed(7)
    cfg_v28 = _make_config(
        class1_overhang_registers=False,
        tcell_class_specific_tcr=False,
        class_conditioned_assay_offset=False,
    )
    m_v28 = Presto(cfg_v28).eval()
    ckpt_path = tmp_path / "v28_parent.pt"
    scales = {
        "ic50": TargetScale(mean=2.0, scale=1.0, exact_count=100),
        "kd_via_ic50": TargetScale(mean=2.0, scale=1.0, exact_count=100),
        "kd_via_ec50": TargetScale(mean=2.0, scale=1.0, exact_count=100),
        "half_life": TargetScale(mean=2.0, scale=1.0, exact_count=100),
    }
    save_checkpoint(ckpt_path, m_v28, scales)

    cfg_v29 = _make_config(
        class1_overhang_registers=True,
        tcell_class_specific_tcr=True,
        class_conditioned_assay_offset=True,
    )
    m_v29 = Presto(cfg_v29).eval()

    initialize_from_checkpoint(m_v29, ckpt_path, allow_interaction_expansion=True)

    # Exact 0.0 step-0 perturbation on canonical 9-mer Class I affinity & elution
    m1 = _mhc("I", _G1_I, _G2_I)
    r_ic50 = Observation(
        observation_id="c1",
        source="test",
        endpoint="ic50",
        peptide="GILGFVFTL",
        mhc=(m1,),
        split="predict",
    )
    r_elut = Observation(
        observation_id="e1",
        source="test",
        endpoint="elution_observation",
        peptide="GILGFVFTL",
        mhc=(m1,),
        context=Context(),
        split="predict",
    )
    with torch.no_grad():
        assert abs(m_v28([r_ic50])[0].value.item() - m_v29([r_ic50])[0].value.item()) < 1e-6
        assert abs(m_v28([r_elut])[0].value.item() - m_v29([r_elut])[0].value.item()) < 1e-6

    # Class II stability supply must have EXACT 0.0 influence when tcell_class_specific_tcr=True
    head = m_v29.tcell_staged_head
    assert "II" not in head.stab_weight_raw
    assert "I" in head.stab_weight_raw
    h = torch.randn(3, 32)
    bm = torch.randn(3, 32)
    bx = torch.randn(3, 32)
    ph = torch.randn(3, 8)
    aff = torch.randn(3)
    pres = torch.randn(3)
    _, rec_a, _, sup_a = head.score_registers(
        h,
        bm,
        bx,
        ph,
        mhc_class="II",
        detached_affinity=aff,
        detached_stability=torch.zeros(3),
        detached_presentation=pres,
        context=None,
    )
    _, rec_b, _, sup_b = head.score_registers(
        h,
        bm,
        bx,
        ph,
        mhc_class="II",
        detached_affinity=aff,
        detached_stability=torch.full((3,), 10.0),
        detached_presentation=pres,
        context=None,
    )
    assert torch.allclose(rec_a, rec_b, atol=1e-7)
    assert torch.allclose(sup_a, sup_b, atol=1e-7)


def test_class_conditioned_assay_offset_and_controls():
    cfg = _make_config(class_conditioned_assay_offset=True)
    model = Presto(cfg)
    expected_keys = {"ic50__II", "kd_via_ic50__II", "kd_via_ec50__II"}
    assert set(model.assay_class_offset.keys()) == expected_keys

    validate_controls(
        {
            "allele_diff_training": {
                "weight": 0.6,
                "pairs_per_class": 12,
                "hard_close_fraction": 0.65,
                "beta": 0.5,
                "min_diff_log10": 0.15,
                "close_identity_threshold": 0.78,
            }
        },
        cfg,
        balance_by=(),
    )

    g2b = _G2_DR0401[:54] + "S" + _G2_DR0401[55:66] + "R" + _G2_DR0401[67:]
    ma, mb = _mhc("II", _G1_II, _G2_DR0401), _mhc("II", _G1_II, g2b)
    rows = [
        Observation(
            observation_id="r1",
            source="test",
            endpoint="ic50",
            peptide="PKYVKQNTLKLAT",
            mhc=(ma,),
            value=15.0,
            unit="nM",
            qualifier="=",
            split="train",
        ),
        Observation(
            observation_id="r2",
            source="test",
            endpoint="kd_via_ic50",
            peptide="PKYVKQNTLKLAT",
            mhc=(mb,),
            value=1000.0,
            unit="nM",
            qualifier="=",
            split="train",
        ),
    ]
    pool = AlleleDiffPool.from_observations(
        [
            rows[0],
            Observation(
                observation_id="r3",
                source="test",
                endpoint="ic50",
                peptide="PKYVKQNTLKLAT",
                mhc=(mb,),
                value=1500.0,
                unit="nM",
                qualifier="=",
                split="train",
            ),
        ]
    )
    scales = {
        "ic50": TargetScale(mean=2.0, scale=1.0, exact_count=100),
        "kd_via_ic50": TargetScale(mean=2.0, scale=1.0, exact_count=100),
        "kd_via_ec50": TargetScale(mean=2.0, scale=1.0, exact_count=100),
        "half_life": TargetScale(mean=2.0, scale=1.0, exact_count=100),
    }
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    res = train_step(
        model,
        opt,
        rows,
        scales,
        allele_diff_pairs=pool.sample_pairs(random.Random(0), pairs_per_class=4),
        allele_diff_weight=0.6,
    )
    assert "allele_diff_loss" in res["loss_parts"]
    assert float(model.assay_class_offset["ic50__II"].item()) != 0.0
    assert float(model.assay_class_offset["kd_via_ic50__II"].item()) != 0.0
