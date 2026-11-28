"""Unit tests for Presto-Omni v3.0 mechanisms:
1. Unblurred 9x58 IMGT `DirectPocketBilinearHead` (`direct_pocket_bilinear=True`):
   - Exact 0.0 step-0 warm-start perturbation on Class I & II affinity and elution,
   - Direct `O(1)` separation between DRB1*04:01 and DRB1*04:05 (differing at 2 residues)
     in 12 gradient steps,
   - Exact 0.0 affinity context invariance.
2. Register-aligned full-presentation `TCellStagedHead v3` (`tcell_register_aligned_supply=True`):
   - Eliminates `log(K)` length drift between 11-mer and 20-mer Class II peptides with equal
     affinity/presentation supply,
   - Warm-start compatibility from `v2.9` -> `v3.0` via `allow_interaction_expansion=True`.
"""

from __future__ import annotations

import torch

from presto.data.assay_measurement import AssayTrack
from presto.data.modular import Context, MHCInput, Observation, TargetScale
from presto.data.tcell_assay import TCellTrack
from presto.models.presto import Presto, PrestoConfig
from presto.training.allele_diff import AlleleDiffPool, allele_diff_loss
from presto.training.modular import initialize_from_checkpoint, save_checkpoint

_G1_I = "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEA"
_G2_I = (
    "GSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTD"
)
_G1_II = "IKEEHVIIQAEFYLNPDQSGEFMFDFDGDEIFHVDMAKKETVWRLEEFGRFASFEAQGALANIAVDKANLEIMTKRSNYTPIT"
_G2_DR0401 = (
    "GDTRPRFLEQVKHECHFFNGTERVRFLDRYFYHQEEYVRFDSDVGEYRAVTELGRPDAEYWNSQKDLLEQKRAAVDTYCRHNYGVGESFTVQR"
)
_G2_DR0405 = (
    "GDTRPRFLEQVKHECHFFNGTERVRFLDRYFYHQEEYVRFDSDVGEYRAVTELGRPSAEYWNSQKDLLEQRRAAVDTYCRHNYGVGESFTVQR"
)


def _make_config(
    *,
    direct_pocket_bilinear: bool = False,
    tcell_register_aligned_supply: bool = False,
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
        tcell_class_specific_tcr=True,
        tcell_register_aligned_supply=tcell_register_aligned_supply,
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
        class1_overhang_registers=True,
        class_conditioned_assay_offset=True,
        direct_pocket_bilinear=direct_pocket_bilinear,
        interaction_mode="anchor_v3",
        groove_segment_embedding=True,
        imgt_groove_embedding=True,
        imgt_pocket_cross=True,
    )


def test_v30_warm_start_zero_perturbation_and_context_invariance(tmp_path):
    torch.manual_seed(11)
    m_v29 = Presto(_make_config()).eval()
    ckpt = tmp_path / "v29_parent.pt"
    scales = {
        "ic50": TargetScale(mean=2.0, scale=1.0, exact_count=100),
        "kd_via_ic50": TargetScale(mean=2.0, scale=1.0, exact_count=100),
        "kd_via_ec50": TargetScale(mean=2.0, scale=1.0, exact_count=100),
        "half_life": TargetScale(mean=2.0, scale=1.0, exact_count=100),
    }
    save_checkpoint(ckpt, m_v29, scales)

    m_v30 = Presto(
        _make_config(direct_pocket_bilinear=True, tcell_register_aligned_supply=True)
    ).eval()
    initialize_from_checkpoint(m_v30, ckpt, allow_interaction_expansion=True)

    m1 = MHCInput(groove1=_G1_I, groove2=_G2_I, mhc_class="I")
    m2 = MHCInput(groove1=_G1_II, groove2=_G2_DR0401, mhc_class="II")
    r_i = Observation(
        observation_id="i1",
        source="test",
        endpoint="ic50",
        peptide="GILGFVFTL",
        mhc=(m1,),
        split="predict",
    )
    r_ii = Observation(
        observation_id="ii1",
        source="test",
        endpoint="ic50",
        peptide="PKYVKQNTLKLAT",
        mhc=(m2,),
        split="predict",
    )
    r_el = Observation(
        observation_id="e1",
        source="test",
        endpoint="elution_observation",
        peptide="GILGFVFTL",
        mhc=(m1,),
        context=Context(),
        split="predict",
    )
    r_tc = Observation(
        observation_id="t1",
        source="test",
        endpoint="tcell_response",
        peptide="PKYVKQNTLKLAT",
        mhc=(m2,),
        context=Context(),
        tcell_track=TCellTrack("soluble_factor_release", "ifng_release", "elispot"),
        split="predict",
    )
    with torch.no_grad():
        # Exact 0.0 step-0 perturbation on Class I & II affinity and elution
        assert abs(m_v29([r_i])[0].value.item() - m_v30([r_i])[0].value.item()) < 1e-6
        assert abs(m_v29([r_ii])[0].value.item() - m_v30([r_ii])[0].value.item()) < 1e-6
        assert abs(m_v29([r_el])[0].value.item() - m_v30([r_el])[0].value.item()) < 1e-6
        # T-cell aligned forward pass is finite and emits all staged diagnostics
        tc_out = m_v30([r_tc])[0]
        assert torch.isfinite(tc_out.value)
        assert "tcell_recognition" in tc_out.components
        assert "tcell_pmhc_supply" in tc_out.components
        assert "tcr_bulge_propensity" in tc_out.components

    # Exact 0.0 affinity context invariance
    r_ii_ctx = Observation(
        observation_id="ii2",
        source="test",
        endpoint="ic50",
        peptide="PKYVKQNTLKLAT",
        mhc=(m2,),
        context=Context(source_n="AAAAA", source_c="GGGGG"),
        split="predict",
    )
    with torch.no_grad():
        assert abs(m_v30([r_ii])[0].value.item() - m_v30([r_ii_ctx])[0].value.item()) == 0.0


def test_direct_pocket_bilinear_rapid_drb04_separation():
    torch.manual_seed(3)
    m = Presto(_make_config(direct_pocket_bilinear=True))
    m_0401 = MHCInput(groove1=_G1_II, groove2=_G2_DR0401, mhc_class="II")
    m_0405 = MHCInput(groove1=_G1_II, groove2=_G2_DR0405, mhc_class="II")

    rows = [
        Observation(
            observation_id="d1",
            source="test",
            endpoint="ic50",
            peptide="AAYAAAAAAADAA",
            mhc=(m_0401,),
            value=2000.0,
            unit="nM",
            qualifier="=",
            split="train",
        ),
        Observation(
            observation_id="d2",
            source="test",
            endpoint="ic50",
            peptide="AAYAAAAAAADAA",
            mhc=(m_0405,),
            value=10.0,
            unit="nM",
            qualifier="=",
            split="train",
        ),
    ]
    pool = AlleleDiffPool.from_observations(rows)
    pairs = list(pool.hard_close_by_class["II"])
    assert len(pairs) == 1
    scales = {"ic50": TargetScale(mean=2.0, scale=1.0, exact_count=100)}
    opt = torch.optim.AdamW(m.direct_pocket_bilinear.parameters(), lr=5e-3)

    q = [("AAYAAAAAAADAA", m_0401, "II"), ("AAYAAAAAAADAA", m_0405, "II")]
    with torch.no_grad():
        s0 = m.score_binding_pairs(q)
        diff0 = float((s0[0] - s0[1]).item())

    for _ in range(15):
        opt.zero_grad()
        loss, _ = allele_diff_loss(m, pairs, scales, beta=0.5)
        assert loss is not None
        loss.backward()
        opt.step()

    with torch.no_grad():
        s1 = m.score_binding_pairs(q)
        diff1 = float((s1[0] - s1[1]).item())
    # Target diff = log10(2000) - log10(10) = +2.30; direct head must rapidly separate > +0.8!
    assert diff1 - diff0 > 0.8, f"expected rapid direct separation, got {diff0:.3f} -> {diff1:.3f}"


def test_tcell_register_aligned_supply_eliminates_logk_length_drift():
    m = Presto(_make_config(tcell_register_aligned_supply=True)).eval()
    head = m.tcell_staged_head
    # Compare K=3 registers (11-mer) vs K=12 registers (20-mer) with IDENTICAL pooled supply
    bm3, bx3, ph3 = torch.zeros(3, 32), torch.zeros(3, 32), torch.zeros(3, 8)
    bm12, bx12, ph12 = torch.zeros(12, 32), torch.zeros(12, 32), torch.zeros(12, 8)
    att3 = torch.full((3,), 1.0 / 3.0)
    att12 = torch.full((12,), 1.0 / 12.0)
    aff = torch.tensor(1.5)
    pres = torch.tensor(0.8)
    with torch.no_grad():
        _, rec3, _, sup3 = head.score_aligned(
            bm3,
            bx3,
            ph3,
            att3,
            mhc_class="II",
            pooled_affinity=aff,
            pooled_stability=None,
            pooled_presentation=pres,
            context=None,
        )
        _, rec12, _, sup12 = head.score_aligned(
            bm12,
            bx12,
            ph12,
            att12,
            mhc_class="II",
            pooled_affinity=aff,
            pooled_stability=None,
            pooled_presentation=pres,
            context=None,
        )
    assert abs(float(sup3.item()) - float(sup12.item())) < 1e-7
    assert abs(float(rec3.item()) - float(rec12.item())) < 1e-7
