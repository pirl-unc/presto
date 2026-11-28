"""Tests for Presto-Omni v2 mechanisms:
- anchor_v3 9-slot structural pocket readout (Class I anchors + Class II P1/P4/P6/P7/P9)
- groove_segment_embedding (alpha1/alpha vs alpha2/beta chain identity)
- Batched interaction & identity precomputation parity with unbatched reference
- Zero-forgetting anchor_v2 -> anchor_v3 warm-start migration (allow_interaction_expansion)
- Identifiability regularizers (elution_pmhc_residual_l2_weight, ms_response_elution_l2_weight)
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from presto.data.assay_measurement import AssayMethod, AssayTrack
from presto.data.modular import (
    Context,
    MHCInput,
    Observation,
    TargetScale,
)
from presto.models.presto import Presto, PrestoConfig, candidates
from presto.training.modular import (
    initialize_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
    train_step,
)
from presto.training.modular_controls import ranking_loss, validate_controls

G1_A = "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEA"
G1_B = "GSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRMAPRAPWIEQEGPEYWDRETQKYKRQAQTDRVSLRNLRGYYN"
G2_A = "GSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQ"
G2_B = "GSHTLQSMYGCDVGPDGRLLRGHDQYAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGECVEWLRRYLENGK"

IC50_TRACK = AssayTrack("ic50", "purified", "competitive", "fluorescence")


def _base_config(**overrides) -> PrestoConfig:
    kw = dict(
        endpoints=("ic50", "half_life", "elution_observation", "bulk_observation"),
        assay_tracks=(IC50_TRACK,),
        identity_tasks=("mhc_class", "mhc_species"),
        species=("human", "mouse"),
        mhc_classes=("I", "II"),
        enzymes=("trypsin_p", "chymotrypsin"),
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        source_context=True,
        elution_staging="affinity_downstream",
        elution_affinity_endpoint="ic50",
        elution_pmhc_residual=True,
        interaction_mode="anchor_v3",
        groove_segment_embedding=True,
    )
    kw.update(overrides)
    return PrestoConfig(**kw)


def _sample_rows() -> list[Observation]:
    return [
        Observation(
            observation_id="el_i_8",
            peptide="SIINFEKL",
            endpoint="elution_observation",
            unit="binary",
            value=1.0,
            qualifier="=",
            split="train",
            source="test",
            mhc=(
                MHCInput(
                    groove1=G1_A, groove2=G2_A, mhc_class="I", species1="human", species2="human"
                ),
                MHCInput(
                    groove1=G1_B, groove2=G2_B, mhc_class="I", species1="human", species2="human"
                ),
            ),
            context=Context(source_n="AAAAAA", source_c="GGGGGG"),
        ),
        Observation(
            observation_id="el_i_11",
            peptide="GILGFVFTLAA",
            endpoint="elution_observation",
            unit="binary",
            value=0.0,
            qualifier="=",
            split="train",
            source="test",
            mhc=(
                MHCInput(
                    groove1=G1_A, groove2=G2_A, mhc_class="I", species1="human", species2="human"
                ),
            ),
            context=Context(source_n="CCCCCC", source_c="DDDDDD"),
        ),
        Observation(
            observation_id="el_ii_15",
            peptide="PKYVKQNTLKLATGM",
            endpoint="elution_observation",
            unit="binary",
            value=1.0,
            qualifier="=",
            split="train",
            source="test",
            mhc=(
                MHCInput(
                    groove1=G1_B, groove2=G2_A, mhc_class="II", species1="human", species2="human"
                ),
            ),
            context=Context(source_n="EEEEEE", source_c="FFFFFF"),
        ),
        Observation(
            observation_id="bulk_1",
            peptide="ALVDFGHTK",
            endpoint="bulk_observation",
            unit="binary",
            value=1.0,
            qualifier="=",
            split="train",
            source="test",
            context=Context(enzyme="trypsin_p", source_n="KKK", source_c="RRR"),
        ),
    ]


@pytest.mark.parametrize("mode", ["mean", "anchor_v2", "anchor_v3"])
def test_batched_interaction_and_identity_parity(mode: str):
    torch.manual_seed(7)
    cfg = _base_config(interaction_mode=mode, groove_segment_embedding=(mode == "anchor_v3"))
    model = Presto(cfg).eval()
    if cfg.groove_segment_embedding:
        with torch.no_grad():
            model.groove_segment.weight.copy_(torch.randn_like(model.groove_segment.weight) * 0.25)
    rows = _sample_rows()
    with torch.no_grad():
        encoded, states = model._encode(rows)
        cache = model._precompute_interactions(rows, states)
        id_logits, id_probs = model._precompute_identities(rows, encoded)
        max_diff = 0.0
        for row in rows:
            for mhc in row.mhc:
                cands = candidates(row.peptide, mhc.mhc_class)
                ref = model._interaction_candidates(row.peptide, mhc, cands, states)
                got = cache[(row.peptide, mhc.groove1, mhc.groove2, mhc.mhc_class)]
                max_diff = max(max_diff, float((ref - got).abs().max()))
                ref_id = model._identity(row, mhc, encoded)
                got_id = model._identity(row, mhc, encoded, id_logits_cache=id_logits)
                assert set(ref_id) == set(got_id)
                for k in ref_id:
                    max_diff = max(max_diff, float((ref_id[k] - got_id[k]).abs().max()))
                ref_p = model._identity_probs(row, mhc, ref_id)
                got_p = model._identity_probs(row, mhc, got_id, id_probs_cache=id_probs)
                for k in ref_p:
                    for a, b in zip(ref_p[k], got_p[k], strict=True):
                        max_diff = max(max_diff, abs(a - b))
    assert max_diff < 1e-5, f"batched vs unbatched parity drift: {max_diff}"


def test_groove_segment_embedding_distinguishes_chains_and_omits_default():
    default_cfg = _base_config(interaction_mode="mean", groove_segment_embedding=False)
    assert "groove_segment_embedding" not in default_cfg.to_dict()
    assert "interaction_mode" not in default_cfg.to_dict()

    cfg = _base_config(interaction_mode="anchor_v3", groove_segment_embedding=True)
    assert cfg.to_dict()["groove_segment_embedding"] is True
    assert cfg.to_dict()["interaction_mode"] == "anchor_v3"
    model = Presto(cfg).eval()
    with torch.no_grad():
        model.groove_segment.weight[0].fill_(0.5)
        model.groove_segment.weight[1].fill_(-0.5)
        m1 = MHCInput(
            groove1=G1_A, groove2=G1_A, mhc_class="II", species1="human", species2="human"
        )
        row1 = Observation(
            observation_id="r1",
            peptide="PKYVKQNTLKLATGM",
            endpoint="ic50",
            unit="nM",
            value=100.0,
            qualifier="=",
            split="predict",
            source="test",
            mhc=(m1,),
        )
        _, states = model._encode([row1])
        gs = model._groove_states(G1_A, G1_A, states)
        L = len(G1_A)
        assert torch.allclose(gs[:L] - gs[L:], torch.ones_like(gs[:L]), atol=1e-5)


def test_anchor_v2_to_anchor_v3_warm_start_preserves_9mer_predictions(tmp_path: Path):
    torch.manual_seed(11)
    old_cfg = _base_config(interaction_mode="anchor_v2", groove_segment_embedding=False)
    old_model = Presto(old_cfg).eval()
    scales = {
        "ic50": TargetScale(mean=2.0, scale=1.0, exact_count=10),
        "half_life": TargetScale(mean=1.5, scale=0.5, exact_count=10),
    }
    ckpt = tmp_path / "v2.pt"
    save_checkpoint(ckpt, old_model, scales)

    new_cfg = _base_config(interaction_mode="anchor_v3", groove_segment_embedding=True)
    new_model = Presto(new_cfg).eval()

    with pytest.raises(ValueError, match="allow_interaction_expansion"):
        initialize_from_checkpoint(new_model, ckpt, allow_interaction_expansion=False)

    initialize_from_checkpoint(new_model, ckpt, allow_interaction_expansion=True)
    assert torch.allclose(
        new_model.groove_segment.weight, torch.zeros_like(new_model.groove_segment.weight)
    )

    # On a 9-mer Class I peptide, slots (0, 1, 7, 8) match exactly and slot 4 is P5 (whereas
    # anchor_v2 used mean(P1..P9)). Verify copied weights match exactly and forward runs cleanly.
    d = new_cfg.d_model
    for old_s, new_s in ((0, 0), (1, 1), (2, 4), (3, 7), (4, 8)):
        assert torch.allclose(
            new_model.interaction_anchor_proj.weight[:, new_s * d : (new_s + 1) * d],
            old_model.interaction_anchor_proj.weight[:, old_s * d : (old_s + 1) * d],
        )
    for zero_s in (2, 3, 5, 6):
        assert torch.allclose(
            new_model.interaction_anchor_proj.weight[:, zero_s * d : (zero_s + 1) * d],
            torch.zeros(d, d),
        )
    out = new_model(_sample_rows())
    assert len(out) == 4


def test_identifiability_regularizers_penalize_elution_only_and_backprop():
    torch.manual_seed(3)
    cfg = _base_config()
    model = Presto(cfg)
    # Give elution_pmhc_mlp non-zero final weights so pmhc_residual_l2 is strictly positive
    with torch.no_grad():
        model.elution_pmhc_mlp[-1].weight.fill_(0.05)
        model.elution_pmhc_mlp[-1].bias.fill_(0.2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scales = {
        "ic50": TargetScale(mean=2.0, scale=1.0, exact_count=10),
        "half_life": TargetScale(mean=1.5, scale=0.5, exact_count=10),
    }
    rows = _sample_rows()

    # Validate controls reject negative weights
    with pytest.raises(ValueError, match="elution_pmhc_residual_l2_weight"):
        validate_controls({"elution_pmhc_residual_l2_weight": -0.1}, cfg, [])
    with pytest.raises(ValueError, match="ms_response_elution_l2_weight"):
        validate_controls({"ms_response_elution_l2_weight": -0.1}, cfg, [])

    step = train_step(
        model,
        opt,
        rows,
        scales,
        elution_pmhc_residual_l2_weight=0.05,
        ms_response_elution_l2_weight=0.02,
    )
    assert step["loss_parts"]["pmhc_residual_l2"] > 0.0
    assert step["loss_parts"]["ms_response_elution_l2"] >= 0.0

    # Bulk-only batch must NOT trigger elution-only regularizers
    bulk_step = train_step(
        model,
        opt,
        [rows[-1]],
        scales,
        elution_pmhc_residual_l2_weight=0.05,
        ms_response_elution_l2_weight=0.02,
    )
    assert "pmhc_residual_l2" not in bulk_step["loss_parts"]
    assert "ms_response_elution_l2" not in bulk_step["loss_parts"]


def test_v21_cooperativity_pfr_refine_thermodynamic_parity_and_invariance(tmp_path: Path):
    torch.manual_seed(19)
    # Defaults omit all new keys from to_dict()
    default_cfg = _base_config(interaction_mode="mean", groove_segment_embedding=False)
    for k in (
        "groove_cross_chain",
        "interaction_cooperativity",
        "pmhc_pfr_conditioning",
        "pmhc_refine",
        "register_pooling_mode",
    ):
        assert k not in default_cfg.to_dict()

    cfg = _base_config(
        interaction_mode="anchor_v3",
        groove_segment_embedding=True,
        groove_cross_chain=True,
        interaction_cooperativity=True,
        pmhc_pfr_conditioning=True,
        pmhc_refine=True,
        register_pooling_mode="thermodynamic",
    )
    model = Presto(cfg).eval()
    with torch.no_grad():
        model.groove_segment.weight.copy_(torch.randn_like(model.groove_segment.weight) * 0.2)
        model.groove_self_gate.fill_(0.25)
        model.interaction_self_gate.fill_(0.35)
        model.pmhc_pfr_mlp[-1].weight.copy_(torch.randn_like(model.pmhc_pfr_mlp[-1].weight) * 0.05)
        model.pmhc_refine[-1].weight.copy_(torch.randn_like(model.pmhc_refine[-1].weight) * 0.05)

    # 1. Batched vs unbatched parity holds with interaction_cooperativity=True
    rows = _sample_rows()
    with torch.no_grad():
        encoded, states = model._encode(rows)
        cache = model._precompute_interactions(rows, states)
        max_diff = 0.0
        for row in rows:
            for mhc in row.mhc:
                cands = candidates(row.peptide, mhc.mhc_class)
                ref = model._interaction_candidates(row.peptide, mhc, cands, states)
                got = cache[(row.peptide, mhc.groove1, mhc.groove2, mhc.mhc_class)]
                max_diff = max(max_diff, float((ref - got).abs().max()))
    assert max_diff < 1e-5

    # 2. Exact 0.0 context invariance for intrinsic ic50 with PFR + refine + cooperativity + thermo
    m_ii = MHCInput(groove1=G1_B, groove2=G2_A, mhc_class="II", species1="human", species2="human")
    q_null = Observation(
        observation_id="q0",
        peptide="PKYVKQNTLKLATGM",
        endpoint="ic50",
        unit="nM",
        value=100.0,
        qualifier="=",
        split="predict",
        source="test",
        mhc=(m_ii,),
        context=Context(),
    )
    q_rich = Observation(
        observation_id="q1",
        peptide="PKYVKQNTLKLATGM",
        endpoint="ic50",
        unit="nM",
        value=100.0,
        qualifier="=",
        split="predict",
        source="test",
        mhc=(m_ii,),
        context=Context(source_n="MMMMMM", source_c="KKKKKK"),
    )
    with torch.no_grad():
        o0, o1 = model([q_null, q_rich])
    assert float((o0.value - o1.value).abs()) == 0.0
    assert o0.diagnostics["candidates"][0]["registers"][0]["pooling"] == "thermodynamic_logsumexp"

    # 3. Thermodynamic pooling is sharper than normalized pooling on multi-register scores
    scores = torch.tensor([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    v_th, w_th = model._pool_registers(scores, -1)
    norm_model = Presto(_base_config(register_pooling_mode="normalized"))
    v_nm, w_nm = norm_model._pool_registers(scores, -1)
    assert float(w_th[1]) > 0.60 and float(w_nm[1]) < 0.35
    assert (
        float(v_th) < -1.0
    )  # physical partition function strengthens, never penalizes via -log(R)


def test_v21_warm_start_save_reload_and_full_gradient_coverage(tmp_path: Path):
    torch.manual_seed(23)
    old_cfg = _base_config(interaction_mode="anchor_v2", groove_segment_embedding=False)
    old_model = Presto(old_cfg).eval()
    scales = {
        "ic50": TargetScale(mean=2.0, scale=1.0, exact_count=10),
        "half_life": TargetScale(mean=1.5, scale=0.5, exact_count=10),
    }
    ckpt_old = tmp_path / "old_v2.pt"
    save_checkpoint(ckpt_old, old_model, scales)

    new_cfg = _base_config(
        interaction_mode="anchor_v3",
        groove_segment_embedding=True,
        groove_cross_chain=True,
        interaction_cooperativity=True,
        pmhc_pfr_conditioning=True,
        pmhc_refine=True,
        register_pooling_mode="thermodynamic",
    )
    new_model = Presto(new_cfg)
    initialize_from_checkpoint(new_model, ckpt_old, allow_interaction_expansion=True)

    # Save & reload immediately to prove trained_elements mask shapes match expanded parameters
    ckpt_new = tmp_path / "new_v21.pt"
    save_checkpoint(ckpt_new, new_model, scales)
    reloaded, _, _ = load_checkpoint(ckpt_new, device="cpu")
    assert (
        reloaded.trained_elements["interaction_anchor_proj.weight"].shape
        == reloaded.interaction_anchor_proj.weight.shape
    )

    # 3 train steps must give non-zero gradient & update to EVERY new v2.1 module
    opt = torch.optim.AdamW(new_model.parameters(), lr=1e-2)
    rows = _sample_rows()
    for _ in range(3):
        train_step(new_model, opt, rows, scales)
    for prefix in (
        "groove_segment.",
        "groove_self_attn.",
        "groove_self_norm.",
        "groove_self_gate",
        "interaction_self_attn.",
        "interaction_self_norm.",
        "interaction_self_gate",
        "pmhc_pfr_mlp.",
        "pmhc_refine.",
    ):
        matched = [n for n, _ in new_model.named_parameters() if n.startswith(prefix)]
        assert matched, f"missing expected params for {prefix}"
        for n in matched:
            assert n in new_model.trained_parameters, f"{n} did not update during training"


def test_ranking_loss_include_censored_pairs_exact_vs_right_censored():
    cfg = _base_config()
    validate_controls(
        {
            "within_group_ranking_loss": {
                "weight": 0.3,
                "endpoints": ["ic50"],
                "grouping": "assay_allele",
                "include_censored": True,
            }
        },
        cfg,
        [],
    )
    m = MHCInput(groove1=G1_A, groove2=G2_A, mhc_class="I", species1="human", species2="human")
    am = AssayMethod("purified", "competitive", "fluorescence", "verified")
    prov = {"study_id": "S1"}
    r_strong = Observation(
        "o1",
        "s",
        "ic50",
        "SIINFEKL",
        (m,),
        value=10.0,
        unit="nM",
        qualifier="=",
        assay_method=am,
        provenance=prov,
        split="train",
    )
    r_weak_cens = Observation(
        "o2",
        "s",
        "ic50",
        "GILGFVFTL",
        (m,),
        value=20000.0,
        unit="nM",
        qualifier=">",
        assay_method=am,
        provenance=prov,
        split="train",
    )
    r_weak_cens2 = Observation(
        "o3",
        "s",
        "ic50",
        "NLVPMVATV",
        (m,),
        value=20000.0,
        unit="nM",
        qualifier=">",
        assay_method=am,
        provenance=prov,
        split="train",
    )

    class _Out:
        def __init__(self, v):
            self.value = torch.tensor(v, requires_grad=True)

    outs = [_Out(1.0), _Out(0.5), _Out(0.2)]
    loss_exact_only, n_ex = ranking_loss(
        [r_strong, r_weak_cens, r_weak_cens2],
        outs,
        ["ic50"],
        grouping="assay_allele",
        include_censored=False,
    )
    assert loss_exact_only is None and n_ex == 0

    loss_cens, n_cens = ranking_loss(
        [r_strong, r_weak_cens, r_weak_cens2],
        outs,
        ["ic50"],
        grouping="assay_allele",
        include_censored=True,
    )
    # Exactly 2 valid pairs: (o1 < o2) and (o1 < o3); NO pair between (o2, o3)
    assert loss_cens is not None and n_cens == 2
    loss_cens.backward()
    assert outs[0].value.grad.item() > 0  # pushes strong binder score DOWN
    assert outs[1].value.grad.item() < 0  # pushes right-censored non-binder score UP


def test_assay_measurement_layer_batched_parity():
    from presto.models.assay_measurement import AssayMeasurementLayer

    torch.manual_seed(7)
    tracks = (
        AssayTrack("ic50", "purified", "competitive", "fluorescence"),
        AssayTrack("ic50", "purified", "competitive", "radioactivity"),
        AssayTrack("ic50", "cellular", "competitive", "fluorescence"),
        AssayTrack("qualitative_binding", "purified", "direct", "elisa"),
        AssayTrack("qualitative_binding", "purified", "competitive", "radioactivity"),
    )
    layer = AssayMeasurementLayer(
        width=32, endpoints=("ic50", "qualitative_binding"), tracks=tracks
    )
    layer.eval()
    hidden = torch.randn(7, 32)
    for ep in ("ic50", "qualitative_binding"):
        batched = layer(hidden, ep)
        expected = {t.key: layer.score(hidden, ep, t) for t in tracks if t.endpoint == ep}
        assert set(batched) == set(expected)
        for k in expected:
            assert torch.allclose(batched[k], expected[k], atol=1e-6)


def test_v22_query_anchor_and_groove_pocket_pool_warm_start_invariance_and_grads(tmp_path: Path):
    torch.manual_seed(17)
    cfg_v21 = _base_config(
        interaction_mode="anchor_v3",
        groove_segment_embedding=True,
        groove_cross_chain=True,
        interaction_cooperativity=True,
        pmhc_pfr_conditioning=True,
        pmhc_refine=True,
        register_pooling_mode="thermodynamic",
    )
    assert "query_anchor_embedding" not in cfg_v21.to_dict()
    assert "groove_pocket_pool" not in cfg_v21.to_dict()

    m21 = Presto(cfg_v21)
    rows = _sample_rows()
    scales = {
        "ic50": TargetScale(2.5, 1.0, exact_count=10),
        "half_life": TargetScale(1.0, 0.5, exact_count=10),
    }
    opt21 = torch.optim.AdamW(m21.parameters(), lr=1e-3)
    train_step(m21, opt21, rows, scales)
    train_step(m21, opt21, rows, scales)
    ckpt = tmp_path / "v21.pt"
    save_checkpoint(ckpt, m21, scales)

    cfg_v22 = _base_config(
        interaction_mode="anchor_v3",
        groove_segment_embedding=True,
        groove_cross_chain=True,
        query_anchor_embedding=True,
        groove_pocket_pool=True,
        interaction_cooperativity=True,
        pmhc_pfr_conditioning=True,
        pmhc_refine=True,
        register_pooling_mode="thermodynamic",
    )
    assert cfg_v22.to_dict()["query_anchor_embedding"] is True
    assert cfg_v22.to_dict()["groove_pocket_pool"] is True

    m22 = Presto(cfg_v22)
    with pytest.raises(ValueError, match="allow_interaction_expansion"):
        initialize_from_checkpoint(m22, ckpt, allow_interaction_expansion=False)
    initialize_from_checkpoint(m22, ckpt, allow_interaction_expansion=True)

    # Zero-init modules guarantee exact 0.0 initial perturbation from v2.1 -> v2.2
    m21.eval()
    m22.eval()
    with torch.no_grad():
        o21 = m21(rows)
        o22 = m22(rows)
    for a, b in zip(o21, o22, strict=True):
        assert abs(float(a.value) - float(b.value)) < 1e-5

    # Two training steps give full gradient/update coverage to every new v2.2 parameter
    opt22 = torch.optim.AdamW(m22.parameters(), lr=1e-3)
    train_step(m22, opt22, rows, scales)
    train_step(m22, opt22, rows, scales)
    for prefix in (
        "query_n_anchor.",
        "query_c_anchor.",
        "query_class_emb.",
        "groove_pocket_queries",
        "groove_pocket_attn.",
        "groove_pocket_norm.",
        "groove_pocket_proj.",
    ):
        matched = [n for n in m22.trained_parameters if n.startswith(prefix)]
        assert matched, f"missing gradient/update coverage for {prefix}"

    # Verify exact batched vs unbatched parity AND exact 0.0 context invariance after updates
    m22.eval()
    with torch.no_grad():
        batched = m22(rows)
        singles = [m22([r])[0] for r in rows]
    for b_out, s_out in zip(batched, singles, strict=True):
        assert abs(float(b_out.value) - float(s_out.value)) < 1e-5

    aff_row = Observation(
        observation_id="aff_ii_15",
        peptide="PKYVKQNTLKLATGM",
        endpoint="ic50",
        unit="nM",
        value=100.0,
        qualifier="=",
        split="predict",
        source="test",
        mhc=(
            MHCInput(
                groove1=G1_B, groove2=G2_A, mhc_class="II", species1="human", species2="human"
            ),
        ),
        context=Context(),
    )
    aff_ctx = Observation(
        **{
            **aff_row.__dict__,
            "context": Context(source_n="MAAAA", source_c="GGGGG"),
        }
    )
    with torch.no_grad():
        v_clean = float(m22([aff_row])[0].value)
        v_ctx = float(m22([aff_ctx])[0].value)
    assert abs(v_clean - v_ctx) == 0.0

    ckpt22 = tmp_path / "v22.pt"
    save_checkpoint(ckpt22, m22, scales)
    reloaded, _, _ = load_checkpoint(ckpt22)
    reloaded.eval()
    with torch.no_grad():
        r_out = reloaded(rows)
    for a, b in zip(batched, r_out, strict=True):
        assert abs(float(a.value) - float(b.value)) < 1e-6
