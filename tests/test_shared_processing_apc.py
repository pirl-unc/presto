"""Regression tests for SharedProcessingModule, signed APC perturbations, and APC contrasts."""

from __future__ import annotations

from dataclasses import replace
import random

import pytest
import torch

from presto.data.modular import Context, ContextAssertion, Exposure
from presto.data.modular_context import APM_HOST_GENES, supported_perturbation_genes
from presto.inference.modular import ModularPredictor
from presto.models import apm_state
from presto.models.modular_dependencies import parameter_dependencies
from presto.training.apc_contrasts import APCContrastPool
from presto.training.modular import (
    initialize_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
    train_step,
)
from presto.training.processing_pairs import pair_loss

from test_apm_state_bottleneck import elution, scales
from test_harmonized_apc import build
from test_processing_pairs import pair


def _example(cls: str = "I", peptide: str | None = None, **ctx_kwargs):
    row = elution(cls)
    if peptide is not None:
        row = replace(row, peptide=peptide)
    default_ctx = dict(source_n="RSTVWY", source_c="ACDEFG", n_terminus=False, c_terminus=False)
    default_ctx.update(ctx_kwargs)
    return replace(row, context=Context(**default_ctx))


def test_apm_host_genes_cover_apm_state_registry():
    expected = {
        g.replace("_", "-") if g.startswith("HLA_") else g for g in apm_state.APM_GENE_STEPS
    }
    assert expected <= APM_HOST_GENES
    genes = supported_perturbation_genes(
        (
            "knockout:B2M",
            "knockout:TAP1",
            "transduction:CIITA",
            "transfection:HLA-DM",
            "transduction:HIV Env",
            "drug:decitabine",
        )
    )
    assert genes == ("B2M", "CIITA", "HLA-DM", "TAP1")


def test_shared_processing_warmstart_parity_and_checkpoint_roundtrip(tmp_path):
    base = build(
        seed=11,
        source_context=True,
        cytokines=("IFNG",),
        interventions=(
            "knockout:B2M",
            "knockout:TAP1",
            "transduction:CIITA",
            "transfection:HLA-DM",
        ),
    )
    rng = torch.random.get_rng_state()
    expanded = build(
        seed=11,
        source_context=True,
        cytokines=("IFNG",),
        interventions=(
            "knockout:B2M",
            "knockout:TAP1",
            "transduction:CIITA",
            "transfection:HLA-DM",
        ),
        elution_shared_processing=True,
    )
    assert torch.equal(rng, torch.random.get_rng_state())
    assert "elution_shared_processing" not in base.config.to_dict()
    assert expanded.config.to_dict()["elution_shared_processing"] is True
    for name, value in base.state_dict().items():
        assert torch.equal(value, expanded.state_dict()[name]), name

    rows = [
        _example("I", peptide="GILGFVFTL", cytokines={"IFNG": Exposure("present")}),
        _example(
            "II",
            peptide="PKYVKQNTLKLAT",
            interventions={"transduction:CIITA": ContextAssertion("present")},
        ),
    ]
    for a, b in zip(base(rows), expanded(rows), strict=True):
        torch.testing.assert_close(a.value, b.value, rtol=0, atol=0)

    parent_pt = tmp_path / "parent.pt"
    save_checkpoint(parent_pt, base, scales())
    with pytest.raises(ValueError, match="explicit allow_shared_processing_expansion"):
        initialize_from_checkpoint(expanded, parent_pt)
    initialize_from_checkpoint(expanded, parent_pt, allow_shared_processing_expansion=True)

    child_pt = tmp_path / "child.pt"
    save_checkpoint(child_pt, expanded, scales())
    with pytest.raises(ValueError, match="cannot remove pretrained shared processing"):
        initialize_from_checkpoint(base, child_pt)
    reloaded, _, _ = load_checkpoint(child_pt)
    for a, b in zip(expanded(rows), reloaded.eval()(rows), strict=True):
        torch.testing.assert_close(a.value, b.value, rtol=0, atol=0)


def test_shared_processing_active_without_exterior_flanks_and_preserves_affinity():
    net = build(
        seed=5,
        source_context=True,
        cytokines=("IFNG",),
        interventions=("knockout:B2M", "knockout:TAP1", "transduction:CIITA"),
        elution_shared_processing=True,
    )
    with torch.no_grad():
        net.elution_shared.supply_readout.weight.fill_(0.15)
        net.elution_shared.register_readout.weight.fill_(0.10)

    # Even with source_n=None, source_c=None, interior cut motifs and length alter presentation
    r_short = replace(elution("I"), peptide="GILGFVFTL", context=Context())
    r_long = replace(elution("I"), peptide="GILGFVFTLAA", context=Context())
    out_short, out_long = net([r_short, r_long])
    assert not torch.allclose(out_short.value, out_long.value)

    # Intrinsic affinity dependencies and outputs are strictly untouched
    pres_deps = parameter_dependencies(net, "elution_observation")
    aff_deps = parameter_dependencies(net, "ic50")
    shared_params = {n for n, _ in net.named_parameters() if n.startswith("elution_shared.")}
    assert shared_params <= pres_deps
    assert not (shared_params & aff_deps)

    aff_base = replace(r_short, endpoint="ic50", unit="nM", value=50.0)
    aff_pert = replace(
        aff_base,
        context=Context(
            source_n="MMMMMM",
            source_c="KKKKKK",
            cytokines={"IFNG": Exposure("present")},
            interventions={"knockout:B2M": ContextAssertion("present")},
        ),
    )
    a, b = net([aff_base, aff_pert])
    torch.testing.assert_close(a.value, b.value, rtol=0, atol=0)


def test_pathway_scoping_prevents_cross_class_perturbation_leakage():
    net = build(
        seed=7,
        source_context=True,
        cytokines=("IFNG",),
        interventions=(
            "knockout:B2M",
            "knockout:TAP1",
            "knockout:HLA-A",
            "transduction:CIITA",
            "transfection:HLA-DM",
        ),
        elution_shared_processing=True,
    )
    with torch.no_grad():
        for name, param in net.named_parameters():
            if name.startswith(("apm_state.", "elution_shared.")):
                param.normal_(mean=0.2, std=0.1)

    # Class-I perturbations have zero effect on Class-II observations
    base_ii = _example("II", peptide="PKYVKQNTLKLAT")
    for claim in ("knockout:B2M", "knockout:TAP1", "knockout:HLA-A", "overexpression:TAP1"):
        pert_ii = _example(
            "II",
            peptide="PKYVKQNTLKLAT",
            interventions={claim: ContextAssertion("present")},
        )
        a, b = net([base_ii, pert_ii])
        torch.testing.assert_close(a.value, b.value, rtol=0, atol=0)

    # Class-II perturbations have zero effect on Class-I observations
    base_i = _example("I", peptide="GILGFVFTL")
    for claim in ("transduction:CIITA", "transfection:HLA-DM", "knockout:CIITA", "knockout:HLA-DM"):
        pert_i = _example(
            "I",
            peptide="GILGFVFTL",
            interventions={claim: ContextAssertion("present")},
        )
        a, b = net([base_i, pert_i])
        torch.testing.assert_close(a.value, b.value, rtol=0, atol=0)


def test_training_monotonic_ko_kd_wt_oe_and_cytokine_semantics_with_strict_predict(tmp_path):
    interventions = (
        "knockout:B2M",
        "knockout:TAP1",
        "knockout:HLA-A",
        "knockdown:TAX1BP1",
        "transduction:CIITA",
        "transfection:HLA-DM",
    )
    net = build(
        seed=13,
        source_context=True,
        cytokines=("IFNG",),
        interventions=interventions,
        elution_shared_processing=True,
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3)

    # Build a small realistic multi-arm TRAIN corpus covering all configured factors & flanks
    mhc_i_a = replace(
        elution("I").mhc[0],
        provenance={"catalog_alleles": ["HLA-A*02:01"]},
    )
    base_i = replace(
        _example("I", peptide="GILGFVFTL", n_terminus=True, c_terminus=False),
        mhc=(mhc_i_a,),
        provenance={"study_id": "s_i", "source_protein": "protA"},
    )
    neg_i = replace(
        _example("I", peptide="KLWESPQRT", n_terminus=False, c_terminus=True),
        observation_id="neg_i",
        mhc=(mhc_i_a,),
        value=0.0,
        provenance={"study_id": "s_i", "source_protein": "protA"},
    )
    ko_tap1_pos = replace(
        base_i,
        observation_id="ko_tap1",
        peptide="MLVLLVAVV",
        context=replace(
            base_i.context,
            interventions={"knockout:TAP1": ContextAssertion("present")},
            cytokines={"IFNG": Exposure("absent")},
        ),
    )
    ko_b2m_pos = replace(
        base_i,
        observation_id="ko_b2m",
        peptide="LLVAVVAAA",
        context=replace(
            base_i.context,
            interventions={"knockout:B2M": ContextAssertion("present")},
        ),
    )
    ko_hlaa_pos = replace(
        base_i,
        observation_id="ko_hlaa",
        peptide="SLYNTVATL",
        context=replace(
            base_i.context,
            interventions={"knockout:HLA-A": ContextAssertion("present")},
        ),
    )
    ifng_pos = replace(
        base_i,
        observation_id="ifng_i",
        peptide="ELAGIGILTV",
        context=replace(
            base_i.context,
            cytokines={"IFNG": Exposure("present")},
            interventions={k: ContextAssertion("absent") for k in interventions},
        ),
    )

    base_ii = replace(
        _example("II", peptide="PKYVKQNTLKLAT", n_terminus=False, c_terminus=False),
        provenance={"study_id": "s_ii", "source_protein": "protB"},
    )
    neg_ii = replace(
        _example("II", peptide="APGDRGPEPPQAA", n_terminus=True, c_terminus=True),
        observation_id="neg_ii",
        value=0.0,
        provenance={"study_id": "s_ii", "source_protein": "protB"},
    )
    ciita_pos = replace(
        base_ii,
        observation_id="ciita_ii",
        peptide="GELIGILNAAKVPAD",
        context=replace(
            base_ii.context,
            interventions={"transduction:CIITA": ContextAssertion("present")},
        ),
    )
    dm_pos = replace(
        base_ii,
        observation_id="dm_ii",
        peptide="AAYAAAKAAALAA",
        context=replace(
            base_ii.context,
            interventions={"transfection:HLA-DM": ContextAssertion("present")},
            cytokines={"IFNG": Exposure("present")},
        ),
    )
    tax_kd_neg = replace(
        neg_ii,
        observation_id="tax_kd_neg",
        peptide="GLKPGKLLPLAGP",
        context=replace(
            neg_ii.context,
            interventions={"knockdown:TAX1BP1": ContextAssertion("present")},
        ),
    )
    aff_i = replace(base_i, observation_id="aff_i", endpoint="ic50", unit="nM", value=50.0)
    aff_ii = replace(base_ii, observation_id="aff_ii", endpoint="ic50", unit="nM", value=120.0)

    train_rows = [
        base_i,
        neg_i,
        ko_tap1_pos,
        ko_b2m_pos,
        ko_hlaa_pos,
        ifng_pos,
        base_ii,
        neg_ii,
        ciita_pos,
        dm_pos,
        tax_kd_neg,
        aff_i,
        aff_ii,
    ]
    pool = APCContrastPool(train_rows, net.config)
    assert pool.counts["selectivity_factors"] >= 4
    assert (
        pool.counts["yield_factors"] >= 6
    )  # 5 genes + IFNG (+ cross-locus orthogonality if >1 locus)

    p_i, b_i = pair("I")
    p_ii, b_ii = pair("II")
    p_i = replace(p_i, mhc=(mhc_i_a,))
    b_i = replace(b_i, mhc=(mhc_i_a,))

    rng = random.Random(42)
    for _ in range(32):
        contrasts = pool.draw(train_rows, rng, selectivity_per_step=6, yield_per_step=8)
        step = train_step(
            net,
            optimizer,
            train_rows,
            scales(),
            processing_pairs=[(p_i, b_i), (p_ii, b_ii)],
            processing_pair_weight=0.5,
            processing_pair_selection_weight=1.0,
            processing_pair_margin=0.6,
            apc_contrasts=contrasts,
            apc_contrast_options={"weight": 0.8, "yield_weight": 1.0, "margin": 0.2},
        )
    assert "apc_selectivity_loss" in step["loss_parts"]
    assert "apc_yield_loss" in step["loss_parts"]
    assert step["loss_parts"]["processing_pair_selection_weight"] == 1.0
    assert step["loss_parts"]["processing_pair_margin"] == 0.6

    # Every parameter in elution_shared must have gradient+update support
    shared_names = {n for n, _ in net.named_parameters() if n.startswith("elution_shared.")}
    assert shared_names <= net.trained_parameters

    # Save & run strict ModularPredictor.predict (which enforces feature_inputs(enforce=True))
    ckpt = tmp_path / "trained.pt"
    save_checkpoint(ckpt, net, scales())
    predictor = ModularPredictor(ckpt)

    def score_i(**ctx_kw):
        q = replace(base_i, split="predict", value=None, context=replace(base_i.context, **ctx_kw))
        return predictor.predict([q], unsupported_apc_policy="unknown")[0]["logit"]

    # Class-I KO < KD < WT < OE continuity via shared signed gene axis
    s_tap1_ko = score_i(interventions={"knockout:TAP1": ContextAssertion("present")})
    s_tap1_kd = score_i(interventions={"knockdown:TAP1": ContextAssertion("present")})
    s_wt_i = score_i(interventions={})
    s_tap1_oe = score_i(interventions={"overexpression:TAP1": ContextAssertion("present")})
    assert s_tap1_ko < s_tap1_kd < s_wt_i < s_tap1_oe

    s_b2m_ko = score_i(interventions={"knockout:B2M": ContextAssertion("present")})
    s_b2m_oe = score_i(interventions={"overexpression:B2M": ContextAssertion("present")})
    assert s_b2m_ko < s_wt_i < s_b2m_oe

    # Cytokine monotonicity: IFNG absent < IFNG unknown < IFNG present
    s_ifng_abs = score_i(cytokines={"IFNG": Exposure("absent")})
    s_ifng_unk = score_i(cytokines={})
    s_ifng_pres = score_i(cytokines={"IFNG": Exposure("present")})
    assert s_ifng_abs < s_ifng_unk < s_ifng_pres

    # Class-II KO < WT < transduction/transfection/overexpression continuity
    def score_ii(**ctx_kw):
        q = replace(
            base_ii, split="predict", value=None, context=replace(base_ii.context, **ctx_kw)
        )
        return predictor.predict([q], unsupported_apc_policy="unknown")[0]["logit"]

    s_ciita_ko = score_ii(interventions={"knockout:CIITA": ContextAssertion("present")})
    s_wt_ii = score_ii(interventions={})
    s_ciita_tr = score_ii(interventions={"transduction:CIITA": ContextAssertion("present")})
    assert s_ciita_ko < s_wt_ii < s_ciita_tr

    s_dm_ko = score_ii(interventions={"knockout:HLA-DM": ContextAssertion("present")})
    s_dm_tr = score_ii(interventions={"transfection:HLA-DM": ContextAssertion("present")})
    assert s_dm_ko < s_wt_ii < s_dm_tr

    # Knockdown-only discrete vocabulary token (like TAX1BP1) still obeys KO < KD < WT < OE
    s_tax_ko = score_ii(interventions={"knockout:TAX1BP1": ContextAssertion("present")})
    s_tax_kd = score_ii(interventions={"knockdown:TAX1BP1": ContextAssertion("present")})
    s_tax_oe = score_ii(interventions={"overexpression:TAX1BP1": ContextAssertion("present")})
    assert s_tax_ko < s_tax_kd < s_wt_ii < s_tax_oe


def test_pair_loss_logit_weight_supervises_both_supply_and_presentation_logit():
    net = build(source_context=True, elution_shared_processing=True)
    p, b = pair("II")
    outputs = net([p, b])
    loss_supply_only = pair_loss(outputs, logit_weight=0.0)
    loss_both = pair_loss(outputs, logit_weight=0.5)
    assert float(loss_both.detach()) > float(loss_supply_only.detach())
    loss_both.backward()
    # Register overhang readout enters selection -> output.value, trained by logit_weight > 0
    assert net.elution_shared.register_readout.weight.grad.abs().sum() > 0


def test_pair_loss_selection_weight_excludes_intrinsic_binding_gradients():
    net = build(source_context=True, elution_shared_processing=True)
    p, b = pair("II")
    outputs = net([p, b])
    loss_sel = pair_loss(outputs, logit_weight=0.0, selection_weight=1.0)
    loss_sel.backward()
    assert net.elution_shared.supply_readout.weight.grad.abs().sum() > 0
    assert net.elution_shared.register_readout.weight.grad.abs().sum() > 0
    assert net.elution_affinity_weight.grad is None
    for name, param in net.named_parameters():
        if name.startswith(("assay_layer.", "pmhc_fusion.")):
            assert param.grad is None, name


def test_hla_cross_locus_orthogonality_and_control_validation():
    from presto.training.apc_contrasts import apc_contrast_loss
    from presto.training.modular_controls import validate_controls

    net = build(
        seed=23,
        source_context=True,
        cytokines=("IFNG",),
        interventions=(
            "knockout:HLA-A",
            "knockout:HLA-B",
            "knockout:HLA-C",
            "transduction:HLA-B",
        ),
        elution_shared_processing=True,
    )
    # Pan APC columns exclude HLA-A/B/C so locus KOs never shift pan peptide supply
    assert net.elution_shared.apc_feature_names() == ["shared_apc:cytokine:IFNG:signed"]
    assert len(net.elution_shared.hla_columns) == 3

    mhc_a = replace(
        elution("I").mhc[0],
        groove1="GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEA",
        groove2="GSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTD",
        provenance={"catalog_alleles": ["HLA-A*02:01"]},
    )
    mhc_b = replace(
        elution("I").mhc[0],
        groove1="GSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRTEPRAPWIEQEGPEYWDRNTQIFKTNTQTYRENLRIALRYYNQSEA",
        groove2="GSHTWQTMYGCDVGPDGRLLRGHNQYAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAAREAEQLRAYLEGLCVEWLRRHLENGKETLQRAD",
        provenance={"catalog_alleles": ["HLA-B*51:01"]},
    )
    row_a = replace(_example("I", peptide="GILGFVFTL"), observation_id="ra", mhc=(mhc_a,))
    row_b = replace(_example("I", peptide="LPFDKTTIM"), observation_id="rb", mhc=(mhc_b,))

    pool = APCContrastPool([row_a, row_b], net.config)
    assert "hla_orth:HLA-A_on_HLA-B" in pool.yield_buckets
    assert "hla_orth:AC_on_HLA-B" in pool.yield_buckets

    opt = torch.optim.AdamW(net.parameters(), lr=5e-3)
    rng = random.Random(7)
    for _ in range(55):
        opt.zero_grad()
        _, y_draws, _ = pool.draw([row_a, row_b], rng, selectivity_per_step=0, yield_per_step=12)
        loss, _ = apc_contrast_loss(net, [], y_draws, yield_weight=1.0, margin=0.2)
        loss.backward()
        opt.step()

    net.eval()

    def _eval_val(r):
        return float(net([r])[0].value.detach())

    # On HLA-B*51:01, HLA-A+HLA-C KO has near-zero effect while HLA-B KO < WT < rescue < OE
    b_wt = _eval_val(row_b)
    b_ac_ko = _eval_val(
        replace(
            row_b,
            context=replace(
                row_b.context,
                interventions={
                    "knockout:HLA-A": ContextAssertion("present"),
                    "knockout:HLA-C": ContextAssertion("present"),
                },
            ),
        )
    )
    b_b_ko = _eval_val(
        replace(
            row_b,
            context=replace(
                row_b.context,
                interventions={"knockout:HLA-B": ContextAssertion("present")},
            ),
        )
    )
    b_b_rescue = _eval_val(
        replace(
            row_b,
            context=replace(
                row_b.context,
                interventions={
                    "knockout:HLA-A": ContextAssertion("present"),
                    "knockout:HLA-B": ContextAssertion("present"),
                    "knockout:HLA-C": ContextAssertion("present"),
                    "transduction:HLA-B": ContextAssertion("present"),
                },
            ),
        )
    )
    b_b_oe = _eval_val(
        replace(
            row_b,
            context=replace(
                row_b.context,
                interventions={"transduction:HLA-B": ContextAssertion("present")},
            ),
        )
    )
    assert abs(b_ac_ko - b_wt) < 0.05
    assert b_b_ko < b_wt < b_b_rescue < b_b_oe

    validate_controls(
        {
            "batch_size": 16,
            "max_steps": 100,
            "sampling_mode": "weighted_steps",
            "new_module_lr_scale": 8.0,
            "processing_pair_training": {
                "weight": 0.8,
                "pairs_per_class": 6,
                "selection_weight": 1.0,
            },
            "selection_endpoint": None,
            "selection_reduction": "observation_mean",
        },
        net.config,
        ["endpoint"],
    )


def test_apc_vocab_expansion_and_half_life_preserves_step0_outputs(tmp_path):
    """Adding half_life and superset APC interventions preserves step-0 predictions exactly."""
    from presto.models.modular import ModularConfig, ModularPresto

    torch.manual_seed(17)
    base_cfg = ModularConfig(
        endpoints=("ic50", "elution_observation"),
        mhc_classes=("I", "II"),
        species=("human",),
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        source_context=True,
        elution_staging="affinity_downstream",
        elution_affinity_endpoint="ic50",
        elution_shared_processing=True,
        state_mode="harmonized",
        state_bottleneck=8,
        cytokines=("IFNG",),
        interventions=(
            "knockout:TAP1",
            "knockout:HLA-B",
            "transduction:HLA-B",
            "transfection:HLA-DM",
        ),
    )
    base_model = ModularPresto(base_cfg)
    with torch.no_grad():
        base_model.elution_shared.supply_readout.weight.fill_(0.25)
        base_model.elution_shared.yield_theta.fill_(0.3)
        base_model.elution_shared.logit_temp.copy_(torch.tensor([0.2, -0.15]))
    ckpt = tmp_path / "base.pt"
    save_checkpoint(ckpt, base_model, scales())

    exp_cfg = replace(
        base_cfg,
        endpoints=("ic50", "half_life", "elution_observation"),
        interventions=(
            *base_cfg.interventions,
            "knockout:TAPBPR",
            "transduction:TAPBPR",
            "transduction:HLA-DM",
            "drug:palbociclib",
        ),
    )
    exp_model = ModularPresto(exp_cfg)
    initialize_from_checkpoint(exp_model, ckpt, allow_apc_vocab_expansion=True)

    base_model.eval()
    exp_model.eval()
    q_i = replace(
        _example(
            "I",
            "SIINFEKL",
            cytokines={"IFNG": Exposure("present")},
            interventions={"knockout:TAP1": ContextAssertion("present")},
        ),
        value=None,
        split="predict",
    )
    q_ii = replace(
        _example(
            "II",
            "AAYAAYAAYAAYAA",
            interventions={"transfection:HLA-DM": ContextAssertion("present")},
        ),
        value=None,
        split="predict",
    )
    q_aff = replace(q_i, observation_id="q_aff", endpoint="ic50", unit="nM")
    for q in (q_i, q_ii, q_aff):
        v_base = float(base_model([q])[0].value.detach())
        v_exp = float(exp_model([q])[0].value.detach())
        assert abs(v_base - v_exp) < 1e-6


def test_quantitative_contrast_and_tapbpr_half_life_training():
    """Quantitative perturbation contrasts & half_life train cleanly with zero affinity leakage."""
    from presto.models.modular import ModularConfig, ModularPresto
    from presto.training.apc_contrasts import QUANT_CONTRAST_KEY, apc_contrast_loss

    torch.manual_seed(29)
    cfg = ModularConfig(
        endpoints=("ic50", "half_life", "elution_observation"),
        mhc_classes=("I", "II"),
        species=("human",),
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        source_context=True,
        elution_staging="affinity_downstream",
        elution_affinity_endpoint="ic50",
        elution_shared_processing=True,
        state_mode="harmonized",
        state_bottleneck=8,
        cytokines=("IFNG",),
        interventions=(
            "drug:DMSO",
            "drug:palbociclib",
            "knockout:TAPBPR",
            "transduction:TAPBPR",
            "transfection:HLA-DM",
        ),
    )
    net = ModularPresto(cfg)

    base_i = replace(_example("I", "GILGFVFTL"), observation_id="b_i", value=1.0)
    base_ii = replace(_example("II", "AAYAAYAAYAAYAA"), observation_id="b_ii", value=1.0)
    q_up = replace(
        _example(
            "I",
            "NLVPMVATV",
            cytokines={"IFNG": Exposure("present")},
            interventions={"drug:palbociclib": ContextAssertion("present")},
        ),
        observation_id="q_up",
        value=1.0,
        provenance={
            QUANT_CONTRAST_KEY: {
                "plex_id": "stopfer_mel_ifng",
                "target_log2_fc": 2.5,
                "control_interventions": {"drug:DMSO": "present"},
                "control_cytokines": {"IFNG": "absent"},
            }
        },
    )
    q_dn = replace(
        _example(
            "I",
            "LLFGYPVYV",
            cytokines={"IFNG": Exposure("present")},
            interventions={"drug:palbociclib": ContextAssertion("present")},
        ),
        observation_id="q_dn",
        value=1.0,
        provenance={
            QUANT_CONTRAST_KEY: {
                "plex_id": "stopfer_mel_ifng",
                "target_log2_fc": -2.0,
                "control_interventions": {"drug:DMSO": "present"},
                "control_cytokines": {"IFNG": "absent"},
            }
        },
    )

    pool = APCContrastPool([base_i, base_ii, q_up, q_dn], cfg)
    assert pool.auxiliary_indices == {2, 3}
    assert pool.counts["quant_plexes"] == 1
    assert pool.counts["quant_items"] == 2
    assert "gene:TAPBPL" in pool.yield_buckets

    opt = torch.optim.AdamW(net.parameters(), lr=6e-3)
    rng = random.Random(11)
    for _ in range(50):
        opt.zero_grad()
        _, y_draws, q_draws = pool.draw(
            [base_i, base_ii, q_up, q_dn],
            rng,
            selectivity_per_step=0,
            yield_per_step=8,
            quant_per_step=4,
        )
        loss, parts = apc_contrast_loss(
            net, [], y_draws, quant_draws=q_draws, yield_weight=0.8, quant_weight=1.0
        )
        assert "apc_quant_loss" in parts
        loss.backward()
        opt.step()

    net.eval()

    def _val(r):
        return float(net([r])[0].value.detach())

    def _with_interv(row, key):
        return replace(
            row,
            context=replace(row.context, interventions={key: ContextAssertion("present")}),
        )

    # TAPBPR strict KO < KD < WT < OE on Class I, exact 0.0 cross-class delta on Class II
    t_ko = _val(_with_interv(base_i, "knockout:TAPBPR"))
    t_kd = _val(_with_interv(base_i, "knockdown:TAPBPR"))
    t_wt = _val(base_i)
    t_oe = _val(_with_interv(base_i, "overexpression:TAPBPR"))
    assert t_ko < t_kd < t_wt < t_oe

    ii_wt = _val(base_ii)
    ii_tapbpr_ko = _val(_with_interv(base_ii, "knockout:TAPBPR"))
    assert abs(ii_wt - ii_tapbpr_ko) == 0.0

    # Quantitative contrast separation: q_up (FC=+2.5) > q_dn (FC=-2.0)
    ctrl_ctx_up = replace(
        q_up.context,
        interventions={"drug:DMSO": ContextAssertion("present")},
        cytokines={"IFNG": Exposure("absent")},
    )
    ctrl_ctx_dn = replace(
        q_dn.context,
        interventions={"drug:DMSO": ContextAssertion("present")},
        cytokines={"IFNG": Exposure("absent")},
    )
    delta_up = _val(q_up) - _val(replace(q_up, context=ctrl_ctx_up))
    delta_dn = _val(q_dn) - _val(replace(q_dn, context=ctrl_ctx_dn))
    assert delta_up > delta_dn

    # Exact 0.0 invariance for both ic50 and half_life under APC/flank perturbations
    for ep, unit in (("ic50", "nM"), ("half_life", "min")):
        r0 = replace(base_i, endpoint=ep, unit=unit, context=Context())
        r1 = replace(base_i, endpoint=ep, unit=unit, context=q_up.context)
        assert abs(_val(r0) - _val(r1)) == 0.0


def test_anchor_v2_pmhc_residual_shared_v2_and_ensemble(tmp_path):
    """Verify anchor_v2, pMHC residual + Langmuir binding, shared_processing_v2, and ensemble."""
    from presto.inference.modular import ModularEnsemblePredictor
    from presto.models.modular import ModularConfig, ModularPresto

    # Legacy default config must omit all new v8 fields
    legacy_dict = ModularConfig().to_dict()
    assert "interaction_mode" not in legacy_dict
    assert "elution_pmhc_residual" not in legacy_dict
    assert "shared_processing_v2" not in legacy_dict

    torch.manual_seed(23)
    cfg = ModularConfig(
        endpoints=("ic50", "half_life", "elution_observation"),
        mhc_classes=("I", "II"),
        species=("human",),
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        source_context=True,
        elution_staging="affinity_downstream",
        elution_affinity_endpoint="ic50",
        elution_shared_processing=True,
        interaction_mode="anchor_v2",
        elution_pmhc_residual=True,
        shared_processing_v2=True,
        state_mode="harmonized",
        state_bottleneck=8,
        cytokines=("IFNG",),
        interventions=("knockout:TAP1", "knockout:TAPBPR", "transduction:HLA-DM"),
    )
    assert ModularConfig.from_dict(cfg.to_dict()) == cfg
    net = ModularPresto(cfg)
    assert not hasattr(net, "elution_affinity_weight")
    assert not hasattr(net, "elution_stability_weight")
    assert hasattr(net, "elution_pmhc_mlp")
    assert hasattr(net, "elution_binding_linear")
    assert hasattr(net, "elution_stability_class_weight")
    assert hasattr(net.elution_shared, "junction_conv3")

    rows = [
        replace(
            _example(
                "I",
                "SIINFEKL",
                source_n="AAAA",
                source_c="GGGG",
                cytokines={"IFNG": Exposure("present")},
                interventions={"knockout:TAP1": ContextAssertion("present")},
            ),
            observation_id="el_i1",
            value=1.0,
        ),
        replace(
            _example(
                "I",
                "YLQPRTFLL",
                source_n="DDDD",
                source_c="KKKK",
                interventions={"knockout:TAPBPR": ContextAssertion("present")},
            ),
            observation_id="el_i2",
            value=0.0,
        ),
        replace(
            _example("I", "GILGFVFTLAAA", source_n="EEEE", source_c="PPPP"),
            observation_id="el_i3",
            value=1.0,
        ),
        replace(
            _example(
                "II",
                "AAYAAYAAYAAYAAA",
                source_n="MMMM",
                source_c="NNNN",
                interventions={"transduction:HLA-DM": ContextAssertion("present")},
            ),
            observation_id="el_ii1",
            value=1.0,
        ),
        replace(
            _example("I", "SIINFEKL"),
            observation_id="ic50_i",
            endpoint="ic50",
            value=50.0,
            unit="nM",
            context=Context(),
        ),
        replace(
            _example("I", "SIINFEKL"),
            observation_id="hl_i",
            endpoint="half_life",
            value=120.0,
            unit="min",
            context=Context(),
        ),
    ]
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    sc = scales()
    sc["half_life"] = sc["ic50"]
    for _ in range(3):
        step = train_step(net, opt, rows, sc, amp_bfloat16=True)
        assert step["loss"] > 0

    # Verify pmhc_residual diagnostic is present and strictly invariant to APC/flank context
    net.eval()
    q_clean = replace(
        _example("I", "GILGFVFTLAAA", source_n=None, source_c=None),
        value=None,
        split="predict",
        context=Context(),
    )
    q_pert = replace(
        q_clean,
        context=Context(
            source_n="RRRRRR",
            source_c="YYYYYY",
            cytokines={"IFNG": Exposure("present")},
            interventions={"knockout:TAP1": ContextAssertion("present")},
        ),
    )
    with torch.no_grad():
        out_clean, out_pert = net([q_clean, q_pert])
    terms_c = out_clean.diagnostics["candidates"][0]["registers"][0]["elution_terms"]
    terms_p = out_pert.diagnostics["candidates"][0]["registers"][0]["elution_terms"]
    res_c = float(terms_c["pmhc_residual"])
    res_p = float(terms_p["pmhc_residual"])
    assert abs(res_c - res_p) < 1e-6

    # Exact 0.0 invariance for ic50 and half_life
    with torch.no_grad():
        for ep, unit in (("ic50", "nM"), ("half_life", "min")):
            a0 = float(net([replace(q_clean, endpoint=ep, unit=unit)])[0].value)
            a1 = float(net([replace(q_pert, endpoint=ep, unit=unit)])[0].value)
            assert abs(a0 - a1) == 0.0

    # Multi-checkpoint ModularEnsemblePredictor verification
    c1, c2 = tmp_path / "m1.pt", tmp_path / "m2.pt"
    net.trained_parameters = {name for name, _ in net.named_parameters()}
    save_checkpoint(c1, net, sc)
    with torch.no_grad():
        net.offsets["elution_observation"].bias.add_(0.4)
    save_checkpoint(c2, net, sc)
    ens = ModularEnsemblePredictor([c1, c2], device="cpu")
    pred_ens = ens.predict([q_clean])[0]
    assert pred_ens["ensemble_size"] == 2
    assert len(pred_ens["member_logits"]) == 2
    assert abs(pred_ens["logit"] - sum(pred_ens["member_logits"]) / 2.0) < 1e-6


def test_presto_omni_warmstart_multiassay_expression_celltype_tcell(tmp_path):
    """Warm-starting Presto-M into Presto-Omni preserves step-0 outputs and passes full coverage."""
    from presto.data.modular import Component, Observation, TargetScale
    from presto.data.tcell_assay import TCellTrack
    from presto.models.modular import ModularConfig, ModularPresto

    torch.manual_seed(29)
    base_cfg = ModularConfig(
        endpoints=("ic50", "half_life", "elution_observation"),
        mhc_classes=("I", "II"),
        species=("human",),
        d_model=24,
        n_layers=1,
        n_heads=4,
        d_ff=48,
        dropout=0.0,
        source_context=True,
        interaction_mode="anchor_v2",
        elution_staging="affinity_downstream",
        elution_affinity_endpoint="ic50",
        elution_pmhc_residual=True,
        elution_shared_processing=True,
        shared_processing_v2=True,
        state_mode="harmonized",
        state_bottleneck=8,
        cytokines=("IFNG",),
        interventions=("knockout:TAP1", "transduction:HLA-DM"),
    )
    base_model = ModularPresto(base_cfg)
    with torch.no_grad():
        base_model.elution_shared.supply_readout.weight.fill_(0.15)
        base_model.elution_shared.yield_theta.fill_(0.2)
        base_model.apm_state.encoder.weight.fill_(0.05)
        base_model.apm_state.supply.weight.fill_(0.05)
    base_scales = {
        "ic50": TargetScale(2.0, 1.0, 2),
        "half_life": TargetScale(1.5, 0.8, 2),
    }
    ckpt = tmp_path / "presto_m.pt"
    save_checkpoint(ckpt, base_model, base_scales)

    elispot = TCellTrack("cytokine_release", "ifng_release", "elispot")
    tetramer = TCellTrack("multimer_binding", "qualitative_binding", "multimer_tetramer")
    omni_cfg = replace(
        base_cfg,
        endpoints=(
            "ic50",
            "half_life",
            "koff",
            "kon",
            "td50",
            "qualitative_binding",
            "bulk_observation",
            "elution_observation",
            "tcell_response",
        ),
        enzymes=("chymotrypsin", "trypsin_p"),
        tcell_tracks=(elispot, tetramer),
        cell_types=("moDC", "ebv_lcl", "melanoma"),
        components=("PSMB8", "HLA-DMA"),
        component_activity_units={"PSMB8": "log1p_tpm", "HLA-DMA": "log1p_tpm"},
        cytokines=("IFNG", "TNF"),
        interventions=(
            "knockout:TAP1",
            "knockout:ERAP2",
            "transduction:HLA-DM",
            "drug:entinostat",
        ),
    )
    omni = ModularPresto(omni_cfg)
    initialize_from_checkpoint(omni, ckpt, allow_apc_vocab_expansion=True)

    # 1. Step-0 exact preservation of base Presto-M elution / ic50 / half_life predictions
    base_model.eval()
    omni.eval()
    q_i = replace(
        _example(
            "I",
            "GILGFVFTL",
            cytokines={"IFNG": Exposure("present")},
            interventions={"knockout:TAP1": ContextAssertion("present")},
        ),
        value=None,
        split="predict",
    )
    q_ii = replace(
        _example(
            "II",
            "AAYAAYAAYAAYAA",
            interventions={"transduction:HLA-DM": ContextAssertion("present")},
        ),
        value=None,
        split="predict",
    )
    with torch.no_grad():
        for q in (q_i, q_ii, replace(q_i, endpoint="ic50", unit="nM", context=Context())):
            v0 = float(base_model([q])[0].value)
            v1 = float(omni([q])[0].value)
            assert abs(v0 - v1) < 1e-6

    # 2. Pathway isolation of continuous log1p_tpm expression across MHC Class I / II
    with torch.no_grad():
        omni.apm_state.encoder.weight.uniform_(-0.2, 0.2)
        omni.apm_state.supply.weight.uniform_(-0.2, 0.2)
        # PSMB8 is Class-I-only -> changing PSMB8 log1p_tpm must have exact 0.0 delta on Class II
        q_ii_psmb8_lo = replace(
            q_ii,
            context=Context(
                components={"PSMB8": Component("unknown", activity=0.5, unit="log1p_tpm")}
            ),
        )
        q_ii_psmb8_hi = replace(
            q_ii,
            context=Context(
                components={"PSMB8": Component("unknown", activity=8.0, unit="log1p_tpm")}
            ),
        )
        assert (
            abs(float(omni([q_ii_psmb8_lo])[0].value) - float(omni([q_ii_psmb8_hi])[0].value))
            == 0.0
        )
        # HLA-DMA is Class-II-only -> changing log1p_tpm must have exact 0.0 delta on Class I
        q_i_dma_lo = replace(
            q_i,
            context=Context(
                components={"HLA-DMA": Component("unknown", activity=0.5, unit="log1p_tpm")}
            ),
        )
        q_i_dma_hi = replace(
            q_i,
            context=Context(
                components={"HLA-DMA": Component("unknown", activity=8.0, unit="log1p_tpm")}
            ),
        )
        assert abs(float(omni([q_i_dma_lo])[0].value) - float(omni([q_i_dma_hi])[0].value)) == 0.0

    # 3. Exact 0.0 invariance of all intrinsic pMHC endpoints to cell_type & log1p_tpm
    q_rich_ctx = Context(
        cell_type="moDC",
        components={
            "PSMB8": Component("unknown", activity=7.5, unit="log1p_tpm"),
            "HLA-DMA": Component("unknown", activity=6.8, unit="log1p_tpm"),
        },
        cytokines={"IFNG": Exposure("present"), "TNF": Exposure("present")},
        interventions={"knockout:ERAP2": ContextAssertion("present")},
    )
    with torch.no_grad():
        for ep, unit in (
            ("ic50", "nM"),
            ("half_life", "min"),
            ("koff", "1/s"),
            ("kon", "1/(M*s)"),
            ("td50", "C"),
            ("qualitative_binding", "binary"),
        ):
            base_q = replace(q_i, endpoint=ep, unit=unit, context=Context())
            pert_q = replace(q_i, endpoint=ep, unit=unit, context=q_rich_ctx)
            assert abs(float(omni([base_q])[0].value) - float(omni([pert_q])[0].value)) == 0.0

    # 4. Multi-task train_step with amp_bfloat16 across all 8 endpoints + TCellAssayLayer
    omni.train()
    omni_scales = {
        **base_scales,
        "koff": TargetScale(-3.0, 1.0, 2),
        "kon": TargetScale(5.0, 1.0, 2),
        "td50": TargetScale(55.0, 8.0, 2),
    }
    rows = [
        replace(
            _example(
                "I",
                "GILGFVFTL",
                source_n="KLMNPQ",
                source_c="STVWYA",
                cytokines={"IFNG": Exposure("present"), "TNF": Exposure("absent")},
                interventions={"knockout:ERAP2": ContextAssertion("present")},
            ),
            observation_id="el_modc",
            value=1.0,
            context=Context(
                source_n="KLMNPQ",
                source_c="STVWYA",
                cell_type="moDC",
                components={"PSMB8": Component("unknown", activity=7.2, unit="log1p_tpm")},
                cytokines={"IFNG": Exposure("present"), "TNF": Exposure("absent")},
                interventions={"knockout:ERAP2": ContextAssertion("present")},
            ),
        ),
        replace(
            _example("II", "AAYAAYAAYAAYAA"),
            observation_id="el_mel",
            value=0.0,
            context=Context(
                cell_type="melanoma",
                components={"HLA-DMA": Component("unknown", activity=1.1, unit="log1p_tpm")},
                interventions={"drug:entinostat": ContextAssertion("present")},
            ),
        ),
        replace(
            _example("I", "GILGFVFTL"),
            observation_id="ic50",
            endpoint="ic50",
            value=25.0,
            unit="nM",
            context=Context(),
        ),
        replace(
            _example("I", "GILGFVFTL"),
            observation_id="hl",
            endpoint="half_life",
            value=240.0,
            unit="min",
            context=Context(),
        ),
        replace(
            _example("I", "GILGFVFTL"),
            observation_id="koff",
            endpoint="koff",
            value=1e-4,
            unit="1/s",
            context=Context(),
        ),
        replace(
            _example("I", "GILGFVFTL"),
            observation_id="kon",
            endpoint="kon",
            value=1e5,
            unit="1/(M*s)",
            context=Context(),
        ),
        replace(
            _example("I", "GILGFVFTL"),
            observation_id="td50",
            endpoint="td50",
            value=62.5,
            unit="C",
            context=Context(),
        ),
        replace(
            _example("I", "GILGFVFTL"),
            observation_id="qb",
            endpoint="qualitative_binding",
            value=1.0,
            unit="binary",
            context=Context(),
        ),
        replace(
            _example("I", "GILGFVFTL"),
            observation_id="tc_el_hi",
            endpoint="tcell_response",
            value=1.0,
            unit="binary",
            tcell_track=elispot,
            tcell_intensity=2,
            context=Context(cell_type="moDC", coreceptor="CD8"),
        ),
        replace(
            _example("I", "SIINFEKL"),
            observation_id="tc_tet_neg",
            endpoint="tcell_response",
            value=0.0,
            unit="binary",
            tcell_track=tetramer,
            context=Context(cell_type="ebv_lcl", coreceptor="CD8"),
        ),
        Observation(
            observation_id="bulk_chymo",
            source="bj2017",
            split="train",
            group_id="P12345",
            endpoint="bulk_observation",
            peptide="GILGFVFTL",
            mhc=(),
            value=1.0,
            unit="binary",
            context=Context(source_n="KLMNPQ", source_c="STVWYA", enzyme="chymotrypsin"),
            provenance={"study_id": "28561026:chymotrypsin"},
        ),
    ]
    from presto.training.modular_sampling import stratum

    assert stratum(rows[-1], ["endpoint", "mhc_class", "study"]) == (
        "bulk_observation",
        "none",
        "28561026:chymotrypsin",
    )
    opt = torch.optim.AdamW(omni.parameters(), lr=2e-3)
    step = train_step(omni, opt, rows, omni_scales, amp_bfloat16=True)
    assert step["loss"] > 0
    assert "tcell_binary" in step["loss_parts"]
    assert "tcell_conditional_ordinal" in step["loss_parts"]
