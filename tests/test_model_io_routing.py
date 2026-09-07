"""Behavioral checks for the sequence-only encoder and shared I/O contract."""

import pytest
import torch
import torch.nn.functional as F

from presto.data.collate import PrestoCollator, PrestoSample
from presto.models.presto import Presto, _processing_species_idx_tensor
from presto.scripts.train_synthetic import (
    LOSS_TASK_SPECS,
    _compute_task_loss_vector,
    _get_batch_mask,
    _get_batch_target,
    _get_mil_channel,
    _resolve_task_prediction,
    _run_mil_forward,
    _slice_mil_channel,
    compute_loss,
)
from presto.training.holdout_eval import collect_holdout_predictions


def _model():
    torch.manual_seed(13)
    return Presto(d_model=32, n_layers=2, n_heads=4).eval()


def _batch(**fields):
    return PrestoCollator(max_pep_len=12, max_mhc_len=24)(
        [
            PrestoSample(
                peptide="SIINFEKL",
                mhc_a="ACDEFGHIKLMNPQRS",
                mhc_b="TVWYACDEFGHIKLMN",
                mhc_class="I",
                **fields,
            )
        ]
    )


def test_host_context_is_downstream_and_not_a_molecular_species_override():
    model = _model()
    with torch.no_grad():
        model.processing_species_embed.weight.normal_()
        kwargs = _batch().model_inputs()
        a = model(**dict(kwargs, species="human"))
        b = model(**dict(kwargs, species="mouse"))
    for key in (
        "pep_vec",
        "mhc_a_vec",
        "mhc_b_vec",
        "groove_vec",
        "mhc_species_probs",
        "binding_logit",
        "binding_affinity_score",
    ):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert not torch.allclose(a["processing_logit"], b["processing_logit"])
    for key in a["assays"]:
        torch.testing.assert_close(a["assays"][key], b["assays"][key], rtol=0, atol=0)
    unknown = _processing_species_idx_tensor(None, 1, torch.device("cpu"))
    human = _processing_species_idx_tensor("human", 1, torch.device("cpu"))
    assert unknown.item() != human.item()


def test_other_segment_presence_does_not_change_encoded_mhc_or_binding():
    model = _model()
    kwargs = _batch().model_inputs()
    with torch.no_grad():
        a = model(**kwargs)
        b = model(**dict(kwargs, flank_n_tok=torch.tensor([[4, 5, 6]])))
    for key in ("pep_vec", "mhc_a_vec", "mhc_b_vec", "binding_logit"):
        torch.testing.assert_close(a[key], b[key], rtol=1e-5, atol=1e-6)


def test_cd4_and_cd8_read_their_own_recognition_latents():
    model = _model()
    with torch.no_grad():
        out = model(**_batch().model_inputs())
        for lineage in ("cd4", "cd8"):
            expected = getattr(model, f"recognition_{lineage}_head")(
                out["latent_vecs"][f"recognition_{lineage}"]
            )
            torch.testing.assert_close(out[f"recognition_{lineage}_logit"], expected)
        assert not torch.allclose(
            model.recognition_cd4_head(out["latent_vecs"]["recognition_cd8"]),
            out["recognition_cd4_logit"],
        )


def test_tcell_panels_supervise_response_at_selected_column_and_dump_same_value():
    batch = _batch(tcell_label=1.0, tcell_assay_method="ELISPOT")
    spec = next(s for s in LOSS_TASK_SPECS if s.name == "tcell_assay_method")
    idx = batch.tcell_context["assay_method_idx"].item()
    panel = torch.arange(11, dtype=torch.float32).unsqueeze(0).requires_grad_()
    outputs = {"tcell_panel_logits": {"assay_method": panel}}
    target = _get_batch_target(batch, spec)
    assert target.item() == 1.0
    pred = _resolve_task_prediction(outputs, batch, spec)
    loss = _compute_task_loss_vector(spec, pred, target).sum()
    torch.testing.assert_close(loss, F.softplus(-panel[0, idx]))
    loss.backward()
    assert torch.nonzero(panel.grad[0]).flatten().tolist() == [idx]
    acc = collect_holdout_predictions(
        model=_model(),
        loader=[batch],
        device="cpu",
        specs=[spec],
        forward_fn=lambda model, moved: outputs,
        resolve_pred_fn=_resolve_task_prediction,
        get_target_fn=_get_batch_target,
        get_mask_fn=_get_batch_mask,
    )[spec.name]
    assert acc.rows()[0]["y_true"] == 1.0
    assert acc.rows()[0]["y_pred"] == float(idx)


def test_termini_survive_row_loss_and_capped_mil_forwards():
    batch = _batch(
        flank_n_is_terminus=True,
        flank_c_is_terminus=False,
        elution_label=1.0,
        mil_mhc_a_list=["ACDE", "FGHI"],
        mil_mhc_b_list=["KLMN", "PQRS"],
        mil_mhc_class_list=["I", "I"],
    )
    model = _model()
    seen = []
    handle = model.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs), with_kwargs=True
    )
    try:
        loss, _, _ = compute_loss(model, batch, "cpu")
        assert torch.isfinite(loss)
        for kwargs in seen:
            assert kwargs["flank_n_is_terminus"].all()
            assert not kwargs["flank_c_is_terminus"].any()
        channel = _get_mil_channel(batch.to("cpu"), "mil")
        capped = _slice_mil_channel(channel, torch.tensor([1]))
        _run_mil_forward(model, channel=capped, device="cpu", provenance=capped["provenance"])
        assert seen[-1]["flank_n_is_terminus"].tolist() == [True]
    finally:
        handle.remove()


def test_tcell_mil_carries_boundary_flags():
    batch = _batch(tcell_label=1.0, use_tcell_pathway_mil=True, flank_c_is_terminus=True)
    channel = _get_mil_channel(batch.to("cpu"), "tcell_mil")
    assert channel["flank_c_is_terminus"].tolist() == [True]


@pytest.mark.parametrize(
    "qual,pred,expected", [(1, 3.0, 0.0), (-1, 1.0, 0.0), (0, 3.0, 1.0), (1, 1.0, 1.0)]
)
def test_binding_panel_respects_measurement_qualifiers(qual, pred, expected):
    batch = _batch(bind_value=100.0, bind_qual=qual, bind_measurement_type="KD")

    class PanelModel(torch.nn.Module):
        def forward(self, **kwargs):
            return {"binding_assay_panel_assay_type": torch.full((1, 11), pred)}

    _, losses, _ = compute_loss(PanelModel(), batch, "cpu")
    assert float(losses["binding_assay_panel"]) == pytest.approx(expected)


def test_convenience_forwards_preserve_boundary_flags():
    model = _model()
    seen = []
    original = model.forward

    def record(**kwargs):
        seen.append(kwargs)
        return original(**kwargs)

    model.forward = record
    batch = _batch(flank_c_is_terminus=True)
    kwargs = batch.model_inputs()
    kwargs.pop("provenance")
    kwargs.pop("machinery")
    with torch.no_grad():
        model.forward_affinity_only(**kwargs)
        model.forward_presentation_only(**kwargs)
    assert len(seen) == 2
    assert all(k["flank_c_is_terminus"].item() for k in seen)


def test_presentation_predictor_and_tiling_carry_known_boundaries(monkeypatch):
    from presto.inference.predictor import Predictor

    model = _model()
    predictor = Predictor(model, device="cpu", auto_load_index_csv=False)
    monkeypatch.setattr(
        predictor, "_resolve_mhc_pair_sequences", lambda **kwargs: ("ACDEFGHIKLMN", "PQRSACDEFGHI")
    )
    seen = []
    handle = model.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs), with_kwargs=True
    )
    try:
        predictor.predict_presentation(peptide="SIINFEKL", flank_n="AC", flank_n_is_terminus=True)
        assert seen[-1]["flank_n_is_terminus"].tolist() == [True]
        assert seen[-1]["flank_c_is_terminus"].tolist() == [False]
        seen.clear()
        protein = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMN"
        predictor.predict_tiled_presentation(
            protein_sequence=protein,
            min_length=8,
            max_length=8,
            flank_size=3,
            batch_size=4,
        )
        n_flags = [v for k in seen for v in k["flank_n_is_terminus"].tolist()]
        c_flags = [v for k in seen for v in k["flank_c_is_terminus"].tolist()]
        assert n_flags == [start <= 3 for start in range(len(protein) - 7)]
        assert c_flags == [start + 8 + 3 >= len(protein) for start in range(len(protein) - 7)]
    finally:
        handle.remove()
