"""The model is many-output: conditions are predicted, never supplied.

The DNA-model analogy is exact. Enformer-style trunks read sequence alone and
emit thousands of tracks, one per (cell type, assay); the track identity is an
output index, never an input feature. Presto reads peptide and MHC and emits
one output per assay configuration and per condition.

`docs/assay_modeling_contract.md` states the rule and names the forbidden
inputs. These tests assert it as an observable property -- passing a forbidden
input must not move the prediction -- rather than by reading the source, since
the source has been wrong twice.
"""

import pytest

torch = pytest.importorskip("torch")

from presto.models.presto import Presto  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = Presto(d_model=32, n_layers=2, n_heads=4)
    m.eval()
    return m


@pytest.fixture(scope="module")
def sequence_inputs():
    torch.manual_seed(1)
    return dict(
        pep_tok=torch.randint(4, 24, (3, 10)),
        mhc_a_tok=torch.randint(4, 24, (3, 40)),
        mhc_b_tok=torch.randint(4, 24, (3, 40)),
        mhc_class="I",
    )


#: Every key the contract names as a forbidden per-example input, grouped by
#: the argument that used to carry it.
FORBIDDEN_TCELL_CONTEXT = {
    "assay_method_idx": torch.tensor([1, 2, 1]),
    "assay_readout_idx": torch.tensor([1, 2, 1]),
    "apc_type_idx": torch.tensor([1, 2, 1]),
    "culture_context_idx": torch.tensor([1, 2, 1]),
    "stim_context_idx": torch.tensor([1, 2, 1]),
    "peptide_format_idx": torch.tensor([1, 2, 1]),
}
FORBIDDEN_BINDING_CONTEXT = {
    "assay_type_idx": torch.tensor([1, 2, 3]),
    "assay_prep_idx": torch.tensor([1, 2, 3]),
    "assay_geometry_idx": torch.tensor([1, 2, 3]),
    "assay_readout_idx": torch.tensor([1, 2, 3]),
}


class TestBindingIsAssayInvariant:
    def test_binding_prediction_ignores_the_assay_label(self, model, sequence_inputs):
        with torch.no_grad():
            base = model(**sequence_inputs)
            with_ctx = model(
                **sequence_inputs, binding_context=FORBIDDEN_BINDING_CONTEXT
            )
        for key in ("binding_logit", "binding_affinity_score", "binding_mixed_kd_log10"):
            assert torch.allclose(base[key], with_ctx[key]), (
                f"{key} moved when an assay label was attached; the prediction "
                "is conditioned on the assay rather than predicting it"
            )

    def test_the_input_side_context_vector_is_gone(self, model, sequence_inputs):
        """Not merely zeroed -- absent, so nothing can refill it."""
        with torch.no_grad():
            out = model(**sequence_inputs)
        assert "binding_factorized_assay_context_vec" not in out


class TestTCellIsContextInvariant:
    def test_head_remains_conditionable_and_that_is_the_open_half(
        self, model, sequence_inputs
    ):
        """Gap 7 is closed on the training path, not in the head.

        Records the honest state rather than asserting a compliance that does
        not hold. The binding side is structurally invariant -- the input path
        was deleted -- but `TCellAssayHead` still accepts the seven forbidden
        keys and still moves its prediction when given them. What changed is
        that no caller in the training or evaluation path supplies them, so
        every trained model is a context-free predictor.

        Closing the other half means deleting those arguments from the head,
        which changes its input dimensions and so needs a checkpoint
        migration. If this test starts failing because the prediction no
        longer moves, that work has been done and this should become an
        equality assertion.
        """
        with torch.no_grad():
            base = model(**sequence_inputs)
            with_ctx = model(**sequence_inputs, tcell_context=FORBIDDEN_TCELL_CONTEXT)
        assert not torch.allclose(base["tcell_logit"], with_ctx["tcell_logit"]), (
            "the T-cell head is now context-invariant -- gap 7 is fully "
            "closed; convert this to an equality assertion"
        )

    def test_training_loop_does_not_supply_tcell_context(self):
        import inspect

        from presto.scripts import train_synthetic

        source = inspect.getsource(train_synthetic.compute_loss)
        start = source.index("outputs = model(")
        end = source.index("provenance=", start)
        assert "tcell_context=" not in source[start:end]


class TestPanelsCoverEveryConfiguration:
    @pytest.mark.parametrize(
        "key,expected_axis",
        [
            ("binding_assay_panel_assay_type", "assay type"),
            ("binding_assay_panel_assay_prep", "sample prep"),
            ("binding_assay_panel_assay_geometry", "assay geometry"),
            ("binding_assay_panel_assay_readout", "readout chemistry"),
        ],
    )
    def test_one_output_track_per_value(self, model, sequence_inputs, key, expected_axis):
        with torch.no_grad():
            out = model(**sequence_inputs)
        assert key in out, f"no output track for {expected_axis}"
        assert out[key].shape[0] == 3
        assert out[key].shape[1] > 1, f"{expected_axis} has only one track"

    def test_tcell_panel_still_present(self, model, sequence_inputs):
        with torch.no_grad():
            out = model(**sequence_inputs)
        panel = out.get("tcell_panel_logits") or out.get("tcell_context_logits")
        assert panel, "the T-cell condition panel disappeared"
        assert "assay_method" in panel
