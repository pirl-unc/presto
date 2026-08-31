"""Assay descriptors are joint outputs, not input configuration.

`docs/assay_modeling_contract.md` is normative: "Presto must never consume
assay-selector metadata as predictive input", and it names assay type, method,
prep, geometry and readout explicitly.

The T-cell side already complied -- `AssayHeads.predict_panel` sweeps each
descriptor's embedding table to predict the response under *every* value, and
the label only routes supervision. The binding side did not: its descriptors
existed solely as an input-side context vector, which meant a prediction could
not be obtained without first declaring an assay.

These tests pin the corrected shape: every column predicted from peptide and
MHC alone, the label choosing only which column the loss reads.
"""

import pytest

torch = pytest.importorskip("torch")

from presto.data.vocab import (  # noqa: E402
    BINDING_ASSAY_GEOMETRY,
    BINDING_ASSAY_PREP,
    BINDING_ASSAY_READOUT,
    BINDING_ASSAY_TYPES,
)
from presto.models.presto import Presto  # noqa: E402

AXIS_SIZES = {
    "assay_type": len(BINDING_ASSAY_TYPES),
    "assay_prep": len(BINDING_ASSAY_PREP),
    "assay_geometry": len(BINDING_ASSAY_GEOMETRY),
    "assay_readout": len(BINDING_ASSAY_READOUT),
}


@pytest.fixture
def outputs():
    torch.manual_seed(0)
    model = Presto(d_model=32, n_layers=2, n_heads=4)
    model.eval()
    with torch.no_grad():
        return model(
            pep_tok=torch.randint(4, 24, (3, 10)),
            mhc_a_tok=torch.randint(4, 24, (3, 40)),
            mhc_b_tok=torch.randint(4, 24, (3, 40)),
            mhc_class="I",
        )


class TestPanelIsAJointOutput:
    @pytest.mark.parametrize("axis,size", sorted(AXIS_SIZES.items()))
    def test_one_prediction_per_assay_value(self, outputs, axis, size):
        panel = outputs[f"binding_assay_panel_{axis}"]
        assert panel.shape == (3, size)

    def test_panel_needs_no_assay_input(self):
        """The decisive property: a prediction without declaring an assay.

        The forward below passes peptide and MHC only. If the descriptors were
        input configuration this could not produce a per-assay answer at all.
        """
        torch.manual_seed(0)
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        model.eval()
        with torch.no_grad():
            out = model(
                pep_tok=torch.randint(4, 24, (2, 10)),
                mhc_a_tok=torch.randint(4, 24, (2, 40)),
                mhc_b_tok=torch.randint(4, 24, (2, 40)),
                mhc_class="I",
            )
        assert torch.isfinite(out["binding_assay_panel_assay_prep"]).all()

    def test_columns_are_distinguishable(self, outputs):
        """A panel whose columns are identical carries no assay information."""
        panel = outputs["binding_assay_panel_assay_type"]
        spread = panel.std(dim=1)
        assert float(spread.max()) > 0.0


class TestContractCompliance:
    def test_binding_context_is_not_passed_as_model_input(self):
        """The training loop must not feed assay metadata to the model.

        Regression guard: it was wired in, which made the forbidden pattern
        live before the contract was re-read.
        """
        import inspect

        from presto.scripts import train_synthetic

        source = inspect.getsource(train_synthetic.compute_loss)
        forward_start = source.index("outputs = model(")
        forward_end = source.index(")", source.index("provenance=", forward_start))
        forward_call = source[forward_start:forward_end]
        assert "binding_context=" not in forward_call, (
            "assay-selector metadata is being fed to the model as input, which "
            "docs/assay_modeling_contract.md forbids"
        )

    def test_assay_labels_may_still_route_supervision(self):
        """Allowed, and required for the panel to train at all."""
        import inspect

        from presto.scripts import train_synthetic

        source = inspect.getsource(train_synthetic.compute_loss)
        assert "binding_assay_panel" in source
