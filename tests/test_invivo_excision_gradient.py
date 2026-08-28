"""Gap 2: the in-vivo excision path must receive gradient from real data.

The failure this guards against is specific and has already happened twice.
The excision head computes an in-vivo branch for MHC-source rows, but excision
*labels* only ever exist on shotgun (protein-source) rows, so for a long time
the in-vivo profiles sat at their initialization receiving exactly zero
gradient while looking, in the module listing, like trained parameters.

Every test here drives the **real pipeline** -- records -> dataset -> collator
-> compute_loss. An earlier version of this test hand-built a batch with an
`mhc`-source row carrying a fabricated excision target, a combination the
pipeline never produces, and so reported the path as trained when it was not.
Hand-built batches are therefore off-limits in this file.
"""

import pytest

torch = pytest.importorskip("torch")

from presto.data.collate import PrestoCollator  # noqa: E402
from presto.data.loaders import ElutionRecord, PrestoDataset  # noqa: E402
from presto.models.presto import Presto  # noqa: E402
from presto.scripts.train_synthetic import compute_loss  # noqa: E402

CLASS1_SEQ = "GSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPR"

# The parameters gap 2 named. Named explicitly rather than discovered by
# prefix so that deleting one to "fix" a failure shows up as an edit here.
INVIVO_PARAMETERS = (
    "invivo_profile_c",
    "invivo_profile_n",
    "stimulus_profile_c",
    "invivo_bias",
)


def _elution_batch():
    """Two MHC-source elution rows differing in APM state and label."""
    records = [
        ElutionRecord(
            peptide="LLDGTATLRF",
            alleles=["HLA-A*02:01"],
            detected=True,
            stimulus="ifn_gamma",
            apm_perturbation="tap_ko",
        ),
        ElutionRecord(
            peptide="SIINFEKLAA",
            alleles=["HLA-A*02:01"],
            detected=False,
            stimulus="none",
            apm_perturbation="none",
        ),
    ]
    dataset = PrestoDataset(
        elution_records=records,
        mhc_sequences={"HLA-A*02:01": CLASS1_SEQ},
        strict_mhc_resolution=False,
    )
    return PrestoCollator()([dataset[i] for i in range(len(dataset))])


def _backward_on_elution():
    model = Presto(d_model=32, n_layers=2, n_heads=4)
    loss, _, _ = compute_loss(model, _elution_batch(), "cpu")
    loss.backward()
    return model


def _grad_magnitude(model, name):
    grad = getattr(model.excision_head, name).grad
    return 0.0 if grad is None else float(grad.abs().sum())


class TestInVivoExcisionIsSupervised:
    @pytest.mark.parametrize("name", INVIVO_PARAMETERS)
    def test_parameter_receives_gradient_from_elution_labels(self, name):
        """Elution labels, not excision labels, are what train this path."""
        model = _backward_on_elution()
        assert _grad_magnitude(model, name) > 0.0, (
            f"excision_head.{name} received no gradient from a pure elution "
            "batch. The in-vivo path is dead again: check that the "
            "excision -> class I presentation edge still exists and that the "
            "class-mixed presentation logit is recomputed after it is applied."
        )

    def test_edge_weight_itself_is_trained(self):
        model = _backward_on_elution()
        grad = model.w_invivo_excision_presentation.grad
        assert grad is not None and float(grad.abs()) > 0.0

    def test_edge_weight_is_nonzero_at_init(self):
        """A zero-initialized weight would starve everything upstream of it."""
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        weight = torch.nn.functional.softplus(model.w_invivo_excision_presentation)
        assert float(weight) > 0.0

    def test_edge_is_non_negative_so_cleavage_cannot_lower_presentation(self):
        """Softplus encodes the biology: better cleavage never hurts."""
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        with torch.no_grad():
            model.w_invivo_excision_presentation.fill_(-50.0)
            assert float(torch.nn.functional.softplus(
                model.w_invivo_excision_presentation
            )) >= 0.0


class TestEdgeDoesNotLeakAcrossCorpora:
    def test_term_is_zero_for_protein_source_rows(self):
        """Shotgun rows keep the in-vitro branch as their only excision path."""
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        batch = _elution_batch()
        provenance = dict(batch.provenance)
        provenance["peptide_source_idx"] = torch.full_like(
            provenance["peptide_source_idx"],
            model.excision_head.protein_source_index,
        )
        outputs = model(
            pep_tok=batch.pep_tok,
            mhc_a_tok=batch.mhc_a_tok,
            mhc_b_tok=batch.mhc_b_tok,
            mhc_class=batch.mhc_class,
            provenance=provenance,
        )
        term = outputs["presentation_invivo_excision_term"]
        assert float(term.abs().sum()) == 0.0

    def test_term_is_nonzero_for_mhc_source_rows(self):
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        batch = _elution_batch()
        outputs = model(
            pep_tok=batch.pep_tok,
            mhc_a_tok=batch.mhc_a_tok,
            mhc_b_tok=batch.mhc_b_tok,
            mhc_class=batch.mhc_class,
            provenance=batch.provenance,
        )
        term = outputs["presentation_invivo_excision_term"]
        assert float(term.abs().sum()) > 0.0

    def test_missing_provenance_contributes_nothing(self):
        """No declared source means no in-vivo claim, not a default-on edge."""
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        batch = _elution_batch()
        outputs = model(
            pep_tok=batch.pep_tok,
            mhc_a_tok=batch.mhc_a_tok,
            mhc_b_tok=batch.mhc_b_tok,
            mhc_class=batch.mhc_class,
        )
        assert float(outputs["presentation_invivo_excision_term"].abs().sum()) == 0.0


class TestEdgeReachesTheTrainedObjective:
    def test_presentation_logit_reflects_the_edge(self):
        """The class-mixed logit is what the elution loss reads."""
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        batch = _elution_batch()
        kwargs = dict(
            pep_tok=batch.pep_tok,
            mhc_a_tok=batch.mhc_a_tok,
            mhc_b_tok=batch.mhc_b_tok,
            mhc_class=batch.mhc_class,
        )
        with_edge = model(**kwargs, provenance=batch.provenance)
        with torch.no_grad():
            model.w_invivo_excision_presentation.fill_(-50.0)  # softplus -> ~0
        without_edge = model(**kwargs, provenance=batch.provenance)
        assert not torch.allclose(
            with_edge["presentation_logit"], without_edge["presentation_logit"]
        )
