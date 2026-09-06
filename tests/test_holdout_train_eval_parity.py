"""The held-out pass must score the same function training optimizes.

The holdout forward assembled its own kwargs and omitted `provenance`. Every
validation row was therefore evaluated at the *default* cellular state, and the
in-vivo excision -> presentation edge contributed exactly zero -- so both
halves of gap 2 existed during training and vanished at evaluation.

Nothing caught it because both paths ran without error and produced plausible
numbers. The only symptom was that validation measured a different model.
"""

import inspect

import pytest

torch = pytest.importorskip("torch")

from presto.data.collate import PrestoCollator  # noqa: E402
from presto.data.loaders import ElutionRecord, PrestoDataset  # noqa: E402
from presto.models.presto import Presto  # noqa: E402

CLASS1_SEQ = "GSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPR"


@pytest.fixture
def batch():
    record = ElutionRecord(
        peptide="LLDGTATLRF",
        alleles=["HLA-A*02:01"],
        detected=True,
        stimulus="ifn_gamma",
        apm_perturbation="tap_ko",
    )
    dataset = PrestoDataset(
        elution_records=[record],
        mhc_sequences={"HLA-A*02:01": CLASS1_SEQ},
        strict_mhc_resolution=False,
    )
    return PrestoCollator()([dataset[0]])


class TestProvenanceChangesTheScore:
    def test_dropping_provenance_changes_the_prediction(self, batch):
        """If this ever stops being true, the test below is worthless."""
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        model.eval()
        kwargs = dict(
            pep_tok=batch.pep_tok,
            mhc_a_tok=batch.mhc_a_tok,
            mhc_b_tok=batch.mhc_b_tok,
            mhc_class=batch.mhc_class,
        )
        with torch.no_grad():
            with_prov = model(**kwargs, provenance=batch.provenance)
            without = model(**kwargs)
        assert not torch.allclose(with_prov["presentation_logit"], without["presentation_logit"])

    def test_gap2_edge_is_inert_without_provenance(self, batch):
        """Quantifies what the skew cost: the whole edge, at eval."""
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        model.eval()
        with torch.no_grad():
            without = model(
                pep_tok=batch.pep_tok,
                mhc_a_tok=batch.mhc_a_tok,
                mhc_b_tok=batch.mhc_b_tok,
                mhc_class=batch.mhc_class,
            )
        assert float(without["presentation_invivo_excision_term"].abs().sum()) == 0.0


class TestHoldoutForwardPassesProvenance:
    def test_source_passes_provenance(self):
        """Read the trainer's own holdout forward rather than a copy of it.

        Asserted against the source because the closure is defined inline in
        `train_iedb` and is not importable; a duplicated forward in this file
        would pass while the real one stayed broken -- which is precisely how
        the earlier gap-2 test managed to be green and wrong.
        """
        import presto.scripts.train_iedb as train_iedb

        from source_probe import region_between

        source = inspect.getsource(train_iedb)
        forward_src = region_between(
            source,
            "def _forward(model_ref, batch_ref):",
            "accumulators = collect_holdout_predictions",
            where="train_iedb",
        )
        assert "provenance=provenance" in forward_src, (
            "the held-out forward dropped provenance again; validation would "
            "score a different function than training optimizes"
        )

    def test_holdout_failure_is_recorded_not_swallowed(self):
        """A skipped pass must leave evidence, not just a printed line."""
        import presto.scripts.train_iedb as train_iedb

        source = inspect.getsource(train_iedb)
        assert "holdout_error.json" in source
        assert "Held-out metric pass FAILED" in source


class TestHoldoutScoresTheSelectedModel:
    """Training selects a model; evaluation must score that one.

    `--checkpoint` holds the best-validation epoch, and training then continues
    to `--epochs`. The held-out pass was handed the live in-memory model, so it
    reported the *last* epoch while `write_holdout_artifacts` stamped
    `best_val_loss` -- from a different epoch -- onto the same summary. On the
    2026-09-02 masked/s42 run the minimum was epoch 7 at val loss 0.5542 and
    training stopped at epoch 10 on 0.5623, past the minimum and rising.

    Nothing caught it because both models produce plausible numbers, and the
    summary carried a label suggesting the right one had been used. Asserted
    against the source for the same reason as the provenance test above: the
    eval block is inline in `run()` and is not importable.
    """

    @staticmethod
    def _eval_region():
        import inspect as _inspect

        import presto.scripts.train_iedb as train_iedb

        from source_probe import region_between

        return region_between(
            _inspect.getsource(train_iedb),
            "Score the model the run selected",
            "payload = write_holdout_artifacts",
            where="train_iedb",
        )

    def test_the_best_checkpoint_is_reloaded(self):
        region = self._eval_region()
        assert "load_model_from_checkpoint" in region, (
            "the held-out pass stopped reloading the best checkpoint; it would "
            "score whatever epoch training happened to stop on"
        )

    def test_the_reloaded_model_is_what_gets_scored(self):
        """Reloading is useless if the call site still passes the live model."""
        region = self._eval_region()
        assert "model=eval_model" in region, (
            "held-out predictions are being collected from the live model "
            "again, not the reloaded best checkpoint"
        )
        assert "model=model," not in region

    def test_a_failed_reload_never_scores_a_different_model(self):
        """A broken best checkpoint must fail rather than change the eval model."""
        region = self._eval_region()
        assert "final-epoch model" not in region
        assert "except Exception" not in region

    def test_each_selected_checkpoint_split_gets_a_full_overall_loss(self):
        """Per-task prediction metrics do not replace the requested overall loss."""
        import inspect

        import presto.scripts.train_iedb as train_iedb

        source = inspect.getsource(train_iedb)
        assert "heldout_loss, heldout_loss_terms = _call_evaluate_compat" in source
        assert "max_val_batches=0" in source
        assert '"overall_loss": float(heldout_loss)' in source
        assert '"loss_terms": heldout_loss_terms' in source
