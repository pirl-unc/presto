"""Per-instance provenance must survive every MIL transformation.

`provenance` is per-instance, so anything that reindexes MIL instances has to
reindex it too. Two paths did not, with different symptoms:

- the instance cap sliced the inputs and left provenance full-length, so any
  run with `--max-mil-instances` and an oversized bag died in `torch.cat`;
- the contrastive branch dropped provenance entirely, running its synthetic
  negatives at the default cellular state while real bags ran at their true
  one -- a shortcut the condition embedding could learn instead of biology.
"""

import pytest

torch = pytest.importorskip("torch")

from presto.data.collate import PrestoCollator  # noqa: E402
from presto.data.loaders import ElutionRecord, PrestoDataset  # noqa: E402
from presto.models.presto import Presto  # noqa: E402
from presto.scripts.train_synthetic import (  # noqa: E402
    _build_contrastive_mil_channel,
    _get_mil_channel,
    _slice_mil_channel,
    compute_loss,
)

CLASS1_SEQ = "GSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPR"


def _multi_allele_batch(n_records=2):
    records = [
        ElutionRecord(
            peptide=f"SIINFEKL{'ACDEFGHIK'[i]}",
            alleles=["HLA-A*02:01", "HLA-B*07:02"],
            detected=True,
            inducer="basal",
            apm_perturbation="none",
        )
        for i in range(n_records)
    ]
    dataset = PrestoDataset(
        elution_records=records,
        mhc_sequences={"HLA-A*02:01": CLASS1_SEQ, "HLA-B*07:02": CLASS1_SEQ},
        strict_mhc_resolution=False,
    )
    return PrestoCollator()([dataset[i] for i in range(len(dataset))])


class TestInstanceCap:
    def test_capped_forward_runs(self):
        """Regression: this raised `Sizes of tensors must match`."""
        batch = _multi_allele_batch()
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        loss, _, _ = compute_loss(model, batch, "cpu", max_mil_instances=2)
        assert torch.isfinite(loss)

    def test_slice_reindexes_provenance_with_the_inputs(self):
        batch = _multi_allele_batch()
        channel = _get_mil_channel(batch, "mil")
        keep = torch.tensor([0, 1])
        sliced = _slice_mil_channel(channel, keep)
        assert sliced["pep_tok"].shape[0] == 2
        for name, tensor in sliced["provenance"].items():
            assert tensor.shape[0] == 2, f"provenance[{name}] was not sliced"

    def test_slice_does_not_mutate_the_original(self):
        """`dict(channel)` is shallow; provenance must not be edited in place."""
        batch = _multi_allele_batch()
        channel = _get_mil_channel(batch, "mil")
        before = {k: v.clone() for k, v in channel["provenance"].items()}
        _slice_mil_channel(channel, torch.tensor([0]))
        for name, tensor in before.items():
            assert torch.equal(channel["provenance"][name], tensor)


class TestContrastiveChannel:
    def test_provenance_follows_the_anchor(self):
        """The condition belongs to the peptide's sample, like pep_tok."""
        batch = _multi_allele_batch()
        channel = _get_mil_channel(batch, "mil")
        n_bags = int(channel["bag_label"].shape[0])
        if n_bags < 2:
            pytest.skip("need at least two bags to form a contrastive pair")
        contrastive = _build_contrastive_mil_channel(channel, [(0, 1, 1.0)])
        assert contrastive is not None
        assert "provenance" in contrastive
        n_instances = contrastive["pep_tok"].shape[0]
        for name, tensor in contrastive["provenance"].items():
            assert tensor.shape[0] == n_instances, f"provenance[{name}] misaligned"


class TestTCellChannel:
    def test_tcell_mil_provenance_is_carried_not_discarded(self):
        """The collator built this dict and then dropped it on the floor."""
        from presto.data.collate import PrestoBatch
        import dataclasses

        fields = {f.name for f in dataclasses.fields(PrestoBatch)}
        assert "tcell_mil_provenance" in fields

    def test_provenance_moves_with_the_batch(self):
        batch = _multi_allele_batch()
        moved = batch.to("cpu")
        assert isinstance(moved.tcell_mil_provenance, dict)
        assert isinstance(moved.mil_provenance, dict)
