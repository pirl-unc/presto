"""Tests that stability censoring reaches the loss."""

import torch

from presto.data.collate import PrestoCollator, PrestoSample
from presto.scripts.train_synthetic import LOSS_TASK_NAME_TO_SPEC

MHC_A = "GSHSMRYFYT" * 4
MHC_B = "IQRTPKIQVY" * 4


def _sample(t_half, qual):
    return PrestoSample(
        peptide="SIINFEKLA",
        mhc_a=MHC_A,
        mhc_b=MHC_B,
        mhc_class="I",
        t_half=t_half,
        t_half_qual=qual,
        stability_assay_type="half life",
        stability_assay_method="purified MHC/direct/radioactivity",
    )


def test_stability_specs_are_censor_aware():
    for name in ("t_half", "tm"):
        spec = LOSS_TASK_NAME_TO_SPEC[name]
        assert spec.loss_type == "censor", f"{name} is still plain MSE"
        assert spec.qual_key == name


def test_qualifier_reaches_the_batch():
    batch = PrestoCollator()([_sample(2.0, 1), _sample(4.0, 0), _sample(1.0, -1)])
    quals = batch.target_quals["t_half"].reshape(-1).tolist()
    assert quals == [1, 0, -1]


def test_greater_than_measurement_is_not_penalized_above_the_bound():
    """A '>2h' row must not be trained toward exactly 2h."""
    from presto.training.losses import censor_aware_loss

    target = torch.tensor([2.0])
    qual = torch.tensor([1])
    over = censor_aware_loss(torch.tensor([5.0]), target, qual, reduction="none")
    under = censor_aware_loss(torch.tensor([0.5]), target, qual, reduction="none")
    assert over.item() == 0.0
    assert under.item() > 0.0
