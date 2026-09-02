"""Training-time corruption, and the invariants it must not violate.

Two augmentations, both about the model seeing at training time the conditions
it will meet at inference.

**Flank dropout** replaces both flanks with `?`. A caller with a peptide and an
allele and no source protein is the common inference case, and 0.70% of corpus
rows were never mapped either. A model trained only on flanked rows leans on
them and then degrades unpredictably without them.

**Residue dropout** turns scattered residues into `X`. That grounds the token
in context rather than leaving its meaning to the 45,992 initiator-residue rows
that happen to carry one naturally.
"""

import random

import pytest

from presto.data.sequence_augmentation import (  # noqa: E402
    UNKNOWN_FLANK,
    UNKNOWN_RESIDUE,
    AugmentationConfig,
    augment_sample_sequences,
    corrupt_residues,
)
from presto.data.vocab import ENCODABLE_RESIDUES, FLANK_MARKERS  # noqa: E402


class TestConfig:
    def test_inactive_by_default(self):
        """Validation and inference must never be augmented by accident."""
        assert AugmentationConfig().is_active is False

    @pytest.mark.parametrize(
        "field",
        [
            "flank_dropout_rate",
            "flank_residue_dropout_rate",
            "peptide_residue_dropout_rate",
        ],
    )
    def test_any_nonzero_rate_activates(self, field):
        assert AugmentationConfig(**{field: 0.1}).is_active is True

    @pytest.mark.parametrize("rate", [-0.1, 1.5])
    def test_rates_outside_zero_to_one_are_rejected(self, rate):
        with pytest.raises(ValueError, match="must be in"):
            AugmentationConfig(flank_dropout_rate=rate)


class TestResidueDropout:
    def test_residues_become_x(self):
        out = corrupt_residues("ACDEFGHIK", 1.0, random.Random(0))
        assert out == UNKNOWN_RESIDUE * 9

    def test_zero_rate_is_a_no_op(self):
        assert corrupt_residues("ACDEFGHIK", 0.0, random.Random(0)) == "ACDEFGHIK"

    def test_flank_markers_are_never_corrupted(self):
        """A marker already describes an absence. Turning `^` into `X` would
        assert a residue exists where the entire point is that none does."""
        out = corrupt_residues("^^^ACD", 1.0, random.Random(0))
        assert out.startswith("^^^"), out
        assert out == "^^^" + UNKNOWN_RESIDUE * 3

    @pytest.mark.parametrize("marker", sorted(FLANK_MARKERS))
    def test_each_marker_survives(self, marker):
        assert corrupt_residues(marker, 1.0, random.Random(0)) == marker

    def test_only_encodable_residues_are_targeted(self):
        for character in corrupt_residues("ACDEFGHIK", 0.5, random.Random(1)):
            assert character in ENCODABLE_RESIDUES

    def test_length_is_preserved(self):
        """Corruption replaces, never deletes -- positions carry subsite
        meaning in the excision window."""
        original = "ACDEFGHIKLMNPQR"
        assert len(corrupt_residues(original, 0.5, random.Random(2))) == len(original)


class _Sample:
    def __init__(self):
        self.peptide = "SIINFEKL"
        self.flank_n = "ACDEFGHIKLMNPQR"
        self.flank_c = "WYVTSRQPNMLKIHG"
        self.flank_n_is_terminus = True
        self.flank_c_is_terminus = True


class TestFlankDropout:
    def test_both_flanks_go_together(self):
        """Dropping one and keeping the other would teach a correlation that
        does not exist: a peptide whose source protein is unknown has neither
        flank, not one."""
        sample = _Sample()
        augment_sample_sequences(
            sample, AugmentationConfig(flank_dropout_rate=1.0), random.Random(0)
        )
        assert sample.flank_n == UNKNOWN_FLANK
        assert sample.flank_c == UNKNOWN_FLANK

    def test_dropping_the_flank_also_drops_the_terminus_claim(self):
        """The terminus came from the protein mapping, and the mapping is
        exactly what the dropout is pretending not to have. Keeping the flag
        would leak it back."""
        sample = _Sample()
        augment_sample_sequences(
            sample, AugmentationConfig(flank_dropout_rate=1.0), random.Random(0)
        )
        assert sample.flank_n_is_terminus is False
        assert sample.flank_c_is_terminus is False

    def test_the_peptide_is_untouched_by_flank_dropout(self):
        sample = _Sample()
        augment_sample_sequences(
            sample, AugmentationConfig(flank_dropout_rate=1.0), random.Random(0)
        )
        assert sample.peptide == "SIINFEKL"

    def test_a_rate_of_zero_changes_nothing(self):
        sample = _Sample()
        augment_sample_sequences(sample, AugmentationConfig(), random.Random(0))
        assert sample.flank_n == "ACDEFGHIKLMNPQR"
        assert sample.flank_n_is_terminus is True

    def test_dropout_hits_a_subset_not_everything(self):
        """The point is a mix within each batch: some rows with flanks, some
        without, so the model learns both regimes rather than switching."""
        rng = random.Random(0)
        config = AugmentationConfig(flank_dropout_rate=0.5)
        samples = [_Sample() for _ in range(200)]
        for sample in samples:
            augment_sample_sequences(sample, config, rng)
        dropped = sum(1 for s in samples if s.flank_n == UNKNOWN_FLANK)
        assert 0 < dropped < len(samples), dropped
        assert 60 <= dropped <= 140, f"expected roughly half, got {dropped}"


class TestCollatorIntegration:
    def test_the_default_collator_does_not_augment(self):
        from presto.data.collate import PrestoCollator

        assert PrestoCollator().augmentation.is_active is False

    def test_an_augmenting_collator_corrupts(self):
        from presto.data.collate import PrestoCollator, PrestoSample

        samples = [
            PrestoSample(
                peptide="SIINFEKL",
                flank_n="ACDEFGHIKLMNPQR",
                flank_c="WYVTSRQPNMLKIHG",
                mhc_a="A" * 40,
                mhc_class="I",
            )
            for _ in range(40)
        ]
        collator = PrestoCollator(augmentation=AugmentationConfig(flank_dropout_rate=1.0))
        collator(samples)
        assert all(s.flank_n == UNKNOWN_FLANK for s in samples)
