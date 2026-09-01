"""Decoys come in two families, and they must not be conflated.

The six original modes all corrupt one side of the pair -- scrambled or random
peptide, scrambled, random or absent MHC. Separating those from real pairs
needs no binding biology, only "does this look like a real peptide with a real
MHC". On a real run that produced elution AUPRC 1.0000 with *zero* overlap
between the score distributions: a sanity check reported as a solved task.

`allele_mismatch` is the hard version. The peptide is real and unmodified --
genuinely eluted, real source protein, correct length and terminal chemistry --
but paired with a different allele. The only separating signal is the motif.
"""

import pytest

from presto.scripts.train_iedb import (  # noqa: E402
    ALL_SYNTHETIC_MODES,
    EASY_SYNTHETIC_MODES,
    HARD_SYNTHETIC_MODES,
    augment_binding_records_with_synthetic_negatives,
)
from presto.data.loaders import BindingRecord  # noqa: E402

CLASS1_SEQ = "GSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPR"
REAL_PEPTIDES = {
    "HLA-A*02:01": ["SIINFEKLA", "KVFPYALIN", "YLLEMLWRL", "RTLNAWVKV"],
    "HLA-B*07:02": ["APRTLVYLL", "SPRWYFYYL", "TPRVTGGGA", "MPRSGFVCL"],
}


def _records():
    return [
        BindingRecord(
            peptide=peptide,
            mhc_allele=allele,
            value=20.0,
            measurement_type="half maximal inhibitory concentration (IC50)",
            mhc_class="I",
        )
        for allele, peptides in REAL_PEPTIDES.items()
        for peptide in peptides
    ]


def _augment(modes, ratio=1.0, records=None):
    return augment_binding_records_with_synthetic_negatives(
        records if records is not None else _records(),
        negative_ratio=ratio,
        mhc_sequences={a: CLASS1_SEQ for a in REAL_PEPTIDES},
        seed=0,
        modes=modes,
        weak_value_min_nM=20000.0,
        weak_value_max_nM=50000.0,
    )


class TestTaxonomy:
    def test_every_mode_is_classified(self):
        assert set(ALL_SYNTHETIC_MODES) == set(EASY_SYNTHETIC_MODES) | set(HARD_SYNTHETIC_MODES)

    def test_the_families_are_disjoint(self):
        assert not set(EASY_SYNTHETIC_MODES) & set(HARD_SYNTHETIC_MODES)

    def test_allele_mismatch_is_the_hard_one(self):
        assert HARD_SYNTHETIC_MODES == ("allele_mismatch",)


class TestAlleleMismatchDecoys:
    def test_peptides_are_real_and_unmodified(self):
        """The whole point: nothing about the peptide is fake."""
        augmented, _ = _augment(["allele_mismatch"])
        every_real_peptide = {p for ps in REAL_PEPTIDES.values() for p in ps}
        decoys = [r for r in augmented if str(getattr(r, "source", "")).endswith("allele_mismatch")]
        assert decoys, "no allele_mismatch decoys were generated"
        for decoy in decoys:
            assert decoy.peptide in every_real_peptide, (
                f"{decoy.peptide!r} is not a real corpus peptide; the decoy was "
                "corrupted and is no longer hard"
            )

    def test_the_peptide_comes_from_a_different_allele(self):
        augmented, _ = _augment(["allele_mismatch"])
        for decoy in augmented:
            if not str(getattr(decoy, "source", "")).endswith("allele_mismatch"):
                continue
            own = REAL_PEPTIDES.get(decoy.mhc_allele, [])
            assert decoy.peptide not in own, (
                "decoy peptide is a real ligand of the very allele it is "
                "labelled negative for -- that is a mislabelled positive"
            )

    def test_length_distribution_is_preserved(self):
        """Scrambling preserves length; random generation need not. Real
        peptides preserve it by construction, which is why they are harder."""
        augmented, _ = _augment(["allele_mismatch"])
        real_lengths = {len(p) for ps in REAL_PEPTIDES.values() for p in ps}
        for decoy in augmented:
            if str(getattr(decoy, "source", "")).endswith("allele_mismatch"):
                assert len(decoy.peptide) in real_lengths

    def test_single_allele_corpus_does_not_emit_mislabelled_positives(self):
        """With one allele there is no mismatch available.

        Emitting a same-allele "negative" would be a real ligand labelled
        non-binding, which is worse than no decoy at all.
        """
        single = [
            BindingRecord(
                peptide=p,
                mhc_allele="HLA-A*02:01",
                value=20.0,
                measurement_type="half maximal inhibitory concentration (IC50)",
                mhc_class="I",
            )
            for p in REAL_PEPTIDES["HLA-A*02:01"]
        ]
        augmented, _ = _augment(["allele_mismatch"], records=single)
        for record in augmented:
            source = str(getattr(record, "source", ""))
            if source.endswith("allele_mismatch"):
                pytest.fail(
                    "emitted an allele_mismatch decoy from a single-allele "
                    "corpus; its peptide must be a real ligand of that allele"
                )


class TestEasyModesStillWork:
    @pytest.mark.parametrize("mode", EASY_SYNTHETIC_MODES)
    def test_each_easy_mode_generates(self, mode):
        augmented, stats = _augment([mode])
        assert len(augmented) > len(_records())
        assert stats["added"] > 0
