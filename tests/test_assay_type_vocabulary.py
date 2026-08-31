"""Tests for the extended assay-type vocabulary and checkpoint growth."""

import pytest
import torch

from presto.data.collate import PrestoCollator, PrestoSample
from presto.data.vocab import BINDING_ASSAY_TYPES, IDX_TO_BINDING_ASSAY_TYPE
from presto.models.presto import Presto

MHC_A = "GSHSMRYFYT" * 4
MHC_B = "IQRTPKIQVY" * 4


class TestVocabulary:
    def test_appended_entries_did_not_move_existing_indices(self):
        """Append-only is what makes older checkpoints still meaningful."""
        assert BINDING_ASSAY_TYPES[:7] == [
            "unknown", "KD", "KD_PROXY_IC50", "KD_PROXY_EC50", "IC50", "EC50", "OTHER",
        ]

    def test_stability_and_kinetics_types_exist(self):
        for name in ("T_HALF", "TM", "KOFF", "KON"):
            assert name in BINDING_ASSAY_TYPES


class TestCategorization:
    @pytest.mark.parametrize("response,expected", [
        ("half life", "T_HALF"),
        ("50% dissociation temperature", "TM"),
        ("off rate", "KOFF"),
        ("on rate", "KON"),
        ("dissociation constant KD", "KD"),
        ("dissociation constant KD (~IC50)", "KD_PROXY_IC50"),
        ("half maximal inhibitory concentration (IC50)", "IC50"),
    ])
    def test_hitlist_response_strings_map_correctly(self, response, expected):
        assert PrestoCollator()._categorize_binding_assay_type(response) == expected

    def test_dissociation_temperature_is_not_read_as_a_kd(self):
        """It contains 'dissociation', so ordering in the categorizer matters."""
        assert PrestoCollator()._categorize_binding_assay_type(
            "50% dissociation temperature"
        ) == "TM"

    def test_stability_row_carries_its_own_type_through_collation(self):
        sample = PrestoSample(
            peptide="SIINFEKLA", mhc_a=MHC_A, mhc_b=MHC_B, mhc_class="I",
            t_half=4.0, stability_assay_type="half life",
            stability_assay_method="purified MHC/direct/radioactivity",
        )
        context = PrestoCollator()([sample]).binding_context
        label = IDX_TO_BINDING_ASSAY_TYPE[int(context["assay_type_idx"][0])]
        assert label == "T_HALF"


class TestVocabularyGrowthNoLongerMigrates:
    """Appending a vocabulary entry no longer rescues an older checkpoint.

    `Presto._grow_appended_embeddings` used to zero-pad a saved embedding table
    up to the current vocabulary size so a pre-growth checkpoint kept loading.
    It went with the rest of the checkpoint-compat layer: a migration that
    reshapes weights across a vocabulary change yields a model that loads
    cleanly and indexes differently.

    What still matters is the *ordering* rule that made growth safe in the
    first place -- entries are appended, so no existing index changes meaning.
    That is what this pins now.
    """

    def test_appending_does_not_disturb_existing_indices(self):
        from presto.data.vocab import BINDING_ASSAY_TYPES

        # The four entries appended on 2026-08-26. Every index before them must
        # still mean what it meant.
        appended = ["T_HALF", "TM", "KOFF", "KON"]
        for name in appended:
            assert name in BINDING_ASSAY_TYPES
        tail = BINDING_ASSAY_TYPES[-4:]
        assert sorted(tail) == sorted(appended), (
            "the four newest assay types are no longer at the end; appending is "
            "the rule that keeps existing indices meaningful"
        )

    def test_shrunken_table_is_rejected_rather_than_padded(self):
        """A stale table is now a load error, not a silent zero-pad."""
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        state = model.state_dict()
        key = next(k for k in state if k.endswith("assay_panel_embed.assay_type.weight"))
        rows, dim = state[key].shape
        state[key] = torch.zeros(rows - 4, dim)

        fresh = Presto(d_model=32, n_layers=2, n_heads=4)
        with pytest.raises(RuntimeError) as excinfo:
            fresh.load_state_dict(state, strict=False)
        assert "size mismatch" in str(excinfo.value)
