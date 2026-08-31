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


class TestCheckpointGrowth:
    def test_older_checkpoint_embeddings_are_extended_not_rejected(self):
        """A grown vocabulary must not break loading a pre-growth checkpoint.

        Built with a residual mode that reads the factorized assay context: the
        `legacy` default consumes none of it, so those embeddings are no longer
        allocated and there would be no key to grow.
        """
        # Targets the *output-side* panel embedding. The input-side assay
        # embeddings this used to grow no longer exist -- they were the
        # contract violation.
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        state = model.state_dict()

        key = next(k for k in state if k.endswith("assay_panel_embed.assay_type.weight"))
        rows, dim = state[key].shape
        # Simulate a checkpoint saved before the four entries were appended.
        shrunk = torch.arange(float((rows - 4) * dim)).reshape(rows - 4, dim)
        state[key] = shrunk

        fresh = Presto(d_model=32, n_layers=2, n_heads=4)
        fresh.load_state_dict(state, strict=False)

        loaded = dict(fresh.named_parameters())[key]
        assert loaded.shape == (rows, dim)
        # Learned rows survive unchanged...
        assert torch.allclose(loaded[: rows - 4].detach(), shrunk)
        # ...and the appended rows keep their fresh init rather than being
        # zeroed, which would be a degenerate start for an untrained entry.
        assert torch.count_nonzero(loaded[rows - 4:].detach()) > 0
        assert torch.isfinite(loaded.detach()).all()
