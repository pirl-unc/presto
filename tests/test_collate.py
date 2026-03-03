"""Tests for data collation module."""

import pytest
import torch

from presto.data.collate import (
    PrestoSample,
    PrestoBatch,
    PrestoCollator,
    collate_dict_batch,
)
from presto.data.tokenizer import Tokenizer


class TestPrestoSample:
    """Tests for PrestoSample dataclass."""

    def test_minimal_sample(self):
        """Create sample with only required field."""
        sample = PrestoSample(peptide="SIINFEKL")
        assert sample.peptide == "SIINFEKL"
        assert sample.mhc_a == ""
        assert sample.mhc_class == "I"
        assert sample.tcr_a is None

    def test_full_sample(self):
        """Create sample with all fields."""
        sample = PrestoSample(
            peptide="SIINFEKL",
            flank_n="AAA",
            flank_c="GGG",
            mhc_a="MAVMAPRTL",
            mhc_b="BETA2M",
            mhc_class="I",
            tcr_a="CAVRD",
            tcr_b="CASSIR",
            bind_value=4.5,
            bind_qual=0,
            kon=1e5,
            koff=1e-3,
            t_half=3600.0,
            tm=55.0,
            tcell_label=1.0,
            elution_label=1.0,
            chain_type="TRA",
            species="human",
            phenotype="CD8_T",
            sample_id="sample_001",
        )
        assert sample.bind_value == 4.5
        assert sample.tcell_label == 1.0

    def test_sample_bind_qual_values(self):
        """bind_qual can be -1 (less than), 0 (equal), 1 (greater than)."""
        sample_lt = PrestoSample(peptide="AAA", bind_value=5.0, bind_qual=-1)
        sample_eq = PrestoSample(peptide="AAA", bind_value=5.0, bind_qual=0)
        sample_gt = PrestoSample(peptide="AAA", bind_value=5.0, bind_qual=1)

        assert sample_lt.bind_qual == -1
        assert sample_eq.bind_qual == 0
        assert sample_gt.bind_qual == 1


class TestPrestoBatch:
    """Tests for PrestoBatch dataclass."""

    def test_batch_to_device(self):
        """Test moving batch to device."""
        batch = PrestoBatch(
            pep_tok=torch.zeros(2, 10, dtype=torch.long),
            mhc_a_tok=torch.zeros(2, 100, dtype=torch.long),
            mhc_b_tok=torch.zeros(2, 50, dtype=torch.long),
            mhc_class=["I", "I"],
            bind_target=torch.tensor([[4.5], [5.0]]),
            bind_mask=torch.tensor([1.0, 1.0]),
            targets={"binding": torch.tensor([[4.5], [5.0]])},
            target_masks={"binding": torch.tensor([1.0, 1.0])},
            target_quals={"binding": torch.tensor([[0], [0]], dtype=torch.long)},
            sample_ids=["s1", "s2"],
        )

        # Move to same device (cpu->cpu)
        moved = batch.to("cpu")
        assert moved.pep_tok.device.type == "cpu"
        assert moved.bind_target.device.type == "cpu"
        assert moved.targets["binding"].device.type == "cpu"
        assert moved.target_masks["binding"].device.type == "cpu"
        assert moved.target_quals["binding"].device.type == "cpu"
        assert moved.mhc_class == ["I", "I"]
        assert moved.sample_ids == ["s1", "s2"]

    def test_batch_handles_none(self):
        """Batch handles None optional fields when moving."""
        batch = PrestoBatch(
            pep_tok=torch.zeros(2, 10, dtype=torch.long),
            mhc_a_tok=torch.zeros(2, 100, dtype=torch.long),
            mhc_b_tok=torch.zeros(2, 50, dtype=torch.long),
            mhc_class=["I", "I"],
            tcr_a_tok=None,  # None optional
        )

        moved = batch.to("cpu")
        assert moved.tcr_a_tok is None


class TestPrestoCollator:
    """Tests for PrestoCollator class."""

    @pytest.fixture
    def collator(self):
        return PrestoCollator()

    @pytest.fixture
    def simple_samples(self):
        return [
            PrestoSample(peptide="SIINFEKL", mhc_a="MAVMAPRTL", mhc_b="MKM"),
            PrestoSample(peptide="GILGFVFTL", mhc_a="MAVMAPRTL", mhc_b="MKM"),
        ]

    def test_collator_basic(self, collator, simple_samples):
        """Collate basic samples into batch."""
        batch = collator(simple_samples)

        assert isinstance(batch, PrestoBatch)
        assert batch.pep_tok.shape[0] == 2  # batch size
        assert batch.mhc_a_tok.shape[0] == 2
        assert len(batch.mhc_class) == 2

    def test_collator_pads_peptides(self, collator):
        """Peptides of different lengths are padded."""
        samples = [
            PrestoSample(peptide="AAA", mhc_a=""),
            PrestoSample(peptide="AAAAAAAAAA", mhc_a=""),
        ]
        batch = collator(samples)

        # Both should have same length after padding
        assert batch.pep_tok.shape[1] == batch.pep_tok.shape[1]

    def test_collator_returns_lengths(self, collator, simple_samples):
        """Collator returns peptide lengths."""
        batch = collator(simple_samples)
        assert batch.pep_lengths is not None
        assert batch.pep_lengths.shape[0] == 2

    def test_collator_with_flanks(self, collator):
        """Collate samples with processing flanks."""
        samples = [
            PrestoSample(peptide="SIINFEKL", flank_n="AAA", flank_c="GGG"),
            PrestoSample(peptide="GILGFVFTL", flank_n="BBB", flank_c="HHH"),
        ]
        batch = collator(samples)

        assert batch.flank_n_tok is not None
        assert batch.flank_c_tok is not None
        assert batch.flank_n_tok.shape[0] == 2

    def test_collator_flanks_none_when_absent(self, collator, simple_samples):
        """Flanks are None when no samples have them."""
        batch = collator(simple_samples)
        assert batch.flank_n_tok is None
        assert batch.flank_c_tok is None

    def test_collator_with_tcr(self, collator):
        """Collate samples with TCR sequences."""
        samples = [
            PrestoSample(
                peptide="SIINFEKL",
                mhc_a="MAVMAPRTL",
                tcr_a="CAVRDSSYKLIF",
                tcr_b="CASSIRSSYEQYF",
            ),
            PrestoSample(
                peptide="GILGFVFTL",
                mhc_a="MAVMAPRTL",
                tcr_a="CAVMDSSYKLIF",
                tcr_b="CASSLGSSYEQYF",
            ),
        ]
        batch = collator(samples)

        assert batch.tcr_a_tok is not None
        assert batch.tcr_b_tok is not None
        assert batch.tcr_a_tok.shape[0] == 2

    def test_collator_tcr_none_when_absent(self, collator, simple_samples):
        """TCR tokens are None when no samples have TCR."""
        batch = collator(simple_samples)
        assert batch.tcr_a_tok is None
        assert batch.tcr_b_tok is None

    def test_collator_with_binding_labels(self, collator):
        """Collate samples with binding labels."""
        samples = [
            PrestoSample(peptide="AAA", mhc_a="", bind_value=4.5, bind_qual=0),
            PrestoSample(peptide="BBB", mhc_a="", bind_value=6.0, bind_qual=-1),
        ]
        batch = collator(samples)

        assert batch.bind_target is not None
        assert batch.bind_qual is not None
        assert batch.bind_mask is not None
        assert "binding" in batch.targets
        assert "binding" in batch.target_masks
        assert "binding" in batch.target_quals
        assert batch.bind_target.shape == (2, 1)
        assert torch.all(batch.bind_mask == 1.0)

    def test_collator_splits_binding_targets_by_measurement_type(self, collator):
        samples = [
            PrestoSample(
                peptide="AAA",
                mhc_a="",
                bind_value=2.0,
                bind_qual=0,
                bind_measurement_type="KD",
            ),
            PrestoSample(
                peptide="BBB",
                mhc_a="",
                bind_value=3.0,
                bind_qual=-1,
                bind_measurement_type="IC50",
            ),
            PrestoSample(
                peptide="CCC",
                mhc_a="",
                bind_value=4.0,
                bind_qual=1,
                bind_measurement_type="EC50",
            ),
            PrestoSample(
                peptide="DDD",
                mhc_a="",
                bind_value=5.0,
                bind_qual=0,
                bind_measurement_type=None,
            ),
        ]
        batch = collator(samples)

        assert "binding_kd" in batch.targets
        assert "binding_ic50" in batch.targets
        assert "binding_ec50" in batch.targets
        assert "binding_unknown" in batch.targets
        assert batch.target_masks["binding_kd"].tolist() == [1.0, 0.0, 0.0, 0.0]
        assert batch.target_masks["binding_ic50"].tolist() == [0.0, 1.0, 0.0, 0.0]
        assert batch.target_masks["binding_ec50"].tolist() == [0.0, 0.0, 1.0, 0.0]
        assert batch.target_masks["binding_unknown"].tolist() == [0.0, 0.0, 0.0, 1.0]
        assert "binding_kd" in batch.target_quals
        assert "binding_ic50" in batch.target_quals
        assert "binding_ec50" in batch.target_quals
        assert "binding_unknown" in batch.target_quals

    def test_collator_bind_mask_partial(self, collator):
        """Bind mask is 0 for samples without binding labels."""
        samples = [
            PrestoSample(peptide="AAA", mhc_a="", bind_value=4.5),
            PrestoSample(peptide="BBB", mhc_a=""),  # No binding label
        ]
        batch = collator(samples)

        assert batch.bind_mask[0] == 1.0
        assert batch.bind_mask[1] == 0.0

    def test_collator_with_tcell_labels(self, collator):
        """Collate samples with T-cell labels."""
        samples = [
            PrestoSample(peptide="AAA", mhc_a="", tcell_label=1.0),
            PrestoSample(peptide="BBB", mhc_a="", tcell_label=0.0),
        ]
        batch = collator(samples)

        assert batch.tcell_label is not None
        assert batch.tcell_mask is not None
        assert batch.tcell_label.shape == (2,)

    def test_collator_emits_tcell_assay_context_targets(self, collator):
        samples = [
            PrestoSample(
                peptide="AAA",
                mhc_a="",
                tcell_label=1.0,
                tcell_assay_method="ELISPOT",
                tcell_assay_readout="IFNg release",
                tcell_apc_name="dendritic cell",
                tcell_effector_culture="Direct Ex Vivo",
                tcell_apc_culture="Direct Ex Vivo",
                tcell_in_vitro_process="Primary induction in vitro",
                tcell_in_vitro_responder="PBMC",
                tcell_in_vitro_stimulator="T2 cell (B cell)",
            ),
            PrestoSample(peptide="BBB", mhc_a="", tcell_label=None),
        ]
        batch = collator(samples)

        assert "assay_method_idx" in batch.tcell_context
        assert "assay_readout_idx" in batch.tcell_context
        assert "apc_type_idx" in batch.tcell_context
        assert "culture_context_idx" in batch.tcell_context
        assert "stim_context_idx" in batch.tcell_context

        assert "tcell_assay_method" in batch.targets
        assert "tcell_assay_readout" in batch.targets
        assert "tcell_apc_type" in batch.targets
        assert "tcell_culture_context" in batch.targets
        assert "tcell_stim_context" in batch.targets
        assert batch.target_masks["tcell_assay_method"].tolist() == [1.0, 0.0]

    def test_collator_with_elution_labels(self, collator):
        """Collate samples with elution labels."""
        samples = [
            PrestoSample(peptide="AAA", mhc_a="", elution_label=1.0),
            PrestoSample(peptide="BBB", mhc_a="", elution_label=0.0),
        ]
        batch = collator(samples)

        assert batch.elution_label is not None
        assert batch.elution_mask is not None

    def test_collator_builds_multi_allele_mil_tensors(self, collator):
        samples = [
            PrestoSample(
                peptide="SIINFEKL",
                mhc_a="HLAASEQ",
                mhc_b="MKMSEQ",
                mhc_class="I",
                elution_label=1.0,
                species="human",
                sample_id="elut_a",
                mil_mhc_a_list=["HLAASEQ", "HLABSEQ"],
                mil_mhc_b_list=["MKMSEQ", "MKMSEQ"],
                mil_mhc_class_list=["I", "I"],
                mil_species_list=["human", "human"],
            ),
            PrestoSample(
                peptide="GILGFVFTL",
                mhc_a="HLACSEQ",
                mhc_b="MKMSEQ",
                mhc_class="I",
                elution_label=0.0,
                species="human",
                sample_id="elut_b",
                mil_mhc_a_list=["HLACSEQ"],
                mil_mhc_b_list=["MKMSEQ"],
                mil_mhc_class_list=["I"],
                mil_species_list=["human"],
            ),
        ]

        batch = collator(samples)

        assert batch.mil_pep_tok is not None
        assert batch.mil_mhc_a_tok is not None
        assert batch.mil_mhc_b_tok is not None
        assert batch.mil_instance_to_bag is not None
        assert batch.mil_bag_label is not None
        assert batch.mil_pep_tok.shape[0] == 3
        assert batch.mil_instance_to_bag.tolist() == [0, 0, 1]
        assert batch.mil_bag_label.tolist() == [1.0, 0.0]
        assert batch.mil_bag_sample_ids == ["elut_a", "elut_b"]

    def test_collator_sample_ids(self, collator):
        """Collator preserves sample IDs."""
        samples = [
            PrestoSample(peptide="AAA", sample_id="id_1"),
            PrestoSample(peptide="BBB", sample_id="id_2"),
        ]
        batch = collator(samples)

        assert batch.sample_ids == ["id_1", "id_2"]

    def test_collator_with_stability_targets_and_masks(self, collator):
        """Stability targets are emitted with per-target masks."""
        samples = [
            PrestoSample(peptide="AAA", mhc_a="", t_half=1.0, tm=50.0),  # 1 hour
            PrestoSample(peptide="BBB", mhc_a="", t_half=None, tm=65.0),
        ]

        batch = collator(samples)

        assert batch.t_half_target is not None
        assert batch.tm_target is not None
        assert batch.t_half_mask is not None
        assert batch.tm_mask is not None
        assert "t_half" in batch.targets
        assert "tm" in batch.targets
        assert "t_half" in batch.target_masks
        assert "tm" in batch.target_masks

        # t_half target is log10(minutes), so 1 hour => log10(60)
        expected_t_half = torch.log10(torch.tensor(60.0))
        assert torch.isclose(batch.t_half_target[0, 0], expected_t_half)
        assert batch.t_half_mask.tolist() == [1.0, 0.0]

        # Tm target is normalized around 50C with std=15C
        assert torch.isclose(batch.tm_target[0, 0], torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(batch.tm_target[1, 0], torch.tensor(1.0), atol=1e-6)
        assert batch.tm_mask.tolist() == [1.0, 1.0]

    def test_collator_custom_max_lengths(self):
        """Collator respects custom max lengths."""
        collator = PrestoCollator(
            max_pep_len=15,
            max_mhc_len=200,
            max_tcr_len=100,
        )
        samples = [
            PrestoSample(
                peptide="A" * 20,  # longer than max
                mhc_a="M" * 300,  # longer than max
            ),
        ]
        batch = collator(samples)

        assert batch.pep_tok.shape[1] == 15
        assert batch.mhc_a_tok.shape[1] == 200


class TestCollateDictBatch:
    """Tests for collate_dict_batch function."""

    def test_collate_dict_basic(self):
        """Collate list of dicts into batch dict."""
        batch = [
            {"peptide": "SIINFEKL", "mhc_a": "MAVMAPRTL"},
            {"peptide": "GILGFVFTL", "mhc_a": "MAVMAPRTL"},
        ]
        result = collate_dict_batch(batch)

        assert "pep_tok" in result
        assert "mhc_a_tok" in result
        assert result["pep_tok"].shape[0] == 2

    def test_collate_dict_with_mhc_class(self):
        """Collate preserves MHC class."""
        batch = [
            {"peptide": "AAA", "mhc_class": "I"},
            {"peptide": "BBB", "mhc_class": "II"},
        ]
        result = collate_dict_batch(batch)

        assert result["mhc_class"] == ["I", "II"]

    def test_collate_dict_default_mhc_class(self):
        """Default MHC class is 'I'."""
        batch = [{"peptide": "AAA"}]
        result = collate_dict_batch(batch)

        assert result["mhc_class"] == ["I"]

    def test_collate_dict_with_tcr(self):
        """Collate dict batch with TCR sequences."""
        batch = [
            {"peptide": "AAA", "tcr_a": "CAVRD", "tcr_b": "CASSIR"},
            {"peptide": "BBB", "tcr_a": "CAVMD", "tcr_b": "CASSLG"},
        ]
        result = collate_dict_batch(batch)

        assert "tcr_a_tok" in result
        assert "tcr_b_tok" in result

    def test_collate_dict_with_binding(self):
        """Collate dict batch with binding labels."""
        batch = [
            {"peptide": "AAA", "bind_value": 4.5, "bind_qual": 0},
            {"peptide": "BBB", "bind_value": 6.0, "bind_qual": -1},
        ]
        result = collate_dict_batch(batch)

        assert "bind_target" in result
        assert "bind_qual" in result
        assert "bind_mask" in result
        assert "targets" in result
        assert "target_masks" in result
        assert "target_quals" in result
        assert "binding" in result["targets"]
        assert result["bind_target"].shape == (2, 1)

    def test_collate_dict_with_tcell(self):
        """Collate dict batch with T-cell labels."""
        batch = [
            {"peptide": "AAA", "tcell_label": 1.0},
            {"peptide": "BBB", "tcell_label": 0.0},
        ]
        result = collate_dict_batch(batch)

        assert "tcell_label" in result
        assert "tcell_mask" in result
        assert result["target_masks"]["tcell"].shape == (2,)
        assert result["tcell_label"].shape == (2,)

    def test_collate_dict_with_elution(self):
        """Collate dict batch with elution labels."""
        batch = [
            {"peptide": "AAA", "elution_label": 1.0},
            {"peptide": "BBB", "elution_label": 0.0},
        ]
        result = collate_dict_batch(batch)

        assert "elution_label" in result
        assert "elution_mask" in result

    def test_collate_dict_custom_tokenizer(self):
        """Can pass custom tokenizer."""
        tokenizer = Tokenizer()
        batch = [{"peptide": "AAA"}]
        result = collate_dict_batch(batch, tokenizer=tokenizer)

        assert "pep_tok" in result
