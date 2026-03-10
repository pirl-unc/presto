"""Tests for predictor inference API."""

import math
import pytest
import torch
import tempfile
import os

from presto.data.groove import prepare_mhc_input
from presto.inference.predictor import (
    Predictor,
    PresentationResult,
    RecognitionResult,
    TiledPresentationResult,
)
from presto.models.presto import Presto


class TestPresentationResult:
    """Tests for PresentationResult dataclass."""

    def test_result_fields(self):
        """Result has all expected fields."""
        result = PresentationResult(
            peptide="SIINFEKL",
            mhc_class="I",
            processing_prob=0.8,
            binding_prob=0.9,
            presentation_prob=0.72,
            binding_latents={"log_koff": -1.0, "log_kon_intrinsic": 3.0},
            assays={"kd_pred": 4.5},
        )
        assert result.peptide == "SIINFEKL"
        assert result.presentation_prob == 0.72


class TestRecognitionResult:
    """Tests for RecognitionResult dataclass."""

    def test_result_fields(self):
        """Result has all expected fields."""
        result = RecognitionResult(
            peptide="SIINFEKL",
            mhc_class="I",
            presentation_prob=0.8,
            recognition_prob=0.5,
            immunogenicity_prob=0.4,
        )
        assert result.immunogenicity_prob == 0.4


class TestPredictor:
    """Tests for Predictor class."""

    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        return Presto(d_model=64, n_layers=2, n_heads=4)

    @pytest.fixture
    def predictor(self, model):
        """Create predictor with test model."""
        return Predictor(model, device="cpu")

    def test_predictor_init(self, model):
        """Predictor initializes correctly."""
        predictor = Predictor(model)
        assert predictor.model is not None
        assert predictor.tokenizer is not None

    def test_predictor_device(self, model):
        """Predictor uses specified device."""
        predictor = Predictor(model, device="cpu")
        assert predictor.device == "cpu"

    def test_predictor_with_allele_sequences(self, model):
        """Predictor can use allele sequence lookup."""
        allele_seqs = {"HLA-A*02:01": "MAVMAPRTLLLLLSGALALTQ"}
        predictor = Predictor(model, allele_sequences=allele_seqs)
        assert "HLA-A*02:01" in predictor.allele_sequences


class TestPredictorMHCClassInference:
    """Tests for MHC class inference."""

    @pytest.fixture
    def predictor(self):
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        return Predictor(model, device="cpu")

    def test_infer_class_i(self, predictor):
        """Class I alleles are detected."""
        assert predictor._infer_mhc_class("HLA-A*02:01") == "I"
        assert predictor._infer_mhc_class("HLA-B*07:02") == "I"
        assert predictor._infer_mhc_class("HLA-C*04:01") == "I"
        assert predictor._infer_mhc_class("Ia") == "I"

    def test_infer_class_ii(self, predictor):
        """Class II alleles are detected."""
        assert predictor._infer_mhc_class("HLA-DRB1*01:01") == "II"
        assert predictor._infer_mhc_class("HLA-DQB1*02:01") == "II"
        assert predictor._infer_mhc_class("HLA-DPB1*01:01") == "II"
        assert predictor._infer_mhc_class("IIa") == "II"

    def test_infer_default(self, predictor):
        """Default to Class I only when allele input is absent."""
        assert predictor._infer_mhc_class(None) == "I"
        with pytest.raises(ValueError, match="mhcgnomes failed to infer MHC class"):
            predictor._infer_mhc_class("unknown")


class TestPredictPresentation:
    """Tests for presentation prediction."""

    @pytest.fixture
    def predictor(self):
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        return Predictor(model, device="cpu")

    def test_resolve_class_ii_dr_beta_only_uses_default_dra(self):
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        predictor = Predictor(
            model,
            device="cpu",
            auto_load_index_csv=False,
            allele_sequences={
                "HLA-DRA*01:01": "D" * 181,
                "HLA-DRB1*01:01": "E" * 181,
            },
        )
        prepared = prepare_mhc_input(
            mhc_a="D" * 181,
            mhc_b="E" * 181,
            mhc_class="II",
        )

        mhc_a_seq, mhc_b_seq = predictor._resolve_mhc_pair_sequences(
            allele="HLA-DRB1*01:01",
            mhc_sequence=None,
            mhc_b_sequence=None,
            mhc_class="II",
            species="human",
            require_species_for_class_i_b2m=True,
        )

        assert mhc_a_seq == prepared.groove_half_1
        assert mhc_b_seq == prepared.groove_half_2

    def test_predict_presentation_basic(self, predictor):
        """Basic presentation prediction works."""
        result = predictor.predict_presentation(
            peptide="SIINFEKL",
            mhc_sequence="MAVMAPRTLLLLLSGALALTQTWAG",
            species="human",
        )

        assert isinstance(result, PresentationResult)
        assert result.peptide == "SIINFEKL"
        assert 0 <= result.processing_prob <= 1
        assert 0 <= result.binding_prob <= 1
        assert 0 <= result.presentation_prob <= 1

    def test_predict_presentation_with_allele(self, predictor):
        """Prediction with allele name."""
        result = predictor.predict_presentation(
            peptide="GILGFVFTL",
            allele="HLA-A*02:01",
        )

        assert result.mhc_class in {"I", "II"}

    class _CaptureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.last_kwargs = None

        def to(self, device):
            return self

        def eval(self):
            return self

        def forward(self, **kwargs):
            self.last_kwargs = kwargs
            return {
                "processing_logit": torch.tensor([0.0], dtype=torch.float32),
                "binding_logit": torch.tensor([0.0], dtype=torch.float32),
                "presentation_logit": torch.tensor([0.0], dtype=torch.float32),
                "binding_latents": {
                    "log_koff": torch.tensor([0.0]),
                    "log_kon_intrinsic": torch.tensor([0.0]),
                    "log_kon_chaperone": torch.tensor([0.0]),
                },
                "assays": {"KD_nM": torch.tensor([4.0], dtype=torch.float32)},
                "mhc_class_probs": torch.tensor([[0.3, 0.7]], dtype=torch.float32),
            }

    def test_predict_presentation_default_uses_model_class_inference(self):
        model = self._CaptureModel()
        predictor = Predictor(
            model,
            device="cpu",
            auto_load_index_csv=False,
            strict_allele_resolution=False,
        )
        result = predictor.predict_presentation(
            peptide="SIINFEKL",
            allele="HLA-A*02:01",
            require_species_for_class_i_b2m=False,
        )
        assert model.last_kwargs is not None
        assert model.last_kwargs["mhc_class"] is None
        assert result.mhc_class == "II"

    def test_predict_presentation_accepts_hard_class_override(self):
        model = self._CaptureModel()
        predictor = Predictor(
            model,
            device="cpu",
            auto_load_index_csv=False,
            strict_allele_resolution=False,
        )
        result = predictor.predict_presentation(
            peptide="SIINFEKL",
            allele="HLA-A*02:01",
            mhc_class="I",
            require_species_for_class_i_b2m=False,
        )
        assert model.last_kwargs is not None
        class_override = model.last_kwargs["mhc_class"]
        assert class_override == "I"
        assert result.mhc_class == "I"

    def test_predict_presentation_passes_named_override_controls(self):
        model = self._CaptureModel()
        predictor = Predictor(
            model,
            device="cpu",
            auto_load_index_csv=False,
            strict_allele_resolution=False,
        )
        predictor.predict_presentation(
            peptide="SIINFEKL",
            allele="HLA-A*02:01",
            mhc_species="murine",
            immune_species="human",
            species_of_origin="viruses",
            require_species_for_class_i_b2m=False,
        )
        assert model.last_kwargs is not None
        assert model.last_kwargs["mhc_species"] == "murine"
        assert model.last_kwargs["immune_species"] == "human"
        assert model.last_kwargs["species_of_origin"] == "viruses"

    def test_predict_presentation_unresolved_allele_raises_in_strict_mode(self):
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        predictor = Predictor(
            model,
            device="cpu",
            auto_load_index_csv=False,
            strict_allele_resolution=True,
        )
        with pytest.raises(ValueError, match="Could not resolve allele"):
            predictor.predict_presentation(
                peptide="SIINFEKL",
                allele="HLA-A*99:99",
            )

    def test_predict_presentation_unresolved_allele_allowed_when_not_strict(self):
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        predictor = Predictor(
            model,
            device="cpu",
            auto_load_index_csv=False,
            strict_allele_resolution=False,
        )
        result = predictor.predict_presentation(
            peptide="SIINFEKL",
            allele="HLA-A*99:99",
        )
        assert result.mhc_class in {"I", "II"}

    def test_predict_presentation_non_strict_does_not_tokenize_raw_allele_string(self):
        model = self._CaptureModel()
        predictor = Predictor(
            model,
            device="cpu",
            auto_load_index_csv=False,
            strict_allele_resolution=False,
        )
        predictor.predict_presentation(
            peptide="SIINFEKL",
            allele="HLA-A*99:99",
            require_species_for_class_i_b2m=False,
        )
        assert model.last_kwargs is not None
        mhc_a_tok = model.last_kwargs["mhc_a_tok"]
        assert torch.count_nonzero(mhc_a_tok).item() == 0

    def test_predictor_sequence_validation_matches_vocab_without_bzuo(self, predictor):
        assert predictor._looks_like_amino_acid_sequence("ACDEFGHIKLMNPQRSTVWYX")
        assert not predictor._looks_like_amino_acid_sequence("ACDEB")
        assert not predictor._looks_like_amino_acid_sequence("ACDEZ")
        assert not predictor._looks_like_amino_acid_sequence("ACDEU")
        assert not predictor._looks_like_amino_acid_sequence("ACDEO")

    def test_predict_presentation_accepts_class_alias(self, predictor):
        """Class aliases (e.g., Ia) are normalized."""
        result = predictor.predict_presentation(
            peptide="SIINFEKL",
            mhc_sequence="MAVMAPRTLLLLLSGALALTQTWAG",
            mhc_class="Ia",
            species="human",
        )
        assert result.mhc_class == "I"

    def test_predict_presentation_allows_direct_class_i_without_species(self, predictor):
        """Class I direct-chain prediction no longer requires species for B2M."""
        result = predictor.predict_presentation(
            peptide="SIINFEKL",
            mhc_sequence="MAVMAPRTLLLLLSGALALTQTWAG",
            mhc_class="Ia",
        )
        assert result.mhc_class == "I"

    def test_predict_presentation_class_ii(self, predictor):
        """Class II presentation prediction."""
        result = predictor.predict_presentation(
            peptide="AGFKGEQGPKGEPG",  # 15-mer for Class II
            allele="HLA-DRB1*01:01",
            mhc_class="II",
        )

        assert result.mhc_class == "II"

    def test_predict_presentation_with_flanks(self, predictor):
        """Prediction with processing flanks."""
        result = predictor.predict_presentation(
            peptide="SIINFEKL",
            mhc_sequence="MAVMAPRTL",
            species="human",
            flank_n="AAA",
            flank_c="GGG",
        )

        assert isinstance(result, PresentationResult)

    def test_predict_presentation_returns_latents(self, predictor):
        """Prediction returns binding latents."""
        result = predictor.predict_presentation(
            peptide="SIINFEKL",
            mhc_sequence="MAVMAPRTL",
            species="human",
        )

        assert "log_koff" in result.binding_latents
        assert "log_kon_intrinsic" in result.binding_latents
        assert "log_kon_chaperone" in result.binding_latents

    def test_predict_presentation_returns_assays(self, predictor):
        """Prediction returns assay predictions."""
        result = predictor.predict_presentation(
            peptide="SIINFEKL",
            mhc_sequence="MAVMAPRTL",
            species="human",
        )

        # Should have at least KD prediction
        assert len(result.assays) > 0


class TestPredictTiledPresentation:
    """Tests for tiled protein presentation prediction."""

    @pytest.fixture
    def predictor(self):
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        return Predictor(model, device="cpu")

    def test_predict_tiled_presentation_returns_hits(self, predictor):
        result = predictor.predict_tiled_presentation(
            protein_sequence="MPEPSLLQHLIGLQWERTY",
            allele="HLA-A*02:01",
            min_length=9,
            max_length=9,
            flank_size=3,
            top_k=0,
        )
        assert isinstance(result, TiledPresentationResult)
        assert result.total_candidates == len("MPEPSLLQHLIGLQWERTY") - 9 + 1
        assert any(hit.peptide == "SLLQHLIGL" for hit in result.hits)
        assert len(result.hits) == result.total_candidates

    def test_predict_tiled_presentation_flanks_match_source_context(self, predictor):
        result = predictor.predict_tiled_presentation(
            protein_sequence="ACDEFGHIKLMNPQ",
            allele="HLA-A*02:01",
            min_length=4,
            max_length=4,
            flank_size=2,
            top_k=0,
        )
        hits_by_start = {hit.start: hit for hit in result.hits}
        assert hits_by_start[0].flank_n == ""
        assert hits_by_start[0].flank_c == "FG"
        assert hits_by_start[4].peptide == "FGHI"
        assert hits_by_start[4].flank_n == "DE"
        assert hits_by_start[4].flank_c == "KL"

    def test_predict_tiled_presentation_top_k(self, predictor):
        result = predictor.predict_tiled_presentation(
            protein_sequence="MPEPSLLQHLIGLQWERTY",
            allele="HLA-A*02:01",
            min_length=8,
            max_length=10,
            top_k=5,
            sort_by="binding",
        )
        assert len(result.hits) == 5
        scores = [hit.binding_prob for hit in result.hits]
        assert scores == sorted(scores, reverse=True)

    def test_predict_tiled_presentation_validates_length_range(self, predictor):
        with pytest.raises(ValueError):
            predictor.predict_tiled_presentation(
                protein_sequence="MPEPSLLQHLIGLQWERTY",
                allele="HLA-A*02:01",
                min_length=12,
                max_length=8,
            )


class TestPredictPresentationCalibration:
    """Tests for calibrated binding probability from KD predictions."""

    class _DummyModel(torch.nn.Module):
        def __init__(self, kd_log10: float = 3.0, binding_logit: float = 0.0, include_kd: bool = True):
            super().__init__()
            self.kd_log10 = kd_log10
            self.binding_logit = binding_logit
            self.include_kd = include_kd

        def to(self, device):
            return self

        def eval(self):
            return self

        def forward(self, **kwargs):
            assays = {}
            if self.include_kd:
                assays["KD_nM"] = torch.tensor([self.kd_log10], dtype=torch.float32)
            return {
                "processing_logit": torch.tensor([0.0], dtype=torch.float32),
                "binding_logit": torch.tensor([self.binding_logit], dtype=torch.float32),
                "presentation_logit": torch.tensor([0.0], dtype=torch.float32),
                "binding_latents": {
                    "log_koff": torch.tensor([0.0]),
                    "log_kon_intrinsic": torch.tensor([0.0]),
                    "log_kon_chaperone": torch.tensor([0.0]),
                },
                "assays": assays,
            }

    def test_weak_kd_maps_to_near_zero_binding_prob(self):
        predictor = Predictor(self._DummyModel(kd_log10=4.9, binding_logit=10.0), device="cpu")
        result = predictor.predict_presentation(
            peptide="SIINFEKL",
            mhc_sequence="MAVMAPRTLLLLLSGALALTQTWAG",
            species="human",
        )
        assert result.binding_prob < 0.01

    def test_strong_kd_maps_to_high_binding_prob(self):
        predictor = Predictor(self._DummyModel(kd_log10=2.0, binding_logit=-10.0), device="cpu")
        result = predictor.predict_presentation(
            peptide="SIINFEKL",
            mhc_sequence="MAVMAPRTLLLLLSGALALTQTWAG",
            species="human",
        )
        assert result.binding_prob > 0.8

    def test_falls_back_to_binding_logit_when_kd_missing(self):
        predictor = Predictor(
            self._DummyModel(kd_log10=3.0, binding_logit=-2.0, include_kd=False),
            device="cpu",
        )
        result = predictor.predict_presentation(
            peptide="SIINFEKL",
            mhc_sequence="MAVMAPRTLLLLLSGALALTQTWAG",
            species="human",
        )
        assert pytest.approx(result.binding_prob, rel=1e-4) == float(torch.sigmoid(torch.tensor(-2.0)))

    def test_uses_model_binding_calibration_by_default(self):
        model = self._DummyModel(kd_log10=3.2, binding_logit=0.0, include_kd=True)
        model.binding_midpoint_nM = 1000.0
        model.binding_log10_scale = 0.5
        predictor = Predictor(model, device="cpu")
        result = predictor.predict_presentation(
            peptide="SIINFEKL",
            mhc_sequence="MAVMAPRTLLLLLSGALALTQTWAG",
            species="human",
        )
        expected = 1.0 / (
            1.0
            + math.exp(
                -((math.log10(model.binding_midpoint_nM) - model.kd_log10) / model.binding_log10_scale)
            )
        )
        assert result.binding_prob == pytest.approx(expected, rel=1e-6)


class TestPredictPresentationMultiAllele:
    """Tests for multi-allele presentation prediction."""

    @pytest.fixture
    def predictor(self):
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        return Predictor(model, device="cpu")

    def test_multi_allele_basic(self, predictor):
        """Multi-allele prediction with Noisy-OR aggregation."""
        result = predictor.predict_presentation_multi_allele(
            peptide="SIINFEKL",
            mhc_sequences=["MAVMAPRTL", "MAVMAPRTX", "MAVMAPRTQ"],
            species="human",
        )

        assert "bag_presentation_prob" in result
        assert "per_allele" in result
        assert len(result["per_allele"]) == 3
        assert 0 <= result["bag_presentation_prob"] <= 1

    def test_multi_allele_with_names(self, predictor):
        """Multi-allele prediction with allele names."""
        result = predictor.predict_presentation_multi_allele(
            peptide="SIINFEKL",
            alleles=["HLA-A*02:01", "HLA-A*03:01"],
        )

        assert len(result["per_allele"]) == 2
        assert result["per_allele"][0]["allele"] == "HLA-A*02:01"

    def test_multi_allele_bag_geq_max_instance(self, predictor):
        """Noisy-OR bag prob >= max instance prob."""
        result = predictor.predict_presentation_multi_allele(
            peptide="SIINFEKL",
            mhc_sequences=["MAVMAPRTL", "MAVMAPRTX"],
            species="human",
        )

        max_instance = max(r["presentation_prob"] for r in result["per_allele"])
        assert result["bag_presentation_prob"] >= max_instance - 1e-5

    def test_multi_allele_requires_input(self, predictor):
        """Must provide alleles or sequences."""
        with pytest.raises(ValueError):
            predictor.predict_presentation_multi_allele(peptide="SIINFEKL")


class TestPredictRecognition:
    """Tests for repertoire-level recognition API."""

    @pytest.fixture
    def predictor(self):
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        return Predictor(model, device="cpu")

    def test_predict_recognition_basic(self, predictor):
        result = predictor.predict_recognition(
            peptide="SIINFEKL",
            mhc_sequence="MAVMAPRTLLLLLSGALALTQTWAG",
            mhc_class="I",
            species="human",
        )

        assert isinstance(result, RecognitionResult)
        assert result.mhc_class == "I"
        assert 0 <= result.presentation_prob <= 1
        assert 0 <= result.recognition_prob <= 1
        assert 0 <= result.immunogenicity_prob <= 1

    def test_predict_recognition_class_ii(self, predictor):
        result = predictor.predict_recognition(
            peptide="AGFKGEQGPKGEPG",
            allele="HLA-DRB1*01:01",
            mhc_class="II",
            species="human",
        )
        assert result.mhc_class == "II"
        assert 0 <= result.recognition_prob <= 1


class TestEmbeddings:
    """Tests for embedding extraction."""

    @pytest.fixture
    def predictor(self):
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        return Predictor(model, device="cpu")

    def test_embed_pmhc(self, predictor):
        """Extract pMHC embedding."""
        embedding = predictor.embed_pmhc(
            peptide="SIINFEKL",
            mhc_sequence="MAVMAPRTL",
            species="human",
        )

        # pmhc_vec = interaction_vec: pmhc_interaction_vec_dim = n_queries * token_dim
        assert embedding.shape == (1, predictor.model.pmhc_interaction_vec_dim)


class TestPredictorFromCheckpoint:
    """Tests for loading predictor from checkpoint."""

    def test_from_checkpoint(self):
        """Load predictor from saved checkpoint."""
        # Create and save a model
        model = Presto(d_model=64, n_layers=2, n_heads=4)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            checkpoint_path = f.name

        try:
            predictor = Predictor.from_checkpoint(
                checkpoint_path,
                d_model=64,
                n_layers=2,
                n_heads=4,
                device="cpu",
            )

            # Should be able to make predictions
            result = predictor.predict_presentation(
                peptide="SIINFEKL",
                mhc_sequence="MAVMAPRTL",
                species="human",
            )
            assert isinstance(result, PresentationResult)
        finally:
            os.unlink(checkpoint_path)

    def test_from_checkpoint_with_model_state_dict(self):
        """Load from checkpoint with model_state_dict key."""
        model = Presto(d_model=64, n_layers=2, n_heads=4)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"model_state_dict": model.state_dict()}, f.name)
            checkpoint_path = f.name

        try:
            predictor = Predictor.from_checkpoint(
                checkpoint_path,
                d_model=64,
                n_layers=2,
                n_heads=4,
                device="cpu",
            )
            assert predictor.model is not None
        finally:
            os.unlink(checkpoint_path)

    def test_from_checkpoint_uses_config(self):
        """Loads model config from checkpoint when args omitted."""
        model = Presto(d_model=64, n_layers=2, n_heads=4)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {"d_model": 64, "n_layers": 2, "n_heads": 4},
                },
                f.name,
            )
            checkpoint_path = f.name

        try:
            predictor = Predictor.from_checkpoint(checkpoint_path, device="cpu")
            result = predictor.predict_presentation(
                peptide="SIINFEKL",
                mhc_sequence="MAVMAPRTL",
                species="human",
            )
            assert isinstance(result, PresentationResult)
        finally:
            os.unlink(checkpoint_path)

    def test_from_checkpoint_with_serialization_metadata(self):
        """Loads a self-describing checkpoint without architecture args."""
        from presto.training.checkpointing import save_model_checkpoint

        model = Presto(d_model=64, n_layers=2, n_heads=4)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            save_model_checkpoint(
                f.name,
                model=model,
                epoch=1,
                metrics={"val_loss": 0.5},
                train_config={"example": True},
            )
            checkpoint_path = f.name

        try:
            predictor = Predictor.from_checkpoint(checkpoint_path, device="cpu")
            assert predictor.model.d_model == 64
        finally:
            os.unlink(checkpoint_path)
