"""Tests for predictor inference API."""

import pytest
import torch
import tempfile
import os

from presto.inference.predictor import (
    Predictor,
    PresentationResult,
    RecognitionResult,
    ChainClassificationResult,
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
            binding_latents={"stability": 0.5, "intrinsic": 0.6},
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
            tcr_alpha="CAVRD",
            tcr_beta="CASSIR",
            presentation_prob=0.8,
            match_prob=0.5,
            immunogenicity_prob=0.4,
        )
        assert result.immunogenicity_prob == 0.4


class TestChainClassificationResult:
    """Tests for ChainClassificationResult dataclass."""

    def test_result_fields(self):
        """Result has all expected fields."""
        result = ChainClassificationResult(
            sequence="CAVRDSSYKLIF",
            species="human",
            species_prob=0.99,
            chain_type="TRA",
            chain_type_prob=0.95,
            phenotype="CD8_T",
            phenotype_prob=0.85,
        )
        assert result.chain_type == "TRA"


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

    def test_infer_class_ii(self, predictor):
        """Class II alleles are detected."""
        assert predictor._infer_mhc_class("HLA-DRB1*01:01") == "II"
        assert predictor._infer_mhc_class("HLA-DQB1*02:01") == "II"
        assert predictor._infer_mhc_class("HLA-DPB1*01:01") == "II"

    def test_infer_default(self, predictor):
        """Default to Class I when unknown."""
        assert predictor._infer_mhc_class(None) == "I"
        assert predictor._infer_mhc_class("unknown") == "I"


class TestPredictPresentation:
    """Tests for presentation prediction."""

    @pytest.fixture
    def predictor(self):
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        return Predictor(model, device="cpu")

    def test_predict_presentation_basic(self, predictor):
        """Basic presentation prediction works."""
        result = predictor.predict_presentation(
            peptide="SIINFEKL",
            mhc_sequence="MAVMAPRTLLLLLSGALALTQTWAG",
        )

        assert isinstance(result, PresentationResult)
        assert result.peptide == "SIINFEKL"
        assert 0 <= result.processing_prob <= 1
        assert 0 <= result.binding_prob <= 1
        assert 0 <= result.presentation_prob <= 1

    def test_predict_presentation_with_allele(self, predictor):
        """Prediction with allele name (no sequence lookup)."""
        result = predictor.predict_presentation(
            peptide="GILGFVFTL",
            allele="HLA-A*02:01",
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
            flank_n="AAA",
            flank_c="GGG",
        )

        assert isinstance(result, PresentationResult)

    def test_predict_presentation_returns_latents(self, predictor):
        """Prediction returns binding latents."""
        result = predictor.predict_presentation(
            peptide="SIINFEKL",
            mhc_sequence="MAVMAPRTL",
        )

        assert "stability" in result.binding_latents
        assert "intrinsic" in result.binding_latents
        assert "chaperone" in result.binding_latents

    def test_predict_presentation_returns_assays(self, predictor):
        """Prediction returns assay predictions."""
        result = predictor.predict_presentation(
            peptide="SIINFEKL",
            mhc_sequence="MAVMAPRTL",
        )

        # Should have at least KD prediction
        assert len(result.assays) > 0


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
        )

        max_instance = max(r["presentation_prob"] for r in result["per_allele"])
        assert result["bag_presentation_prob"] >= max_instance - 1e-5

    def test_multi_allele_requires_input(self, predictor):
        """Must provide alleles or sequences."""
        with pytest.raises(ValueError):
            predictor.predict_presentation_multi_allele(peptide="SIINFEKL")


class TestPredictRecognition:
    """Tests for TCR-pMHC recognition prediction."""

    @pytest.fixture
    def predictor(self):
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        return Predictor(model, device="cpu")

    def test_predict_recognition_paired(self, predictor):
        """Recognition with paired TCR chains."""
        result = predictor.predict_recognition(
            peptide="SIINFEKL",
            mhc_sequence="MAVMAPRTL",
            tcr_alpha="CAVRDSSYKLIF",
            tcr_beta="CASSIRSSYEQYF",
        )

        assert isinstance(result, RecognitionResult)
        assert 0 <= result.presentation_prob <= 1
        assert 0 <= result.match_prob <= 1
        assert 0 <= result.immunogenicity_prob <= 1

    def test_predict_recognition_beta_only(self, predictor):
        """Recognition with beta chain only."""
        result = predictor.predict_recognition(
            peptide="SIINFEKL",
            mhc_sequence="MAVMAPRTL",
            tcr_beta="CASSIRSSYEQYF",
        )

        assert result.tcr_alpha is None
        assert result.tcr_beta == "CASSIRSSYEQYF"

    def test_predict_recognition_alpha_only(self, predictor):
        """Recognition with alpha chain only."""
        result = predictor.predict_recognition(
            peptide="SIINFEKL",
            mhc_sequence="MAVMAPRTL",
            tcr_alpha="CAVRDSSYKLIF",
        )

        assert result.tcr_alpha == "CAVRDSSYKLIF"
        assert result.tcr_beta is None

    def test_predict_recognition_requires_tcr(self, predictor):
        """Must provide at least one TCR chain."""
        with pytest.raises(ValueError):
            predictor.predict_recognition(
                peptide="SIINFEKL",
                mhc_sequence="MAVMAPRTL",
            )


class TestClassifyChain:
    """Tests for chain classification."""

    @pytest.fixture
    def predictor(self):
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        return Predictor(model, device="cpu")

    def test_classify_chain_basic(self, predictor):
        """Basic chain classification."""
        result = predictor.classify_chain("CAVRDSSYKLIF")

        assert isinstance(result, ChainClassificationResult)
        assert result.sequence == "CAVRDSSYKLIF"
        assert result.species in ["human", "mouse", "macaque", "other"]
        assert 0 <= result.species_prob <= 1
        assert 0 <= result.chain_type_prob <= 1
        assert 0 <= result.phenotype_prob <= 1


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
        )

        assert embedding.shape == (1, 64)  # d_model=64

    def test_embed_tcr_paired(self, predictor):
        """Extract TCR embedding (paired)."""
        embedding = predictor.embed_tcr(
            alpha="CAVRDSSYKLIF",
            beta="CASSIRSSYEQYF",
        )

        assert embedding.shape == (1, 64)

    def test_embed_tcr_single(self, predictor):
        """Extract TCR embedding (single chain)."""
        embedding = predictor.embed_tcr(beta="CASSIRSSYEQYF")
        assert embedding.shape == (1, 64)

    def test_embed_tcr_requires_chain(self, predictor):
        """Must provide at least one TCR chain."""
        with pytest.raises(ValueError):
            predictor.embed_tcr()


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
            )
            assert isinstance(result, PresentationResult)
        finally:
            os.unlink(checkpoint_path)
