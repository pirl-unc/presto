"""High-level predictor API for PRESTO.

User-friendly interface for making predictions without dealing with tokenization.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

import torch
import torch.nn.functional as F

from ..models.presto import Presto
from ..models.pmhc import stable_noisy_or, enumerate_registers
from ..data.tokenizer import Tokenizer
from ..data.vocab import (
    CHAIN_TYPES, CELL_TYPES, MHC_TYPES, SPECIES,
    IDX_TO_CHAIN, IDX_TO_CELL, IDX_TO_MHC, IDX_TO_SPECIES,
    is_cdr3_only, get_base_chain_type,
)


@dataclass
class PresentationResult:
    """Result from presentation prediction."""
    peptide: str
    mhc_class: str
    processing_prob: float
    binding_prob: float
    presentation_prob: float
    binding_latents: Dict[str, float]
    assays: Dict[str, float]


@dataclass
class RecognitionResult:
    """Result from TCR-pMHC recognition prediction."""
    peptide: str
    mhc_class: str
    tcr_alpha: Optional[str]
    tcr_beta: Optional[str]
    presentation_prob: float
    match_prob: float
    immunogenicity_prob: float


@dataclass
class ChainClassificationResult:
    """Result from chain classification."""
    sequence: str
    species: str
    species_prob: float
    chain_type: str
    chain_type_prob: float
    phenotype: str
    phenotype_prob: float


class Predictor:
    """High-level predictor for PRESTO model.

    Example usage:
        predictor = Predictor.from_checkpoint("model.pt")

        # Predict presentation
        result = predictor.predict_presentation(
            peptide="SIINFEKL",
            allele="HLA-A*02:01",  # or mhc_sequence="..."
        )
        print(f"Presentation probability: {result.presentation_prob:.3f}")

        # Predict TCR recognition
        result = predictor.predict_recognition(
            peptide="SIINFEKL",
            allele="HLA-A*02:01",
            tcr_alpha="CAVRDSSYKLIF",
            tcr_beta="CASSIRSSYEQYF",
        )
        print(f"Immunogenicity: {result.immunogenicity_prob:.3f}")
    """

    def __init__(
        self,
        model: Presto,
        tokenizer: Tokenizer = None,
        allele_sequences: Dict[str, str] = None,
        device: str = None,
    ):
        """Initialize predictor.

        Args:
            model: Trained Presto model
            tokenizer: Tokenizer (created if not provided)
            allele_sequences: Dict mapping allele names to sequences
            device: Device to run on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer or Tokenizer()
        self.allele_sequences = allele_sequences or {}

        # Default beta2m sequence (human)
        self.beta2m_sequence = "MSRSVALAVLALLSLSGLEA"  # Truncated for demo

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        d_model: Optional[int] = None,
        n_layers: Optional[int] = None,
        n_heads: Optional[int] = None,
        device: str = None,
        **kwargs,
    ) -> "Predictor":
        """Load predictor from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            d_model, n_layers, n_heads: Model architecture params
            device: Device to run on
            **kwargs: Additional args for Predictor.__init__

        Returns:
            Initialized Predictor
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        config = None
        state_dict = checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            config = checkpoint.get("model_config") or checkpoint.get("config")

        if config:
            if d_model is None:
                d_model = config.get("d_model")
            if n_layers is None:
                n_layers = config.get("n_layers")
            if n_heads is None:
                n_heads = config.get("n_heads")

        d_model = d_model or 256
        n_layers = n_layers or 4
        n_heads = n_heads or 8

        model = Presto(d_model=d_model, n_layers=n_layers, n_heads=n_heads)
        model.load_state_dict(state_dict)

        return cls(model, device=device, **kwargs)

    def _get_mhc_sequence(self, allele: str = None, mhc_sequence: str = None) -> str:
        """Resolve MHC sequence from allele name or direct sequence."""
        if mhc_sequence:
            return mhc_sequence
        if allele and allele in self.allele_sequences:
            return self.allele_sequences[allele]
        # Return allele name as fallback (will be tokenized as sequence)
        return allele or ""

    def _infer_mhc_class(self, allele: str = None) -> str:
        """Infer MHC class from allele name."""
        if not allele:
            return "I"
        allele_upper = allele.upper()
        # Class II indicators
        if any(x in allele_upper for x in ["DR", "DQ", "DP", "D-", "CLASS-II", "CLASSII"]):
            return "II"
        return "I"

    def _tokenize(self, seq: str, max_len: int) -> torch.Tensor:
        """Tokenize a sequence."""
        return self.tokenizer.batch_encode([seq], max_len=max_len, pad=True).to(self.device)

    @torch.no_grad()
    def predict_presentation(
        self,
        peptide: str,
        allele: str = None,
        mhc_sequence: str = None,
        mhc_b_sequence: str = None,
        mhc_class: str = None,
        flank_n: str = None,
        flank_c: str = None,
    ) -> PresentationResult:
        """Predict presentation probability for a peptide-MHC pair.

        Args:
            peptide: Peptide sequence
            allele: MHC allele name (e.g., "HLA-A*02:01")
            mhc_sequence: MHC alpha chain sequence (alternative to allele)
            mhc_b_sequence: MHC beta chain sequence (beta2m for Class I)
            mhc_class: "I" or "II" (inferred from allele if not provided)
            flank_n: N-terminal processing flank
            flank_c: C-terminal processing flank

        Returns:
            PresentationResult with probabilities
        """
        mhc_class = mhc_class or self._infer_mhc_class(allele)
        mhc_a_seq = self._get_mhc_sequence(allele, mhc_sequence)
        mhc_b_seq = mhc_b_sequence or (self.beta2m_sequence if mhc_class == "I" else "")

        # Tokenize
        pep_tok = self._tokenize(peptide, max_len=30)
        mhc_a_tok = self._tokenize(mhc_a_seq, max_len=400)
        mhc_b_tok = self._tokenize(mhc_b_seq, max_len=200)
        flank_n_tok = self._tokenize(flank_n, max_len=30) if flank_n else None
        flank_c_tok = self._tokenize(flank_c, max_len=30) if flank_c else None

        # Forward pass
        outputs = self.model(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            mhc_class=mhc_class,
            flank_n_tok=flank_n_tok,
            flank_c_tok=flank_c_tok,
        )

        # Extract results
        proc_prob = torch.sigmoid(outputs["processing_logit"]).item()
        bind_prob = torch.sigmoid(outputs["binding_logit"]).item()
        pres_prob = torch.sigmoid(outputs["presentation_logit"]).item()

        latents = {
            k: v.item() for k, v in outputs["binding_latents"].items()
        }
        assays = {
            k: v.item() for k, v in outputs["assays"].items()
        }

        return PresentationResult(
            peptide=peptide,
            mhc_class=mhc_class,
            processing_prob=proc_prob,
            binding_prob=bind_prob,
            presentation_prob=pres_prob,
            binding_latents=latents,
            assays=assays,
        )

    @torch.no_grad()
    def predict_presentation_multi_allele(
        self,
        peptide: str,
        alleles: List[str] = None,
        mhc_sequences: List[str] = None,
        mhc_class: str = "I",
    ) -> Dict[str, Any]:
        """Predict presentation with multiple alleles (MS/EL scenario).

        Uses Noisy-OR aggregation over alleles.

        Args:
            peptide: Peptide sequence
            alleles: List of MHC allele names
            mhc_sequences: List of MHC sequences (alternative)
            mhc_class: "I" or "II"

        Returns:
            Dict with bag probability and per-allele results
        """
        alleles = alleles or []
        mhc_sequences = mhc_sequences or []

        if not alleles and not mhc_sequences:
            raise ValueError("Must provide alleles or mhc_sequences")

        # Use alleles or sequences
        seqs = mhc_sequences if mhc_sequences else [
            self._get_mhc_sequence(a) for a in alleles
        ]

        per_allele = []
        probs = []

        for i, seq in enumerate(seqs):
            result = self.predict_presentation(
                peptide=peptide,
                mhc_sequence=seq,
                mhc_class=mhc_class,
            )
            per_allele.append({
                "allele": alleles[i] if i < len(alleles) else f"seq_{i}",
                "presentation_prob": result.presentation_prob,
            })
            probs.append(result.presentation_prob)

        # Noisy-OR aggregation
        probs_tensor = torch.tensor(probs)
        bag_prob = stable_noisy_or(probs_tensor).item()

        return {
            "peptide": peptide,
            "bag_presentation_prob": bag_prob,
            "per_allele": per_allele,
        }

    @torch.no_grad()
    def predict_recognition(
        self,
        peptide: str,
        allele: str = None,
        mhc_sequence: str = None,
        tcr_alpha: str = None,
        tcr_beta: str = None,
        mhc_class: str = None,
    ) -> RecognitionResult:
        """Predict TCR-pMHC recognition.

        Args:
            peptide: Peptide sequence
            allele: MHC allele name
            mhc_sequence: MHC sequence (alternative)
            tcr_alpha: TCR alpha chain sequence
            tcr_beta: TCR beta chain sequence
            mhc_class: "I" or "II"

        Returns:
            RecognitionResult with probabilities
        """
        if tcr_alpha is None and tcr_beta is None:
            raise ValueError("Must provide at least one TCR chain")

        mhc_class = mhc_class or self._infer_mhc_class(allele)
        mhc_a_seq = self._get_mhc_sequence(allele, mhc_sequence)
        mhc_b_seq = self.beta2m_sequence if mhc_class == "I" else ""

        # Tokenize
        pep_tok = self._tokenize(peptide, max_len=30)
        mhc_a_tok = self._tokenize(mhc_a_seq, max_len=400)
        mhc_b_tok = self._tokenize(mhc_b_seq, max_len=200)
        tcr_a_tok = self._tokenize(tcr_alpha or "", max_len=100) if tcr_alpha else None
        tcr_b_tok = self._tokenize(tcr_beta or "", max_len=100) if tcr_beta else None

        # Forward pass
        outputs = self.model(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            mhc_class=mhc_class,
            tcr_a_tok=tcr_a_tok,
            tcr_b_tok=tcr_b_tok,
        )

        pres_prob = torch.sigmoid(outputs["presentation_logit"]).item()
        match_prob = torch.sigmoid(outputs["match_logit"]).item()
        ig_prob = torch.sigmoid(outputs["immunogenicity_logit"]).item()

        return RecognitionResult(
            peptide=peptide,
            mhc_class=mhc_class,
            tcr_alpha=tcr_alpha,
            tcr_beta=tcr_beta,
            presentation_prob=pres_prob,
            match_prob=match_prob,
            immunogenicity_prob=ig_prob,
        )

    @torch.no_grad()
    def classify_chain(self, sequence: str) -> ChainClassificationResult:
        """Classify a receptor chain sequence.

        Predicts species, chain type, and cell phenotype.

        Args:
            sequence: Chain sequence (full or CDR3)

        Returns:
            ChainClassificationResult
        """
        chain_tok = self._tokenize(sequence, max_len=200)
        preds = self.model.predict_chain_attributes(chain_tok)

        species_idx = preds["species"].item()
        chain_idx = preds["chain_type"].item()
        pheno_idx = preds["phenotype"].item()

        return ChainClassificationResult(
            sequence=sequence,
            species=IDX_TO_SPECIES[species_idx],
            species_prob=preds["species_probs"][0, species_idx].item(),
            chain_type=IDX_TO_CHAIN[chain_idx],
            chain_type_prob=preds["chain_probs"][0, chain_idx].item(),
            phenotype=IDX_TO_CELL[pheno_idx],
            phenotype_prob=preds["phenotype_probs"][0, pheno_idx].item(),
        )

    @torch.no_grad()
    def embed_pmhc(
        self,
        peptide: str,
        allele: str = None,
        mhc_sequence: str = None,
        mhc_class: str = None,
    ) -> torch.Tensor:
        """Get pMHC embedding.

        Args:
            peptide: Peptide sequence
            allele: MHC allele name
            mhc_sequence: MHC sequence
            mhc_class: "I" or "II"

        Returns:
            Embedding tensor (1, d_model)
        """
        mhc_class = mhc_class or self._infer_mhc_class(allele)
        mhc_a_seq = self._get_mhc_sequence(allele, mhc_sequence)
        mhc_b_seq = self.beta2m_sequence if mhc_class == "I" else ""

        pep_tok = self._tokenize(peptide, max_len=30)
        mhc_a_tok = self._tokenize(mhc_a_seq, max_len=400)
        mhc_b_tok = self._tokenize(mhc_b_seq, max_len=200)

        return self.model.encode_pmhc(pep_tok, mhc_a_tok, mhc_b_tok, mhc_class)

    @torch.no_grad()
    def embed_tcr(
        self,
        alpha: str = None,
        beta: str = None,
    ) -> torch.Tensor:
        """Get TCR embedding.

        Args:
            alpha: TCR alpha chain sequence
            beta: TCR beta chain sequence

        Returns:
            Embedding tensor (1, d_model)
        """
        if alpha is None and beta is None:
            raise ValueError("Must provide at least one chain")

        alpha_tok = self._tokenize(alpha or "", max_len=100) if alpha else None
        beta_tok = self._tokenize(beta or "", max_len=100) if beta else None

        return self.model.encode_tcr(alpha_tok, beta_tok)
