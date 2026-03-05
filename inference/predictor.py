"""High-level predictor API for Presto.

User-friendly interface for making predictions without dealing with tokenization.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, ClassVar
from pathlib import Path

import torch

from ..models.presto import Presto
from ..models.pmhc import stable_noisy_or
from ..models.affinity import (
    DEFAULT_BINDING_LOG10_SCALE,
    DEFAULT_BINDING_MIDPOINT_NM,
    DEFAULT_MAX_AFFINITY_NM,
    binding_prob_from_kd_log10,
)
from ..data.allele_resolver import (
    infer_species,
    normalize_mhc_class,
    normalize_species_label,
    class_i_beta2m_sequence,
    HUMAN_B2M_SEQUENCE,
)
from ..data.tokenizer import Tokenizer
from ..data.mhc_index import load_mhc_index
from ..training.checkpointing import load_model_from_checkpoint
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


@dataclass
class TiledPresentationHit:
    """One tiled peptide prediction."""
    peptide: str
    start: int
    end: int
    flank_n: str
    flank_c: str
    processing_prob: float
    binding_prob: float
    presentation_prob: float
    assays: Dict[str, float]


@dataclass
class TiledPresentationResult:
    """Protein tiling prediction result."""
    protein_length: int
    total_candidates: int
    min_length: int
    max_length: int
    flank_size: int
    mhc_class: str
    sort_by: str
    hits: List[TiledPresentationHit]


class Predictor:
    """High-level predictor for Presto model.

    Example usage:
        predictor = Predictor.from_checkpoint("model.pt")

        # Predict presentation
        result = predictor.predict_presentation(
            peptide="SIINFEKL",
            allele="HLA-A*02:01",  # or mhc_sequence="..."
        )
        print(f"Presentation probability: {result.presentation_prob:.3f}")
    """

    _AA_SEQUENCE_CHARS: ClassVar[set[str]] = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
    _ALLELE_INDEX_CACHE: ClassVar[Dict[str, Dict[str, str]]] = {}

    def __init__(
        self,
        model: Presto,
        tokenizer: Tokenizer = None,
        allele_sequences: Dict[str, str] = None,
        index_csv: Optional[Union[str, Path]] = None,
        auto_load_index_csv: bool = True,
        strict_allele_resolution: bool = True,
        device: str = None,
        binding_midpoint_nM: Optional[float] = None,
        binding_log10_scale: Optional[float] = None,
    ):
        """Initialize predictor.

        Args:
            model: Trained Presto model
            tokenizer: Tokenizer (created if not provided)
            allele_sequences: Dict mapping allele names to sequences
            index_csv: Optional MHC index CSV for allele->sequence lookup
            auto_load_index_csv: Auto-load default `data/mhc_index.csv` when
                explicit `allele_sequences` are not provided
            strict_allele_resolution: If True, unresolved allele names raise
                instead of being tokenized as amino-acid text
            device: Device to run on
            binding_midpoint_nM: KD value (nM) mapped to ~0.5 binding probability.
                Defaults to model calibration when available.
            binding_log10_scale: Logistic scale in log10(nM) space.
                Defaults to model calibration when available.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer or Tokenizer()
        self.strict_allele_resolution = bool(strict_allele_resolution)
        self.allele_sequences = self._resolve_allele_sequence_lookup(
            allele_sequences=allele_sequences,
            index_csv=index_csv,
            auto_load_index_csv=auto_load_index_csv,
        )
        self.max_affinity_nM = float(
            getattr(model, "max_affinity_nM", DEFAULT_MAX_AFFINITY_NM)
        )
        resolved_midpoint = (
            float(binding_midpoint_nM)
            if binding_midpoint_nM is not None
            else float(getattr(model, "binding_midpoint_nM", DEFAULT_BINDING_MIDPOINT_NM))
        )
        resolved_scale = (
            float(binding_log10_scale)
            if binding_log10_scale is not None
            else float(getattr(model, "binding_log10_scale", DEFAULT_BINDING_LOG10_SCALE))
        )
        self.binding_midpoint_nM = resolved_midpoint
        self.binding_log10_scale = max(resolved_scale, 1e-6)

        # Species-aware beta2m defaults for class I assembly.
        self.beta2m_by_species = {
            "human": HUMAN_B2M_SEQUENCE,
            "mouse": class_i_beta2m_sequence("mouse") or HUMAN_B2M_SEQUENCE,
            "macaque": class_i_beta2m_sequence("macaque") or HUMAN_B2M_SEQUENCE,
        }

    @classmethod
    def _default_index_candidates(cls) -> List[Path]:
        """Default MHC index locations searched when autoloading."""
        candidates = [
            Path.cwd() / "data" / "mhc_index.csv",
            Path(__file__).resolve().parents[1] / "data" / "mhc_index.csv",
            Path.home() / ".cache" / "presto" / "mhc_index.csv",
        ]
        uniq: List[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate.expanduser().resolve()) if candidate.exists() else str(candidate.expanduser())
            if key in seen:
                continue
            seen.add(key)
            uniq.append(candidate)
        return uniq

    @classmethod
    def _normalize_allele_lookup(cls, allele_sequences: Dict[str, str]) -> Dict[str, str]:
        """Normalize lookup keys and sequences for robust matching."""
        normalized: Dict[str, str] = {}
        for allele, sequence in (allele_sequences or {}).items():
            key = str(allele or "").strip()
            seq = str(sequence or "").strip().upper()
            if not key or not seq:
                continue
            normalized[key] = seq
            normalized[key.upper()] = seq
        return normalized

    @classmethod
    def _load_allele_sequences_from_index_csv(
        cls,
        index_csv: Union[str, Path],
    ) -> Dict[str, str]:
        """Load allele->sequence mapping (including prefix aliases) from index CSV."""
        index_path = Path(index_csv).expanduser()
        if not index_path.exists():
            return {}
        cache_key = str(index_path.resolve())
        if cache_key in cls._ALLELE_INDEX_CACHE:
            return dict(cls._ALLELE_INDEX_CACHE[cache_key])

        records = load_mhc_index(str(index_path))
        resolved: Dict[str, str] = {}
        for record in records.values():
            seq = str(record.sequence or "").strip().upper()
            if not seq:
                continue
            for token in {record.normalized, record.allele_raw}:
                name = str(token or "").strip()
                if not name:
                    continue
                resolved[name] = seq
                resolved[name.upper()] = seq
                if ":" in name:
                    parts = name.split(":")
                    for i in range(1, len(parts)):
                        prefix = ":".join(parts[:i]).strip()
                        if not prefix:
                            continue
                        resolved.setdefault(prefix, seq)
                        resolved.setdefault(prefix.upper(), seq)

        cls._ALLELE_INDEX_CACHE[cache_key] = dict(resolved)
        return resolved

    @classmethod
    def _resolve_allele_sequence_lookup(
        cls,
        allele_sequences: Optional[Dict[str, str]],
        index_csv: Optional[Union[str, Path]],
        auto_load_index_csv: bool,
    ) -> Dict[str, str]:
        """Resolve the final allele lookup map from explicit + indexed sources."""
        resolved: Dict[str, str] = {}
        if index_csv is not None:
            resolved.update(cls._load_allele_sequences_from_index_csv(index_csv))
        elif auto_load_index_csv:
            for candidate in cls._default_index_candidates():
                loaded = cls._load_allele_sequences_from_index_csv(candidate)
                if loaded:
                    resolved.update(loaded)
                    break
        if allele_sequences:
            resolved.update(cls._normalize_allele_lookup(allele_sequences))
        return resolved

    @classmethod
    def _looks_like_amino_acid_sequence(cls, value: str) -> bool:
        """Heuristic check for direct amino-acid sequence inputs."""
        seq = str(value or "").strip().upper()
        return bool(seq) and all(ch in cls._AA_SEQUENCE_CHARS for ch in seq)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        d_model: Optional[int] = None,
        n_layers: Optional[int] = None,
        n_heads: Optional[int] = None,
        n_categories: Optional[int] = None,
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
        model, _ = load_model_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_categories=n_categories,
            strict=True,
        )

        return cls(model, device=device, **kwargs)

    def _get_mhc_sequence(self, allele: str = None, mhc_sequence: str = None) -> str:
        """Resolve MHC sequence from allele name or direct sequence."""
        if mhc_sequence:
            return str(mhc_sequence).strip().upper()

        if not allele:
            return ""

        key = str(allele).strip()
        if key in self.allele_sequences:
            return self.allele_sequences[key]

        key_upper = key.upper()
        if key_upper in self.allele_sequences:
            return self.allele_sequences[key_upper]

        if self._looks_like_amino_acid_sequence(key_upper):
            return key_upper

        if self.strict_allele_resolution:
            raise ValueError(
                f"Could not resolve allele '{allele}' to an amino-acid sequence. "
                "Provide `mhc_sequence`, pass `index_csv`, or supply `allele_sequences`."
            )

        return ""

    def _infer_mhc_class(self, allele: str = None) -> str:
        """Infer MHC class from allele name."""
        normalized = normalize_mhc_class(allele)
        if normalized is not None:
            return normalized
        if not allele:
            return "I"
        allele_upper = allele.upper()
        # Class II indicators
        if any(x in allele_upper for x in ["DR", "DQ", "DP", "D-", "CLASS-II", "CLASSII"]):
            return "II"
        return "I"

    def _resolve_species(self, allele: str = None, species: str = None) -> Optional[str]:
        """Resolve species from explicit value or allele prefix."""
        explicit = normalize_species_label(species)
        if explicit is not None:
            return explicit
        if allele:
            return normalize_species_label(infer_species(allele))
        return None

    def _resolve_class_i_beta2m(
        self,
        mhc_class: str,
        mhc_b_sequence: Optional[str],
        allele: Optional[str],
        species: Optional[str],
        require_species_for_class_i_b2m: bool = True,
    ) -> str:
        """Resolve class-I beta2m sequence with species-aware defaults."""
        if mhc_b_sequence:
            return mhc_b_sequence
        if mhc_class != "I":
            return ""

        resolved_species = self._resolve_species(allele=allele, species=species)
        beta2m = self.beta2m_by_species.get(resolved_species or "", None)
        if beta2m is not None:
            return beta2m

        if not require_species_for_class_i_b2m:
            return self.beta2m_by_species["human"]

        raise ValueError(
            "Class I prediction requires beta2m context. Provide `mhc_b_sequence`, "
            "or pass a resolvable `allele`, or set `species`."
        )

    def _tokenize(self, seq: str, max_len: int) -> torch.Tensor:
        """Tokenize a sequence."""
        return self.tokenizer.batch_encode([seq], max_len=max_len, pad=True).to(self.device)

    def _binding_prob_from_kd_log10(self, kd_log10_nM: float) -> float:
        """Map log10(KD nM) to a calibrated [0,1] binding probability.

        Lower KD => higher probability. Values in the 50k-100k nM range map near 0.
        """
        return float(
            binding_prob_from_kd_log10(
                float(kd_log10_nM),
                midpoint_nM=self.binding_midpoint_nM,
                log10_scale=self.binding_log10_scale,
            )
        )

    @staticmethod
    def _label_from_class_probs(class_probs_row: torch.Tensor) -> str:
        """Convert a [pI, pII] row into a hard label for reporting."""
        return "I" if float(class_probs_row[0].item()) >= float(class_probs_row[1].item()) else "II"

    @torch.no_grad()
    def predict_presentation(
        self,
        peptide: str,
        allele: str = None,
        mhc_sequence: str = None,
        mhc_b_sequence: str = None,
        mhc_class: str = None,
        species: str = None,
        mhc_species: str = None,
        immune_species: str = None,
        species_of_origin: str = None,
        flank_n: str = None,
        flank_c: str = None,
        require_species_for_class_i_b2m: bool = True,
    ) -> PresentationResult:
        """Predict presentation probability for a peptide-MHC pair.

        Args:
            peptide: Peptide sequence
            allele: MHC allele name (e.g., "HLA-A*02:01")
            mhc_sequence: MHC alpha chain sequence (alternative to allele)
            mhc_b_sequence: MHC beta chain sequence (beta2m for Class I)
            mhc_class: Optional hard class override, "I"/"II"
            species: Species label used to pick class-I beta2m when needed
            mhc_species: Optional override for MHC species latent path
            immune_species: Optional override for host immune-system context
            species_of_origin: Optional override for peptide source organism latent
            flank_n: N-terminal processing flank
            flank_c: C-terminal processing flank
            require_species_for_class_i_b2m: If True, raise when class-I beta2m
                cannot be resolved from `mhc_b_sequence`, `allele`, or `species`.

        Returns:
            PresentationResult with probabilities
        """
        explicit_mhc_class = normalize_mhc_class(mhc_class)
        class_for_chain_resolution = explicit_mhc_class or self._infer_mhc_class(allele)
        model_class_input = explicit_mhc_class

        mhc_a_seq = self._get_mhc_sequence(allele, mhc_sequence)
        mhc_b_seq = self._resolve_class_i_beta2m(
            mhc_class=class_for_chain_resolution,
            mhc_b_sequence=mhc_b_sequence,
            allele=allele,
            species=species or immune_species or mhc_species,
            require_species_for_class_i_b2m=require_species_for_class_i_b2m,
        )

        # Tokenize
        pep_tok = self._tokenize(peptide, max_len=max(50, len(peptide), 1))
        mhc_a_tok = self._tokenize(mhc_a_seq, max_len=400)
        mhc_b_tok = self._tokenize(mhc_b_seq, max_len=400)
        flank_n_tok = self._tokenize(flank_n, max_len=30) if flank_n else None
        flank_c_tok = self._tokenize(flank_c, max_len=30) if flank_c else None

        # Forward pass
        outputs = self.model(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            mhc_class=model_class_input,
            species=species,
            mhc_species=mhc_species,
            immune_species=immune_species,
            species_of_origin=species_of_origin,
            flank_n_tok=flank_n_tok,
            flank_c_tok=flank_c_tok,
        )

        # Extract results
        proc_prob = torch.sigmoid(outputs["processing_logit"]).item()
        bind_prob = None
        if "assays" in outputs and isinstance(outputs["assays"], dict) and "KD_nM" in outputs["assays"]:
            kd_log10 = outputs["assays"]["KD_nM"].item()
            bind_prob = self._binding_prob_from_kd_log10(kd_log10)
        else:
            bind_prob = torch.sigmoid(outputs["binding_logit"]).item()
        pres_prob = torch.sigmoid(outputs["presentation_logit"]).item()

        latents = {
            k: v.item() for k, v in outputs["binding_latents"].items()
        }
        assays = {
            k: v.item() for k, v in outputs["assays"].items()
        }

        if explicit_mhc_class is not None:
            reported_mhc_class = explicit_mhc_class
        else:
            inferred_probs = outputs.get("mhc_class_probs")
            if (
                isinstance(inferred_probs, torch.Tensor)
                and inferred_probs.ndim == 2
                and inferred_probs.shape[-1] == 2
            ):
                reported_mhc_class = self._label_from_class_probs(inferred_probs[0])
            else:
                reported_mhc_class = class_for_chain_resolution

        return PresentationResult(
            peptide=peptide,
            mhc_class=reported_mhc_class,
            processing_prob=proc_prob,
            binding_prob=bind_prob,
            presentation_prob=pres_prob,
            binding_latents=latents,
            assays=assays,
        )

    @torch.no_grad()
    def predict_tiled_presentation(
        self,
        protein_sequence: str,
        allele: str = None,
        mhc_sequence: str = None,
        mhc_b_sequence: str = None,
        mhc_class: str = None,
        species: str = None,
        mhc_species: str = None,
        immune_species: str = None,
        species_of_origin: str = None,
        min_length: int = 8,
        max_length: int = 15,
        flank_size: int = 15,
        batch_size: int = 128,
        top_k: int = 100,
        sort_by: str = "presentation",
        require_species_for_class_i_b2m: bool = True,
    ) -> TiledPresentationResult:
        """Tile presentation predictions across all subsequences of a protein.

        Override controls:
            mhc_species: Optional override for MHC species latent path
            immune_species: Optional override for host immune-system context
            species_of_origin: Optional override for peptide source organism latent
        """
        sequence = (protein_sequence or "").strip().upper()
        if not sequence:
            raise ValueError("protein_sequence must be non-empty")
        if min_length < 1 or max_length < min_length:
            raise ValueError(
                f"Invalid tiling length range: min_length={min_length}, max_length={max_length}"
            )
        if flank_size < 0:
            raise ValueError("flank_size must be >= 0")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if sort_by not in {"presentation", "binding", "processing"}:
            raise ValueError("sort_by must be one of {'presentation', 'binding', 'processing'}")

        explicit_mhc_class = normalize_mhc_class(mhc_class)
        class_for_chain_resolution = explicit_mhc_class or self._infer_mhc_class(allele)
        model_class_input = explicit_mhc_class
        reported_mhc_class = explicit_mhc_class
        fallback_mhc_class = reported_mhc_class or class_for_chain_resolution

        mhc_a_seq = self._get_mhc_sequence(allele, mhc_sequence)
        mhc_b_seq = self._resolve_class_i_beta2m(
            mhc_class=class_for_chain_resolution,
            mhc_b_sequence=mhc_b_sequence,
            allele=allele,
            species=species or immune_species or mhc_species,
            require_species_for_class_i_b2m=require_species_for_class_i_b2m,
        )

        tiles: List[Dict[str, Union[str, int]]] = []
        seq_len = len(sequence)
        for start in range(seq_len):
            max_here = min(max_length, seq_len - start)
            for length in range(min_length, max_here + 1):
                end = start + length
                flank_n = sequence[max(0, start - flank_size):start] if flank_size > 0 else ""
                flank_c = sequence[end:min(seq_len, end + flank_size)] if flank_size > 0 else ""
                tiles.append(
                    {
                        "peptide": sequence[start:end],
                        "start": start,
                        "end": end,
                        "flank_n": flank_n,
                        "flank_c": flank_c,
                    }
                )

        if not tiles:
            return TiledPresentationResult(
                protein_length=seq_len,
                total_candidates=0,
                min_length=min_length,
                max_length=max_length,
                flank_size=flank_size,
                mhc_class=fallback_mhc_class,
                sort_by=sort_by,
                hits=[],
            )

        mhc_a_tok_single = self._tokenize(mhc_a_seq, max_len=400)
        mhc_b_tok_single = self._tokenize(mhc_b_seq, max_len=400)

        hits: List[TiledPresentationHit] = []
        max_pep_len = max(max_length, 1)
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i : i + batch_size]
            pep_tok = self.tokenizer.batch_encode(
                [str(tile["peptide"]) for tile in batch_tiles],
                max_len=max_pep_len,
                pad=True,
            ).to(self.device)

            flank_n_list = [str(tile["flank_n"]) for tile in batch_tiles]
            flank_c_list = [str(tile["flank_c"]) for tile in batch_tiles]
            flank_n_tok = None
            flank_c_tok = None
            if any(flank_n_list):
                flank_n_tok = self.tokenizer.batch_encode(flank_n_list, max_len=30, pad=True).to(self.device)
            if any(flank_c_list):
                flank_c_tok = self.tokenizer.batch_encode(flank_c_list, max_len=30, pad=True).to(self.device)

            mhc_a_tok = mhc_a_tok_single.expand(pep_tok.shape[0], -1)
            mhc_b_tok = mhc_b_tok_single.expand(pep_tok.shape[0], -1)

            outputs = self.model(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok,
                mhc_b_tok=mhc_b_tok,
                mhc_class=model_class_input,
                species=species,
                mhc_species=mhc_species,
                immune_species=immune_species,
                species_of_origin=species_of_origin,
                flank_n_tok=flank_n_tok,
                flank_c_tok=flank_c_tok,
            )

            if reported_mhc_class is None:
                inferred_probs = outputs.get("mhc_class_probs")
                if (
                    isinstance(inferred_probs, torch.Tensor)
                    and inferred_probs.ndim == 2
                    and inferred_probs.shape[-1] == 2
                ):
                    reported_mhc_class = self._label_from_class_probs(inferred_probs[0])

            proc_probs = torch.sigmoid(outputs["processing_logit"]).view(-1).tolist()
            pres_probs = torch.sigmoid(outputs["presentation_logit"]).view(-1).tolist()

            if "assays" in outputs and isinstance(outputs["assays"], dict) and "KD_nM" in outputs["assays"]:
                kd_log10 = outputs["assays"]["KD_nM"].view(-1)
                bind_probs = [
                    self._binding_prob_from_kd_log10(float(v.item())) for v in kd_log10
                ]
            else:
                bind_probs = torch.sigmoid(outputs["binding_logit"]).view(-1).tolist()

            assays_by_name: Dict[str, List[float]] = {}
            for assay_name, assay_tensor in outputs.get("assays", {}).items():
                assays_by_name[assay_name] = assay_tensor.view(-1).tolist()

            for idx, tile in enumerate(batch_tiles):
                assays = {
                    assay_name: float(values[idx])
                    for assay_name, values in assays_by_name.items()
                }
                hits.append(
                    TiledPresentationHit(
                        peptide=str(tile["peptide"]),
                        start=int(tile["start"]),
                        end=int(tile["end"]),
                        flank_n=str(tile["flank_n"]),
                        flank_c=str(tile["flank_c"]),
                        processing_prob=float(proc_probs[idx]),
                        binding_prob=float(bind_probs[idx]),
                        presentation_prob=float(pres_probs[idx]),
                        assays=assays,
                    )
                )

        if sort_by == "binding":
            hits.sort(key=lambda item: item.binding_prob, reverse=True)
        elif sort_by == "processing":
            hits.sort(key=lambda item: item.processing_prob, reverse=True)
        else:
            hits.sort(key=lambda item: item.presentation_prob, reverse=True)

        if top_k > 0:
            hits = hits[:top_k]

        return TiledPresentationResult(
            protein_length=seq_len,
            total_candidates=len(tiles),
            min_length=min_length,
            max_length=max_length,
            flank_size=flank_size,
            mhc_class=reported_mhc_class or fallback_mhc_class,
            sort_by=sort_by,
            hits=hits,
        )

    @torch.no_grad()
    def predict_presentation_multi_allele(
        self,
        peptide: str,
        alleles: List[str] = None,
        mhc_sequences: List[str] = None,
        mhc_class: Optional[str] = None,
        species: str = None,
        mhc_species: str = None,
        immune_species: str = None,
        species_of_origin: str = None,
        require_species_for_class_i_b2m: bool = True,
    ) -> Dict[str, Any]:
        """Predict presentation with multiple alleles (MS/EL scenario).

        Uses Noisy-OR aggregation over alleles.

        Args:
            peptide: Peptide sequence
            alleles: List of MHC allele names
            mhc_sequences: List of MHC sequences (alternative)
            mhc_class: "I" or "II"
            species: Species label used for class-I beta2m defaults
            mhc_species: Optional override for MHC species latent path
            immune_species: Optional override for host immune-system context
            species_of_origin: Optional override for peptide source organism latent
            require_species_for_class_i_b2m: Enforce explicit class-I beta2m resolution

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
            allele = alleles[i] if i < len(alleles) else None
            result = self.predict_presentation(
                peptide=peptide,
                allele=allele,
                mhc_sequence=seq,
                mhc_class=mhc_class,
                species=species,
                mhc_species=mhc_species,
                immune_species=immune_species,
                species_of_origin=species_of_origin,
                require_species_for_class_i_b2m=require_species_for_class_i_b2m,
            )
            per_allele.append({
                "allele": allele if allele is not None else f"seq_{i}",
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
        mhc_b_sequence: str = None,
        tcr_alpha: str = None,
        tcr_beta: str = None,
        mhc_class: str = None,
        species: str = None,
        require_species_for_class_i_b2m: bool = True,
    ) -> RecognitionResult:
        """Predict TCR-pMHC recognition.

        Args:
            peptide: Peptide sequence
            allele: MHC allele name
            mhc_sequence: MHC sequence (alternative)
            mhc_b_sequence: MHC beta chain sequence (beta2m for Class I)
            tcr_alpha: TCR alpha chain sequence
            tcr_beta: TCR beta chain sequence
            mhc_class: "I" or "II"
            species: Species label used to pick class-I beta2m when needed
            require_species_for_class_i_b2m: If True, require class-I beta2m resolution

        Returns:
            RecognitionResult with probabilities
        """
        raise NotImplementedError(
            "TCR-conditioned recognition is a future feature and is disabled in "
            "canonical Presto inference."
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
        mhc_b_sequence: str = None,
        mhc_class: str = None,
        species: str = None,
        mhc_species: str = None,
        immune_species: str = None,
        species_of_origin: str = None,
        require_species_for_class_i_b2m: bool = True,
    ) -> torch.Tensor:
        """Get pMHC embedding.

        Args:
            peptide: Peptide sequence
            allele: MHC allele name
            mhc_sequence: MHC sequence
            mhc_b_sequence: MHC beta chain sequence (beta2m for Class I)
            mhc_class: "I" or "II"
            species: Species label used to pick class-I beta2m when needed
            mhc_species: Optional override for MHC species latent path
            immune_species: Optional override for host immune-system context
            species_of_origin: Optional override for peptide source organism latent
            require_species_for_class_i_b2m: If True, require class-I beta2m resolution

        Returns:
            Embedding tensor (1, d_model)
        """
        mhc_class = normalize_mhc_class(mhc_class) or self._infer_mhc_class(allele)
        mhc_a_seq = self._get_mhc_sequence(allele, mhc_sequence)
        mhc_b_seq = self._resolve_class_i_beta2m(
            mhc_class=mhc_class,
            mhc_b_sequence=mhc_b_sequence,
            allele=allele,
            species=species or immune_species or mhc_species,
            require_species_for_class_i_b2m=require_species_for_class_i_b2m,
        )

        pep_tok = self._tokenize(peptide, max_len=max(50, len(peptide), 1))
        mhc_a_tok = self._tokenize(mhc_a_seq, max_len=400)
        mhc_b_tok = self._tokenize(mhc_b_seq, max_len=400)

        outputs = self.model(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            mhc_class=mhc_class,
            species=species,
            mhc_species=mhc_species,
            immune_species=immune_species,
            species_of_origin=species_of_origin,
        )
        return outputs["pmhc_vec"]

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
