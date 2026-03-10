"""Task definitions for Presto multi-task training.

Each task specifies:
- required_fields: What data fields the sample must have
- derived_label: How to compute label from sample (if not explicit)
- negative_sampling: How to generate negatives (for positive-only data)
- loss computation and metrics

Inspired by:
- T5/SeqIO task registry (Google)
- Fairseq task registration (Meta)
- Multi-task learning best practices

Usage:
    from presto.training.tasks import TASK_REGISTRY, get_task, route_sample

    # Get task by name
    task = get_task("binding")

    # Check if sample can be used for task
    if task.accepts(sample):
        loss = task.compute_loss(model_output, sample)

    # Route sample to all applicable tasks
    tasks = route_sample(sample)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from presto.data.allele_resolver import (
    infer_gene,
    infer_mhc_class_optional,
    infer_species_identity,
    normalize_mhc_class,
    parse_allele_name,
)


# =============================================================================
# MHC/Species Parsing (using mhcgnomes as the canonical parser)
# =============================================================================

def parse_mhc_allele(allele: str) -> Dict[str, Any]:
    """Parse an MHC allele using mhcgnomes.

    Returns:
        Dict with keys: gene, species, mhc_class, chain_type_id
    """
    result = parse_allele_name(allele) if allele else None
    gene = getattr(result.gene, "name", None) if getattr(result, "gene", None) else None
    species = (
        getattr(result.species, "name", None) if getattr(result, "species", None) else None
    )
    mhc_class = normalize_mhc_class(getattr(result, "mhc_class", None), default=None)
    if gene is None and allele:
        gene = infer_gene(allele)
    if species is None and allele:
        species = infer_species_identity(allele)
    if mhc_class is None and allele:
        mhc_class = infer_mhc_class_optional(allele)
    return {
        "gene": gene,
        "species": species,
        "mhc_class": mhc_class,
        "chain_type_id": _gene_to_chain_type_id(gene, allele),
        "species_id": _species_to_id(species, allele),
    }


def _gene_to_chain_type_id(gene: Optional[str], allele: str) -> int:
    """Map MHC gene name to chain type ID."""
    # Map gene names to IDs (0-11)
    gene_map = {
        'A': 0, 'B': 1, 'C': 2, 'E': 3, 'F': 4, 'G': 5,  # Class I alpha
        'DRA': 6, 'DQA1': 7, 'DPA1': 8,  # Class II alpha
        'DRB': 9, 'DQB1': 10, 'DPB1': 11,  # Class II beta
        # Normalize numbered variants
        'DRB1': 9, 'DRB3': 9, 'DRB4': 9, 'DRB5': 9,
        'DQA': 7, 'DQB': 10, 'DPA': 8, 'DPB': 11,
        # Mouse
        'K': 0, 'D': 1, 'L': 2,  # Map to Class I slots
    }

    if gene and gene in gene_map:
        return gene_map[gene]

    return 0  # Default to HLA-A


def _species_to_id(species: Optional[str], allele: str) -> int:
    """Map species name to ID."""
    if species:
        species_lower = species.lower()
        if 'homo' in species_lower or 'human' in species_lower:
            return 0
        elif 'mus' in species_lower or 'mouse' in species_lower:
            return 1
        elif 'macaca' in species_lower or 'macaque' in species_lower:
            return 2
        elif 'rat' in species_lower:
            return 3

    return 0  # Default to human


# Canonical label maps used by task helpers and tests.
MHC_CHAIN_TYPE_MAP = {
    'HLA-A': 0, 'HLA-B': 1, 'HLA-C': 2, 'HLA-E': 3, 'HLA-F': 4, 'HLA-G': 5,
    'HLA-DRA': 6, 'HLA-DQA': 7, 'HLA-DPA': 8,
    'HLA-DRB': 9, 'HLA-DQB': 10, 'HLA-DPB': 11,
    'H-2-K': 0, 'H-2-D': 1, 'H-2-L': 2,
    'H-2-IA': 6, 'H-2-IE': 9,
}

SPECIES_MAP = {
    'human': 0, 'homo_sapiens': 0, 'HLA': 0,
    'mouse': 1, 'mus_musculus': 1, 'H-2': 1,
    'macaque': 2, 'mamu': 2, 'Mamu': 2,
    'rat': 3,
}


# =============================================================================
# Common Negative Generation Helper
# =============================================================================

def generate_shuffled_negatives(
    positives: List[Dict[str, Any]],
    all_samples: List[Dict[str, Any]],
    field_a: str,
    field_b: str,
    label_field: str,
    n_per_positive: int = 1,
) -> List[Dict[str, Any]]:
    """Generate negatives by shuffling two fields (e.g., chain pairing).

    Args:
        positives: Positive samples
        all_samples: All available samples (for shuffling pool)
        field_a: First field to potentially shuffle
        field_b: Second field to potentially shuffle
        label_field: Field to set to 0 in negatives
        n_per_positive: Number of negatives per positive

    Returns:
        List of negative samples
    """
    negatives = []
    pool_a = [s[field_a] for s in all_samples if field_a in s]
    pool_b = [s[field_b] for s in all_samples if field_b in s]

    for pos in positives:
        for _ in range(n_per_positive):
            neg = pos.copy()
            # Randomly swap either field
            if random.random() < 0.5 and pool_a:
                neg[field_a] = random.choice(pool_a)
            elif pool_b:
                neg[field_b] = random.choice(pool_b)
            neg[label_field] = 0
            negatives.append(neg)

    return negatives


def generate_random_negatives(
    positives: List[Dict[str, Any]],
    all_samples: List[Dict[str, Any]],
    swap_field: str,
    keep_fields: List[str],
    label_field: str,
    n_per_positive: int = 1,
) -> List[Dict[str, Any]]:
    """Generate negatives by replacing one field with random values.

    Args:
        positives: Positive samples
        all_samples: All available samples
        swap_field: Field to replace (e.g., 'pep_seq')
        keep_fields: Fields to keep from positive (e.g., ['mhc_allele'])
        label_field: Field to set to 0 in negatives
        n_per_positive: Number of negatives per positive

    Returns:
        List of negative samples
    """
    positive_values = {s[swap_field] for s in positives}
    all_values = list({s[swap_field] for s in all_samples if swap_field in s})
    candidates = [v for v in all_values if v not in positive_values] or all_values

    negatives = []
    for pos in positives:
        for _ in range(n_per_positive):
            neg = {swap_field: random.choice(candidates), label_field: 0}
            for keep in keep_fields:
                if keep in pos:
                    neg[keep] = pos[keep]
            negatives.append(neg)

    return negatives


# =============================================================================
# Task Base Class
# =============================================================================

@dataclass
class TaskSpec:
    """Specification for what a task needs from samples."""
    # Fields that MUST be present in sample
    required_fields: Set[str]
    # The field containing the label (if from data, not derived)
    label_field: Optional[str] = None
    # Does this task need synthetic negatives?
    needs_negatives: bool = False
    # Ratio of negatives to positives (if needs_negatives)
    negative_ratio: float = 1.0


class Task(ABC):
    """Base class for all tasks.

    Subclasses must implement:
    - name: unique identifier
    - spec: TaskSpec defining requirements
    - compute_loss: loss from model outputs and sample
    - compute_metrics: evaluation metrics

    May override:
    - derive_label: compute label from sample (for synthetic labels)
    - generate_negatives: create negative samples
    - preprocess: transform sample before model
    """
    name: str
    spec: TaskSpec

    def accepts(self, sample: Dict[str, Any]) -> bool:
        """Check if this task can use this sample."""
        sample_fields = set(sample.keys())
        return self.spec.required_fields.issubset(sample_fields)

    def derive_label(self, sample: Dict[str, Any]) -> Optional[Any]:
        """Derive label from sample if not explicit.

        Override for tasks with synthetic labels (e.g., chain type from allele).
        Returns None if label should come from sample[spec.label_field].
        """
        return None

    def get_label(self, sample: Dict[str, Any]) -> Any:
        """Get label for sample (derived or from data)."""
        derived = self.derive_label(sample)
        if derived is not None:
            return derived
        if self.spec.label_field and self.spec.label_field in sample:
            return sample[self.spec.label_field]
        raise ValueError(f"Task {self.name}: no label found in sample")

    def generate_negatives(
        self,
        positive_samples: List[Dict[str, Any]],
        all_samples: List[Dict[str, Any]],
        n_per_positive: int = 1,
    ) -> List[Dict[str, Any]]:
        """Generate negative samples for positive-only data.

        Override for tasks that need synthetic negatives (e.g., elution, matching).
        """
        return []

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess sample before model forward pass.

        Override for task-specific preprocessing.
        """
        return sample

    @abstractmethod
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute task-specific loss.

        Args:
            outputs: Model outputs dict
            batch: Collated batch with targets
            mask: Optional mask for valid samples

        Returns:
            Scalar loss tensor
        """
        pass

    @abstractmethod
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute task-specific metrics.

        Returns:
            Dict of metric_name -> value
        """
        pass


# =============================================================================
# Stage 1: Chain Classification Tasks
# =============================================================================

# Receptor chain type labels
RECEPTOR_CHAIN_TYPE_MAP = {
    "TRAV": 0, "TRAJ": 1, "TRBV": 2, "TRBJ": 3, "TRBD": 4,
    "TRGV": 5, "TRGJ": 6, "TRDV": 7, "TRDJ": 8, "TRDD": 9,
    "IGHV": 10, "IGHJ": 11, "IGHD": 12,
    "IGKV": 13, "IGKJ": 14, "IGLV": 15, "IGLJ": 16,
}


class MHCChainTypeTask(Task):
    """Classify MHC chain type from sequence/allele.

    Uses mhcgnomes for robust parsing when available, with fallback.
    Labels: HLA-A→0, HLA-B→1, ..., HLA-DPB→11
    """
    name = "mhc_chain_type"
    spec = TaskSpec(
        required_fields={"mhc_allele"},
        label_field=None,  # derived
    )
    n_classes = 12

    def derive_label(self, sample: Dict[str, Any]) -> int:
        """Derive chain type from allele name using mhcgnomes."""
        allele = sample.get("mhc_allele", "")
        parsed = parse_mhc_allele(allele)
        return parsed['chain_type_id']

    def compute_loss(self, outputs, batch, mask=None):
        logits = outputs["mhc_chain_logits"]  # (B, n_classes)
        targets = batch["mhc_chain_type_label"]  # (B,)

        if targets.dim() == 0:
            targets = targets.unsqueeze(0)

        loss = F.cross_entropy(logits, targets, reduction='none')
        if mask is not None:
            return (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss.mean()

    def compute_metrics(self, outputs, batch):
        logits = outputs["mhc_chain_logits"]
        targets = batch["mhc_chain_type_label"]
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()
        return {"accuracy": acc}


class ReceptorChainTypeTask(Task):
    """Classify receptor chain type (TRAV, TRBV, etc.)."""
    name = "receptor_chain_type"
    spec = TaskSpec(
        required_fields={"tcr_v_gene"},
        label_field=None,  # derived
    )
    n_classes = 17

    def derive_label(self, sample: Dict[str, Any]) -> int:
        """Derive chain type from V gene name."""
        v_gene = sample.get("tcr_v_gene", "")
        for prefix, label in RECEPTOR_CHAIN_TYPE_MAP.items():
            if v_gene.startswith(prefix):
                return label
        return 0  # Default to TRAV

    def compute_loss(self, outputs, batch, mask=None):
        logits = outputs["receptor_chain_logits"]
        targets = batch["receptor_chain_type_label"]
        loss = F.cross_entropy(logits, targets, reduction='none')
        if mask is not None:
            return (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss.mean()

    def compute_metrics(self, outputs, batch):
        logits = outputs["receptor_chain_logits"]
        targets = batch["receptor_chain_type_label"]
        preds = logits.argmax(dim=-1)
        return {"accuracy": (preds == targets).float().mean().item()}


class SpeciesTask(Task):
    """Classify species from allele name using mhcgnomes."""
    name = "species"
    spec = TaskSpec(
        required_fields={"mhc_allele"},
        label_field=None,  # derived
    )
    n_classes = 4

    def derive_label(self, sample: Dict[str, Any]) -> int:
        """Derive species from allele using mhcgnomes."""
        allele = sample.get("mhc_allele", "")
        parsed = parse_mhc_allele(allele)
        return parsed['species_id']

    def compute_loss(self, outputs, batch, mask=None):
        logits = outputs["species_logits"]
        targets = batch["species_label"]
        loss = F.cross_entropy(logits, targets, reduction='none')
        if mask is not None:
            return (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss.mean()

    def compute_metrics(self, outputs, batch):
        logits = outputs["species_logits"]
        targets = batch["species_label"]
        preds = logits.argmax(dim=-1)
        return {"accuracy": (preds == targets).float().mean().item()}


# =============================================================================
# Stage 2: Pairing Tasks
# =============================================================================

# Valid MHC chain pairings (alpha, beta combinations)
MHC_VALID_PAIRINGS = {
    # Class II: alpha + beta
    ("HLA-DRA", "HLA-DRB"), ("HLA-DQA", "HLA-DQB"), ("HLA-DPA", "HLA-DPB"),
    ("H-2-IA", "H-2-IA"), ("H-2-IE", "H-2-IE"),
}


class MHCPairingTask(Task):
    """Predict if two MHC chains can form a valid complex.

    Positive examples: known pairings from haplotype data
    Negative examples: random mismatched chains (synthetic)
    """
    name = "mhc_pairing"
    spec = TaskSpec(
        required_fields={"mhc_a_seq", "mhc_b_seq"},
        label_field="pairing_label",
        needs_negatives=True,
        negative_ratio=1.0,
    )

    def derive_label(self, sample: Dict[str, Any]) -> Optional[int]:
        """Derive pairing label from allele names if available."""
        a_allele = sample.get("mhc_a_allele", "")
        b_allele = sample.get("mhc_b_allele", "")

        if not a_allele or not b_allele:
            return None  # Need explicit label

        # Use mhcgnomes to extract gene names
        a_parsed = parse_mhc_allele(a_allele)
        b_parsed = parse_mhc_allele(b_allele)
        a_gene = a_parsed.get('gene')
        b_gene = b_parsed.get('gene')

        # Fallback to prefix extraction if mhcgnomes didn't get gene
        def extract_gene_fallback(allele: str) -> str:
            gene = allele.split("*")[0] if "*" in allele else allele
            for prefix in ["HLA-DRA", "HLA-DRB", "HLA-DQA", "HLA-DQB",
                           "HLA-DPA", "HLA-DPB"]:
                if gene.startswith(prefix):
                    return prefix
            return gene

        if not a_gene:
            a_gene = extract_gene_fallback(a_allele)
        if not b_gene:
            b_gene = extract_gene_fallback(b_allele)

        # Normalize gene names to HLA- prefix format for pairing check
        def normalize_gene(gene: str, allele: str) -> str:
            if not gene:
                return allele.split("*")[0] if "*" in allele else allele

            # Strip HLA- prefix if present to normalize consistently
            gene_core = gene[4:] if gene.startswith("HLA-") else gene

            if gene_core in ["A", "B", "C", "E", "F", "G"]:
                return f"HLA-{gene_core}"

            # Normalize class II variants like DRB1 -> DRB, DQA1 -> DQA
            for prefix in ["DRA", "DRB", "DQA", "DQB", "DPA", "DPB"]:
                if gene_core.startswith(prefix):
                    return f"HLA-{prefix}"

            return allele.split("*")[0] if "*" in allele else allele

        a_norm = normalize_gene(str(a_gene), a_allele)
        b_norm = normalize_gene(str(b_gene), b_allele)

        # Class I: alpha only (no beta pairing needed)
        class1_genes = ["HLA-A", "HLA-B", "HLA-C", "H-2-K", "H-2-D"]
        if any(a_norm.startswith(p) for p in class1_genes):
            return 1 if b_allele == "B2M" or not b_allele else 0

        # Class II: check valid alpha-beta combinations
        return 1 if (a_norm, b_norm) in MHC_VALID_PAIRINGS else 0

    def generate_negatives(self, positives, all_samples, n_per_positive=1):
        """Generate negative pairings by shuffling chains."""
        return generate_shuffled_negatives(
            positives, all_samples,
            field_a="mhc_a_seq", field_b="mhc_b_seq",
            label_field="pairing_label",
            n_per_positive=n_per_positive,
        )

    def compute_loss(self, outputs, batch, mask=None):
        logits = outputs["mhc_pairing_logit"].squeeze(-1)
        targets = batch["pairing_label"].float()
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        if mask is not None:
            return (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss.mean()

    def compute_metrics(self, outputs, batch):
        probs = torch.sigmoid(outputs["mhc_pairing_logit"].squeeze(-1))
        targets = batch["pairing_label"]
        preds = (probs > 0.5).long()
        acc = (preds == targets).float().mean().item()

        try:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(
                targets.detach().cpu().numpy(),
                probs.detach().cpu().numpy()
            )
        except Exception:
            auroc = 0.5

        return {"accuracy": acc, "auroc": auroc}


# =============================================================================
# Stage 3: Binding Tasks
# =============================================================================

class BindingTask(Task):
    """Predict peptide-MHC binding affinity (KD, IC50).

    Uses censor-aware loss for inequality measurements (<500 nM, >10000 nM).
    """
    name = "binding"
    spec = TaskSpec(
        required_fields={"pep_seq", "mhc_allele", "bind_value"},
        label_field="bind_value",
    )

    def compute_loss(self, outputs, batch, mask=None):
        """Censor-aware MSE loss."""
        pred = outputs["binding_pred"].squeeze(-1)
        target = batch["bind_target"].squeeze(-1)
        qual = batch.get("bind_qual")

        if qual is None:
            # Standard MSE
            loss = F.mse_loss(pred, target, reduction='none')
        else:
            qual = qual.squeeze(-1)
            diff = pred - target
            loss = torch.zeros_like(diff)

            # For "=" measurements: standard MSE
            eq_mask = (qual == 0)
            loss[eq_mask] = diff[eq_mask] ** 2

            # For "<" measurements (qual=-1): only penalize if pred > target
            lt_mask = (qual == -1)
            loss[lt_mask] = F.relu(diff[lt_mask]) ** 2

            # For ">" measurements (qual=1): only penalize if pred < target
            gt_mask = (qual == 1)
            loss[gt_mask] = F.relu(-diff[gt_mask]) ** 2

        if mask is not None:
            return (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss.mean()

    def compute_metrics(self, outputs, batch):
        pred = outputs["binding_pred"].squeeze(-1).detach().cpu().numpy()
        target = batch["bind_target"].squeeze(-1).detach().cpu().numpy()

        from scipy.stats import spearmanr
        try:
            rho, _ = spearmanr(pred, target)
        except Exception:
            rho = 0.0

        mae = abs(pred - target).mean()
        return {"spearman": rho, "mae": mae}


class ElutionTask(Task):
    """Predict MS/elution detection (was peptide found on cell surface?).

    Positive-only data: need to generate random peptide negatives.
    """
    name = "elution"
    spec = TaskSpec(
        required_fields={"pep_seq", "mhc_allele"},
        label_field="elution_label",
        needs_negatives=True,
        negative_ratio=5.0,  # More negatives for positive-only
    )

    def generate_negatives(self, positives, all_samples, n_per_positive=5):
        """Generate negatives with random peptides not in positive set."""
        return generate_random_negatives(
            positives, all_samples,
            swap_field="pep_seq",
            keep_fields=["mhc_allele"],
            label_field="elution_label",
            n_per_positive=n_per_positive,
        )

    def compute_loss(self, outputs, batch, mask=None):
        logits = outputs["elution_logit"].squeeze(-1)
        targets = batch["elution_label"].float()
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        if mask is not None:
            return (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss.mean()

    def compute_metrics(self, outputs, batch):
        logits = outputs["elution_logit"].squeeze(-1)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets = batch["elution_label"].detach().cpu().numpy()

        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            auroc = roc_auc_score(targets, probs)
            auprc = average_precision_score(targets, probs)
        except Exception:
            auroc, auprc = 0.5, 0.5

        return {"auroc": auroc, "auprc": auprc}


class ProcessingTask(Task):
    """Predict antigen processing (proteasome cleavage, TAP transport)."""
    name = "processing"
    spec = TaskSpec(
        required_fields={"pep_seq", "flank_n", "flank_c"},
        label_field="processing_label",
    )

    def compute_loss(self, outputs, batch, mask=None):
        logits = outputs["processing_logit"].squeeze(-1)
        targets = batch["processing_label"].float()
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        if mask is not None:
            return (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss.mean()

    def compute_metrics(self, outputs, batch):
        probs = torch.sigmoid(outputs["processing_logit"].squeeze(-1))
        preds = (probs > 0.5).long()
        targets = batch["processing_label"]
        return {"accuracy": (preds == targets).float().mean().item()}


class StabilityTask(Task):
    """Predict pMHC stability (t1/2, Tm)."""
    name = "stability"
    spec = TaskSpec(
        required_fields={"pep_seq", "mhc_allele", "stability_value"},
        label_field="stability_value",
    )

    def compute_loss(self, outputs, batch, mask=None):
        pred = outputs["stability_pred"].squeeze(-1)
        target = batch["stability_target"].squeeze(-1)
        loss = F.mse_loss(pred, target, reduction='none')
        if mask is not None:
            return (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss.mean()

    def compute_metrics(self, outputs, batch):
        pred = outputs["stability_pred"].squeeze(-1).detach().cpu().numpy()
        target = batch["stability_target"].squeeze(-1).detach().cpu().numpy()

        from scipy.stats import spearmanr
        try:
            rho, _ = spearmanr(pred, target)
        except Exception:
            rho = 0.0

        return {"spearman": rho}


class TcrEvidenceTask(Task):
    """Predict pMHC-only curated cognate-TCR evidence."""
    name = "tcr_evidence"
    spec = TaskSpec(
        required_fields={"pep_seq", "mhc_allele"},
        label_field="tcr_evidence_label",
    )

    def compute_loss(self, outputs, batch, mask=None):
        logits = outputs["tcr_evidence_logit"].squeeze(-1)
        targets = batch["tcr_evidence_label"].float()
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if mask is not None:
            return (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss.mean()

    def compute_metrics(self, outputs, batch):
        probs = torch.sigmoid(outputs["tcr_evidence_logit"].squeeze(-1)).detach().cpu().numpy()
        targets = batch["tcr_evidence_label"].detach().cpu().numpy()
        try:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(targets, probs)
        except Exception:
            auroc = 0.5
        return {"auroc": auroc}


class ImmunogenicityTask(Task):
    """Predict if pMHC elicits T-cell response (without specific TCR)."""
    name = "immunogenicity"
    spec = TaskSpec(
        required_fields={"pep_seq", "mhc_allele"},
        label_field="immunogenicity_label",
        needs_negatives=True,
        negative_ratio=1.0,
    )

    def generate_negatives(self, positives, all_samples, n_per_positive=1):
        """Generate negatives from binding data that didn't elicit response."""
        return generate_random_negatives(
            positives, all_samples,
            swap_field="pep_seq",
            keep_fields=["mhc_allele"],
            label_field="immunogenicity_label",
            n_per_positive=n_per_positive,
        )

    def compute_loss(self, outputs, batch, mask=None):
        logits = outputs["immunogenicity_logit"].squeeze(-1)
        targets = batch["immunogenicity_label"].float()
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        if mask is not None:
            return (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss.mean()

    def compute_metrics(self, outputs, batch):
        logits = outputs["immunogenicity_logit"].squeeze(-1)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets = batch["immunogenicity_label"].detach().cpu().numpy()

        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            auroc = roc_auc_score(targets, probs)
            auprc = average_precision_score(targets, probs)
        except Exception:
            auroc, auprc = 0.5, 0.5

        return {"auroc": auroc, "auprc": auprc}


class TcellAssayTask(Task):
    """Predict T-cell assay outcome (IFNg, cytotoxicity, etc.)."""
    name = "tcell_assay"
    spec = TaskSpec(
        required_fields={"pep_seq", "mhc_allele", "tcell_label"},
        label_field="tcell_label",
    )

    def compute_loss(self, outputs, batch, mask=None):
        logits = outputs["tcell_logit"].squeeze(-1)
        targets = batch["tcell_label"].float()
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        if mask is not None:
            return (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss.mean()

    def compute_metrics(self, outputs, batch):
        probs = torch.sigmoid(outputs["tcell_logit"].squeeze(-1)).detach().cpu().numpy()
        targets = batch["tcell_label"].detach().cpu().numpy()

        try:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(targets, probs)
        except Exception:
            auroc = 0.5

        return {"auroc": auroc}


# =============================================================================
# Task Registry
# =============================================================================

TASK_REGISTRY: Dict[str, Task] = {}


def register_task(task: Task) -> Task:
    """Register a task in the global registry."""
    TASK_REGISTRY[task.name] = task
    return task


def get_task(name: str) -> Task:
    """Get task by name from registry."""
    if name not in TASK_REGISTRY:
        available = list(TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task: {name}. Available: {available}")
    return TASK_REGISTRY[name]


def route_sample(sample: Dict[str, Any]) -> List[str]:
    """Find all tasks that can use this sample.

    Returns list of task names whose required_fields are satisfied.
    """
    applicable = []
    for name, task in TASK_REGISTRY.items():
        if task.accepts(sample):
            applicable.append(name)
    return applicable


def _register_default_tasks():
    """Register all default tasks."""
    # Stage 1: Chain Classification
    register_task(MHCChainTypeTask())
    register_task(SpeciesTask())

    # Stage 2: Pairing
    register_task(MHCPairingTask())

    # Stage 3: Binding
    register_task(BindingTask())
    register_task(ElutionTask())
    register_task(ProcessingTask())
    register_task(StabilityTask())

    # Stage 4: Matching
    register_task(TcrEvidenceTask())
    register_task(ImmunogenicityTask())
    register_task(TcellAssayTask())


# Register tasks on module import
_register_default_tasks()


# =============================================================================
# Task Balancing (from multi-task learning literature)
# =============================================================================

class TaskBalancer(ABC):
    """Base class for task loss balancing strategies."""

    @abstractmethod
    def get_weights(self, task_losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Return weight for each task."""
        pass


class StaticBalancer(TaskBalancer):
    """Static weights (from config)."""

    def __init__(self, weights: Dict[str, float]):
        self.weights = weights

    def get_weights(self, task_losses):
        return {name: self.weights.get(name, 1.0) for name in task_losses}


class UncertaintyBalancer(TaskBalancer):
    """Learn weights from homoscedastic uncertainty (Kendall et al. 2018)."""

    def __init__(self, task_names: List[str], device="cpu"):
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, device=device))
            for name in task_names
        })

    def get_weights(self, task_losses):
        return {
            name: torch.exp(-self.log_vars[name]).item()
            for name in task_losses
        }

    def compute_total_loss(self, task_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute uncertainty-weighted total loss."""
        total = torch.tensor(0.0, device=next(iter(task_losses.values())).device)
        for name, loss in task_losses.items():
            precision = torch.exp(-self.log_vars[name])
            total = total + precision * loss + self.log_vars[name]
        return total


class RandomBalancer(TaskBalancer):
    """Random loss weighting (Lin et al. - surprisingly competitive)."""

    def get_weights(self, task_losses):
        n = len(task_losses)
        weights = torch.softmax(torch.randn(n), dim=0) * n
        return {name: w.item() for name, w in zip(task_losses.keys(), weights)}


# =============================================================================
# Task Progress Tracking
# =============================================================================

class TaskTracker:
    """Track training progress per task."""

    def __init__(self, task_names: List[str]):
        self.task_names = task_names
        self.steps = {name: 0 for name in task_names}
        self.losses = {name: [] for name in task_names}
        self.metrics = {name: [] for name in task_names}

    def update(self, task_name: str, loss: float, metrics: Dict[str, float] = None):
        """Record a training step for a task."""
        self.steps[task_name] += 1
        self.losses[task_name].append(loss)
        if metrics:
            self.metrics[task_name].append(metrics)

    def get_least_trained(self, n: int = 1) -> List[str]:
        """Get tasks with fewest training steps."""
        sorted_tasks = sorted(self.steps.items(), key=lambda x: x[1])
        return [name for name, _ in sorted_tasks[:n]]

    def all_trained(self, min_steps: int = 1) -> bool:
        """Check if all tasks have been trained at least min_steps times."""
        return all(s >= min_steps for s in self.steps.values())

    def get_avg_loss(self, task_name: str, window: int = 100) -> float:
        """Get average recent loss for a task."""
        losses = self.losses[task_name]
        if not losses:
            return float('inf')
        return sum(losses[-window:]) / len(losses[-window:])

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "steps": self.steps.copy(),
            "avg_losses": {
                name: self.get_avg_loss(name)
                for name in self.task_names
            },
        }
